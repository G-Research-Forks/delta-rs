# Benchmarks

The merge benchmarks are similar to the ones used by [Delta Spark](https://github.com/delta-io/delta/pull/1835).


## Dataset

To generate the database, `duckdb` can be used. Install `duckdb` by following [these instructions](https://duckdb.org/#quickinstall).

Run the following commands:

```bash
❯ duckdb
D CALL dsdgen(sf = 1);
100% ▕██████████████████████████████████████▏ (00:00:05.76 elapsed)
┌─────────┐
│ Success │
│ boolean │
├─────────┤
│ 0 rows  │
└─────────┘
D EXPORT DATABASE 'tpcds_parquet' (FORMAT PARQUET);
```

This will generate a folder called `tpcds_parquet` containing many parquet files. Place it at `crates/benchmarks/data/tpcds_parquet` (or set `TPCDS_PARQUET_DIR`). Credits to [Xuanwo's Blog](https://xuanwo.io/links/2025/02/duckdb-is-the-best-tpc-data-generator/).

## Running benchmarks

Benchmarks use Divan and time only the merge operation. A temporary Delta table is created per iteration from `web_returns.parquet` and removed afterwards.

Environment variables:
- `TPCDS_PARQUET_DIR` (optional): directory containing `web_returns.parquet`. Default: `crates/benchmarks/data/tpcds_parquet`.

From the repo root:
```
cargo bench -p delta-benchmarks --bench merge
```

Filter a specific suite:
```
cargo bench -p delta-benchmarks --bench merge -- delete_only
cargo bench -p delta-benchmarks --bench merge -- multiple_insert_only
cargo bench -p delta-benchmarks --bench merge -- upsert_file_matched
```

## Profiling script

A simple CLI is available to run a single merge with configurable parameters (useful for profiling or ad-hoc runs). It creates a fresh temporary Delta table per sample from `web_returns.parquet`, times only the merge, and prints duration and metrics.

Run (from repo root):
```bash
cargo run --profile profiling -p delta-benchmarks -- merge --op upsert --matched 0.01 --not-matched 0.10
```

Options:
- `--op <upsert|delete|insert>`: operation to benchmark
- `--matched <fraction>`: fraction of rows that match existing keys (default 0.01)
- `--not-matched <fraction>`: fraction of rows that do not match (default 0.10)
- `--case <name>`: run one of the predefined merge scenarios mirrored from the Delta Spark suite

List cases with:
```bash
cargo run --release -p delta-benchmarks -- merge --case single_insert_only_filesMatchedFraction_0.05_rowsNotMatchedFraction_0.05
```

### Flamegraphs using `samply`

Using `samply`, you can generate flamegraphs from the profile script.

To start,

```bash
cargo install samply --locked
cargo build --profile profiling -p delta-benchmarks
samply record ./target/profiling/delta-benchmarks upsert
```

## Sorted streaming reads (sort-order evaluation)

Two subcommands evaluate the file sort order support on the DataFusion table
provider (`with_file_sort_order`): `sort-gen` generates a Delta table whose
files are all sorted by a `timestamp` column, and `sort-bench` measures
`ORDER BY timestamp` streaming queries over it with different provider
configurations.

### Generating test data

```bash
cargo run --release -p delta-benchmarks -- sort-gen --table-path ./data/sorted_table
```

Writes one commit per day, each containing a single batch sorted by
`timestamp`; day ranges do not overlap. Knobs:

- `--days <n>`: number of days to write (default 100)
- `--rows-per-day <n>`: rows per day (default 1,000,000)
- `--extra-columns <n>`: extra float32 data columns (default 20)

Note the defaults produce roughly 12 GB of parquet. The generator writes
through a single-partition DataFusion session: with the default multi-partition
session, the delta-rs write plan repartitions batches across concurrent writer
tasks and row order within the produced files is not preserved. Every file is
re-read after writing to verify it is internally sorted and that no file
ranges overlap, so generation fails loudly if that ever regresses.

### Benchmarking sorted reads

```bash
cargo run --release -p delta-benchmarks -- sort-bench --table-path ./data/sorted_table
```

For each mode the query `SELECT ... FROM t ORDER BY timestamp` is planned and
streamed to completion, reporting plan shape (`sort_exec`, `spm` =
SortPreservingMergeExec, `progressive_eval` = ProgressiveEvalExec), planning
time, time to first batch, total time, and — when `--check-order` is passed —
whether the streamed rows were actually in order (`sorted`).

- `--modes <baseline,declared,unordered,sequential-read,sequential-read-async>`:
  configurations to compare (default all). `baseline` declares no ordering and
  needs a full `SortExec`; `declared` uses `with_file_sort_order`, satisfying
  the ORDER BY at planning time — when the file ranges are mutually
  non-overlapping (as `sort-gen` produces), the scan partitions are contiguous
  range-ordered chunks and the merge is replaced by a `ProgressiveEvalExec`
  concatenation, otherwise a merge over parallel pre-grouped ordered
  partitions remains (this mode alone disables
  `datafusion.optimizer.repartition_file_scans`, whose byte-range splitting
  would otherwise defeat the concatenation; the other modes keep the
  DataFusion default); `unordered` drops the ORDER BY entirely, reading in arbitrary
  order with no sorting needed, as a lower bound for comparison;
  `sequential-read` bypasses the delta-rs scan and DataFusion entirely and
  reads the parquet files directly with the parquet crate, single-threaded and
  one file at a time in ascending timestamp order (representing production
  workloads that read parquet files in a known order — the output is still
  globally sorted because the files are sorted and non-overlapping);
  `sequential-read-async` is the same read through the parquet crate's async
  reader over tokio files — the IO pattern used by object-storage readers and
  DataFusion's parquet source — isolating the cost of the async read path from
  the rest of the stack. The sequential read modes honor `--select-columns`
  and `--limit` but ignore `--memory-limit-gb` and `--target-partitions`.
- `--select-columns <n>`: number of extra float32 columns in the SELECT
  (default: all)
- `--limit <n>`: add a LIMIT to exercise TopK / early termination
- `--iterations <n>`: runs per mode (default 3)
- `--memory-limit-gb <n>`: memory budget backed by a spill pool, so the
  baseline full sort spills instead of exhausting memory (default 0 = no
  limit)
- `--prefetch-streams <n>`: sets
  `delta.progressive_eval_num_prefetch_input_streams` — how many scan
  partitions `ProgressiveEvalExec` executes ahead of the one being streamed
  (default 1, minimum 0 = no read-ahead). Higher values
  overlap more read I/O with streaming at the cost of eagerly executing more
  partitions; only affects `declared` runs that plan the progressive-eval
  concatenation.
- `--target-partitions <n>`: override `datafusion.execution.target_partitions`
- `--check-order`: verify that the streamed timestamps are globally
  non-decreasing, reported as `sorted` with a description of the first
  violation. Off by default because the per-row check adds time to the
  measured run.
- `--show-plan`: print the physical plan for the first iteration of each mode
