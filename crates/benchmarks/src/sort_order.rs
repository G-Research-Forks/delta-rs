//! Test-data generation and read benchmarks for sorted streaming queries.
//!
//! Evaluates the sort-order work on the table provider: tables whose parquet
//! files are all sorted by a timestamp column can be scanned with a
//! `SortPreservingMergeExec` instead of a full `SortExec` when the ordering
//! is declared (`with_file_sort_order`).
//!
//! The generated table writes one commit per day, each containing a single
//! record batch sorted by `timestamp`, with `sorting_columns` metadata set on
//! every file. Day ranges never overlap, so file min/max statistics allow
//! DataFusion to form non-overlapping file groups.

use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::ValueEnum;
use deltalake_core::arrow::array::{ArrayRef, Float32Array, TimestampMicrosecondArray};
use deltalake_core::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use deltalake_core::arrow::record_batch::RecordBatch;
use deltalake_core::datafusion::catalog::Session;
use deltalake_core::datafusion::physical_plan::{displayable, execute_stream};
use deltalake_core::delta_datafusion::{
    create_session, DeltaRuntimeEnvBuilder, DeltaSessionContext, FileSortColumn,
};
use deltalake_core::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use deltalake_core::parquet::arrow::ProjectionMask;
use deltalake_core::parquet::basic::Compression;
use deltalake_core::parquet::file::metadata::SortingColumn;
use deltalake_core::parquet::file::properties::WriterProperties;
use deltalake_core::protocol::SaveMode;
use deltalake_core::{open_table, DeltaResult, DeltaTable, DeltaTableError};
use futures::StreamExt;
use url::Url;

const MICROS_PER_DAY: i64 = 86_400_000_000;
/// 2024-01-01T00:00:00 (naive) in seconds since the epoch.
const BASE_EPOCH_SECONDS: i64 = 1_704_067_200;
const TIMESTAMP_COLUMN: &str = "timestamp";

#[derive(Debug, Clone)]
pub struct SortDataParams {
    /// Number of days to write; one commit / batch per day.
    pub days: usize,
    /// Rows in each day's batch.
    pub rows_per_day: usize,
    /// Number of extra float32 data columns.
    pub extra_columns: usize,
}

impl Default for SortDataParams {
    fn default() -> Self {
        Self {
            days: 100,
            rows_per_day: 1_000_000,
            extra_columns: 20,
        }
    }
}

fn sort_table_schema(extra_columns: usize) -> SchemaRef {
    let mut fields = vec![Field::new(
        TIMESTAMP_COLUMN,
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )];
    for i in 0..extra_columns {
        fields.push(Field::new(format!("value_{i}"), DataType::Float32, false));
    }
    Arc::new(Schema::new(fields))
}

/// Writer properties declaring every file sorted ascending by the first
/// (timestamp) column, matching how the data is actually written.
fn sorting_writer_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_sorting_columns(Some(vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }]))
        .build()
}

/// Build one day's batch: timestamps spread evenly across the day (ascending),
/// plus `extra_columns` float32 columns of deterministic pseudo-random values.
fn day_batch(schema: SchemaRef, day: usize, params: &SortDataParams) -> DeltaResult<RecordBatch> {
    let rows = params.rows_per_day;
    let day_start_us = (BASE_EPOCH_SECONDS + day as i64 * 86_400) * 1_000_000;
    let step_us = MICROS_PER_DAY / rows as i64;
    let timestamps: Vec<i64> = (0..rows as i64)
        .map(|i| day_start_us + i * step_us)
        .collect();

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(params.extra_columns + 1);
    columns.push(Arc::new(TimestampMicrosecondArray::from(timestamps)));
    for col in 0..params.extra_columns {
        // xorshift32 seeded per (day, column) so runs are reproducible and the
        // data is incompressible enough to be a realistic I/O load.
        let mut state = (day as u32)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add((col as u32).wrapping_mul(0x85EB_CA6B))
            | 1;
        let values: Vec<f32> = (0..rows)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        columns.push(Arc::new(Float32Array::from(values)));
    }
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Write the test table: one commit per day, every file sorted by `timestamp`
/// with `sorting_columns` metadata, day ranges non-overlapping.
///
/// The first day is written with `SaveMode::Overwrite`, so re-running replaces
/// the logical content of an existing table at the same location.
///
/// Writes go through a single-partition DataFusion session: with the default
/// multi-partition session the write plan repartitions batches across
/// concurrent writer tasks and the row order within the produced files is NOT
/// preserved (observed as 8192-row chunks landing out of order). After
/// writing, every live file is re-read to verify it is internally sorted and
/// that no two files have overlapping timestamp ranges.
pub async fn generate_sorted_table(table_url: &Url, params: &SortDataParams) -> DeltaResult<()> {
    let schema = sort_table_schema(params.extra_columns);
    let mut table = DeltaTable::try_from_url(table_url.clone()).await?;
    let start = Instant::now();

    let write_ctx = create_session().into_inner();
    write_ctx
        .sql("SET datafusion.execution.target_partitions = 1")
        .await?
        .collect()
        .await?;
    let write_session: Arc<dyn Session> = Arc::new(write_ctx.state());

    for day in 0..params.days {
        let gen_start = Instant::now();
        let batch = day_batch(schema.clone(), day, params)?;
        let gen_elapsed = gen_start.elapsed();

        let mode = if day == 0 {
            SaveMode::Overwrite
        } else {
            SaveMode::Append
        };
        let write_start = Instant::now();
        table = table
            .write(vec![batch])
            .with_save_mode(mode)
            .with_session_state(write_session.clone())
            .with_writer_properties(sorting_writer_properties())
            .await?;
        println!(
            "day={day} rows={} gen_ms={} write_ms={}",
            params.rows_per_day,
            gen_elapsed.as_millis(),
            write_start.elapsed().as_millis()
        );
    }

    let files = table.snapshot()?.log_data().num_files();
    println!(
        "written days={} rows={} files={files} total_s={:.1}",
        params.days,
        params.days * params.rows_per_day,
        start.elapsed().as_secs_f64()
    );

    let verify_start = Instant::now();
    verify_sorted_files(&table, table_url)?;
    println!(
        "verified files_sorted=true verify_s={:.1}",
        verify_start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Re-read the timestamp column of every live file and check that each file is
/// internally sorted ascending and that file ranges do not overlap. Only
/// supported for local (file://) tables; skipped otherwise.
fn verify_sorted_files(table: &DeltaTable, table_url: &Url) -> DeltaResult<()> {
    let Ok(root) = table_url.to_file_path() else {
        println!("verification skipped: not a local file table");
        return Ok(());
    };

    let mut ranges: Vec<(i64, i64, String)> = Vec::new();
    for file in table.snapshot()?.log_data().iter() {
        let name = file.path().to_string();
        let path = root.join(&name);
        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path)?)?;
        let mask = ProjectionMask::leaves(reader.parquet_schema(), [0]);
        let mut first: Option<i64> = None;
        let mut last: Option<i64> = None;
        for batch in reader.with_projection(mask).build()? {
            let batch = batch?;
            let timestamps = batch
                .column(0)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| DeltaTableError::generic("unexpected type for timestamp column"))?
                .values();
            for &ts in timestamps.iter() {
                if let Some(last) = last {
                    if ts < last {
                        return Err(DeltaTableError::generic(format!(
                            "file {name} is not sorted: ts={ts} after previous_ts={last}"
                        )));
                    }
                }
                first.get_or_insert(ts);
                last = Some(ts);
            }
        }
        let (Some(first), Some(last)) = (first, last) else {
            return Err(DeltaTableError::generic(format!("file {name} is empty")));
        };
        ranges.push((first, last, name));
    }

    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        let (_, max_a, name_a) = &pair[0];
        let (min_b, _, name_b) = &pair[1];
        if min_b < max_a {
            return Err(DeltaTableError::generic(format!(
                "files {name_a} and {name_b} have overlapping timestamp ranges"
            )));
        }
    }
    Ok(())
}

/// Which table-provider configuration to benchmark.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum SortBenchMode {
    /// No sort order information: DataFusion must fully sort the scan output.
    Baseline,
    /// Sort order declared up front via `with_file_sort_order`.
    Declared,
}

impl SortBenchMode {
    pub fn name(&self) -> &'static str {
        match self {
            SortBenchMode::Baseline => "baseline",
            SortBenchMode::Declared => "declared",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SortBenchParams {
    pub mode: SortBenchMode,
    /// Number of extra float32 columns included in the SELECT; `None` selects
    /// all of them.
    pub select_columns: Option<usize>,
    /// Optional LIMIT on the query (exercises TopK / early termination).
    pub limit: Option<usize>,
    /// Memory budget in bytes; when set, a `FairSpillPool` of this size is
    /// installed so sorts spill to disk instead of exhausting memory.
    pub memory_limit_bytes: Option<usize>,
    /// Override `datafusion.execution.target_partitions` (defaults to the
    /// number of CPU cores).
    pub target_partitions: Option<usize>,
}

#[derive(Debug)]
pub struct SortBenchReport {
    /// Rendered physical plan.
    pub plan: String,
    pub has_sort_exec: bool,
    pub has_sort_preserving_merge: bool,
    /// Time to build the table provider.
    pub provider: Duration,
    /// Time to plan the query.
    pub planning: Duration,
    /// Time from execution start until the first batch arrived.
    pub first_batch: Option<Duration>,
    /// Time from execution start until the stream was fully drained.
    pub total: Duration,
    pub rows: usize,
    pub batches: usize,
    /// Whether the streamed timestamps were globally non-decreasing.
    pub sorted: bool,
    /// Description of the first out-of-order row, when `sorted` is false.
    pub first_violation: Option<String>,
    /// Peak RSS of the process so far (linux only); cumulative across
    /// iterations, so only the first run of the heaviest mode is meaningful.
    pub peak_rss_mb: Option<u64>,
}

/// Names of the extra (non-timestamp) columns of the table, in schema order.
fn extra_column_names(table: &DeltaTable) -> DeltaResult<Vec<String>> {
    Ok(table
        .snapshot()?
        .schema()
        .fields()
        .map(|f| f.name().to_string())
        .filter(|name| name != TIMESTAMP_COLUMN)
        .collect())
}

fn build_query(extra_columns: &[String], params: &SortBenchParams) -> String {
    let selected = match params.select_columns {
        Some(n) => &extra_columns[..n.min(extra_columns.len())],
        None => extra_columns,
    };
    let mut select_list = format!("\"{TIMESTAMP_COLUMN}\"");
    for name in selected {
        select_list.push_str(&format!(", \"{name}\""));
    }
    let mut sql = format!("SELECT {select_list} FROM t ORDER BY \"{TIMESTAMP_COLUMN}\"");
    if let Some(limit) = params.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }
    sql
}

fn peak_rss_mb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line
        .trim_start_matches("VmHWM:")
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kb / 1024)
}

/// Run one sorted streaming query against the table and measure it.
///
/// A fresh session (and hence a cold file-metadata cache) is used every call.
pub async fn run_sort_bench_once(
    table_url: &Url,
    params: &SortBenchParams,
) -> DeltaResult<SortBenchReport> {
    let table = open_table(table_url.clone()).await?;
    let extra_columns = extra_column_names(&table)?;
    let sql = build_query(&extra_columns, params);

    let ctx = match params.memory_limit_bytes {
        Some(bytes) => DeltaSessionContext::with_runtime_env(
            DeltaRuntimeEnvBuilder::new()
                .with_max_spill_size(bytes)
                .build(),
        ),
        None => create_session(),
    }
    .into_inner();
    if let Some(partitions) = params.target_partitions {
        ctx.sql(&format!(
            "SET datafusion.execution.target_partitions = {partitions}"
        ))
        .await?
        .collect()
        .await?;
    }

    let provider_start = Instant::now();
    let builder = table.table_provider();
    let builder = match params.mode {
        SortBenchMode::Baseline => builder,
        SortBenchMode::Declared => {
            builder.with_file_sort_order([FileSortColumn::asc(TIMESTAMP_COLUMN)])
        }
    };
    let provider = builder.await?;
    let provider_elapsed = provider_start.elapsed();
    ctx.register_table("t", provider)?;

    let planning_start = Instant::now();
    let df = ctx.sql(&sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_elapsed = planning_start.elapsed();
    let rendered = displayable(plan.as_ref()).indent(true).to_string();

    let exec_start = Instant::now();
    let mut stream = execute_stream(plan, ctx.task_ctx())?;
    let mut first_batch = None;
    let mut rows = 0usize;
    let mut batches = 0usize;
    let mut sorted = true;
    let mut first_violation = None;
    let mut last_ts: Option<i64> = None;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        if first_batch.is_none() {
            first_batch = Some(exec_start.elapsed());
        }
        batches += 1;
        let timestamps = batch
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| DeltaTableError::generic("unexpected type for timestamp column"))?
            .values();
        for (row, &ts) in timestamps.iter().enumerate() {
            if let Some(last) = last_ts {
                if ts < last {
                    sorted = false;
                    if first_violation.is_none() {
                        first_violation = Some(format!(
                            "batch={batches} row={} ts={ts} previous_ts={last}",
                            rows + row
                        ));
                    }
                }
            }
            last_ts = Some(ts);
        }
        rows += batch.num_rows();
    }
    let total = exec_start.elapsed();

    Ok(SortBenchReport {
        has_sort_exec: rendered.contains("SortExec"),
        has_sort_preserving_merge: rendered.contains("SortPreservingMergeExec"),
        plan: rendered,
        provider: provider_elapsed,
        planning: planning_elapsed,
        first_batch,
        total,
        rows,
        batches,
        sorted,
        first_violation,
        peak_rss_mb: peak_rss_mb(),
    })
}
