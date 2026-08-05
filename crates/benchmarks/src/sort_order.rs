//! Test-data generation and read benchmarks for sorted streaming queries.
//!
//! Evaluates the sort-order support on the table provider: tables whose
//! parquet files are all sorted by a timestamp column can be scanned with a
//! `SortPreservingMergeExec` instead of a full `SortExec` when the ordering is
//! declared (`with_file_sort_order`). When the file ranges are additionally
//! mutually non-overlapping, the merge is replaced by a `ProgressiveEvalExec`
//! that plainly concatenates the range-ordered scan partitions.
//!
//! The generated table writes one commit per day, each containing a single
//! record batch sorted by `timestamp`. Day ranges never overlap, so file
//! min/max statistics allow DataFusion to form non-overlapping file groups.

use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::ValueEnum;
use deltalake_core::arrow::array::{ArrayRef, Float32Array, TimestampMicrosecondArray};
use deltalake_core::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use deltalake_core::arrow::record_batch::RecordBatch;
use deltalake_core::datafusion::catalog::Session;
use deltalake_core::datafusion::execution::SessionStateBuilder;
use deltalake_core::datafusion::physical_plan::{displayable, execute_stream};
use deltalake_core::datafusion::prelude::{SessionConfig, SessionContext};
use deltalake_core::delta_datafusion::planner::DeltaPlanner;
use deltalake_core::delta_datafusion::{
    create_session, DeltaRuntimeEnvBuilder, DeltaSessionConfig, FileSortColumn, ProgressiveEvalRule,
};
use deltalake_core::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use deltalake_core::parquet::arrow::ParquetRecordBatchStreamBuilder;
use deltalake_core::parquet::arrow::ProjectionMask;
use deltalake_core::parquet::basic::Compression;
use deltalake_core::parquet::file::properties::WriterProperties;
use deltalake_core::parquet::file::statistics::Statistics as ParquetStatistics;
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

fn writer_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
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

/// Write the test table: one commit per day, every file sorted by `timestamp`,
/// day ranges non-overlapping.
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
            .with_writer_properties(writer_properties())
            // Increase target file size to avoid writes being split
            .with_target_file_size(Some((10_000 * 1024 * 1024).try_into().unwrap()))
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
    /// No ORDER BY on the query at all: data is read in arbitrary order with
    /// no sorting needed. Lower bound for the cost of producing sorted output.
    Unordered,
    /// Sequential single-threaded read of every data file with the parquet
    /// crate directly (no delta-rs scan, no DataFusion), in ascending order of
    /// each file's minimum timestamp. Represents production workloads that
    /// read parquet files directly and in a known order; because the files
    /// are internally sorted and non-overlapping, the concatenated stream is
    /// globally ordered without any sorting work.
    SequentialRead,
    /// Like `sequential-read`, but through the parquet crate's async reader
    /// over tokio files: the IO pattern used by object-storage readers (and
    /// by DataFusion's parquet source), still without delta-rs or DataFusion.
    /// Isolates the cost of the async read path from the rest of the stack.
    SequentialReadAsync,
}

impl SortBenchMode {
    pub fn name(&self) -> &'static str {
        match self {
            SortBenchMode::Baseline => "baseline",
            SortBenchMode::Declared => "declared",
            SortBenchMode::Unordered => "unordered",
            SortBenchMode::SequentialRead => "sequential_read",
            SortBenchMode::SequentialReadAsync => "sequential_read_async",
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
    /// Override `delta.progressive_eval_num_prefetch_input_streams`: how many
    /// scan partitions `ProgressiveEvalExec` executes ahead of the one being
    /// streamed.
    pub prefetch_streams: Option<usize>,
    /// Whether to install the `ProgressiveEvalRule` physical optimizer rule.
    /// When disabled, non-overlapping ordered scans keep their
    /// `SortPreservingMergeExec` instead of being concatenated by a
    /// `ProgressiveEvalExec`.
    pub progressive_eval: bool,
    /// Verify that the streamed timestamps are globally non-decreasing. Off by
    /// default because the per-row check adds time to the measured run.
    pub check_order: bool,
}

#[derive(Debug)]
pub struct SortBenchReport {
    /// Rendered physical plan.
    pub plan: String,
    pub has_sort_exec: bool,
    pub has_sort_preserving_merge: bool,
    pub has_progressive_eval: bool,
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
    /// Whether the streamed timestamps were globally non-decreasing; `None`
    /// when order wasn't checked (`check_order` disabled, or the query had no
    /// ORDER BY in unordered mode).
    pub sorted: Option<bool>,
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
    let mut sql = format!("SELECT {select_list} FROM t");
    if params.mode != SortBenchMode::Unordered {
        sql.push_str(&format!(" ORDER BY \"{TIMESTAMP_COLUMN}\""));
    }
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

/// Per-batch accounting shared by the sequential read modes: applies the row
/// limit, verifies ordering, and tracks batch/row counts and first-batch time.
struct SequentialReadState {
    exec_start: Instant,
    first_batch: Option<Duration>,
    rows: usize,
    batches: usize,
    check_order: bool,
    sorted: bool,
    first_violation: Option<String>,
    last_ts: Option<i64>,
    remaining: Option<usize>,
}

impl SequentialReadState {
    fn new(exec_start: Instant, limit: Option<usize>, check_order: bool) -> Self {
        Self {
            exec_start,
            first_batch: None,
            rows: 0,
            batches: 0,
            check_order,
            sorted: true,
            first_violation: None,
            last_ts: None,
            remaining: limit,
        }
    }

    /// Returns true when the row limit has been reached and reading should stop.
    fn consume(&mut self, mut batch: RecordBatch) -> DeltaResult<bool> {
        if let Some(remaining) = self.remaining {
            if batch.num_rows() > remaining {
                batch = batch.slice(0, remaining);
            }
        }
        if self.first_batch.is_none() {
            self.first_batch = Some(self.exec_start.elapsed());
        }
        self.batches += 1;
        if self.check_order {
            let timestamps = batch
                .column(0)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| DeltaTableError::generic("unexpected type for timestamp column"))?
                .values();
            for (row, &ts) in timestamps.iter().enumerate() {
                if let Some(last) = self.last_ts {
                    if ts < last {
                        self.sorted = false;
                        if self.first_violation.is_none() {
                            self.first_violation = Some(format!(
                                "batch={} row={} ts={ts} previous_ts={last}",
                                self.batches,
                                self.rows + row
                            ));
                        }
                    }
                }
                self.last_ts = Some(ts);
            }
        }
        self.rows += batch.num_rows();
        if let Some(remaining) = self.remaining.as_mut() {
            *remaining -= batch.num_rows();
            if *remaining == 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Sequentially read every live data file with the parquet crate directly:
/// no delta-rs scan, no DataFusion. See [`SortBenchMode::SequentialRead`] and
/// [`SortBenchMode::SequentialReadAsync`].
///
/// The delta log is used only to list the live files; they are then ordered
/// by the minimum timestamp in their parquet footer statistics — the stand-in
/// for a production naming or layout convention that makes the read order
/// known up front. Reading is single-threaded, one file at a time, using the
/// sync arrow reader over `std::fs::File` or, when `async_reader` is set, the
/// async arrow reader over `tokio::fs::File`. `memory_limit_bytes` and
/// `target_partitions` are ignored. The batch size matches DataFusion's
/// default (8192) and, when `check_order` is set, the output ordering is
/// verified exactly like in the DataFusion modes.
async fn run_sequential_read(
    table_url: &Url,
    params: &SortBenchParams,
    async_reader: bool,
) -> DeltaResult<SortBenchReport> {
    let provider_start = Instant::now();
    let table = open_table(table_url.clone()).await?;
    let root = table_url.to_file_path().map_err(|_| {
        DeltaTableError::generic("sequential-read mode requires a local file:// table")
    })?;
    let extra_columns = extra_column_names(&table)?;
    let selected = params
        .select_columns
        .map(|n| n.min(extra_columns.len()))
        .unwrap_or(extra_columns.len());

    // The generated schema is flat, so arrow field indices equal parquet leaf
    // indices: leaf 0 is the timestamp, leaves 1..=selected the value columns.
    let leaves: Vec<usize> = (0..=selected).collect();

    let mut files: Vec<(i64, std::path::PathBuf)> = Vec::new();
    for file in table.snapshot()?.log_data().iter() {
        let path = root.join(file.path().to_string());
        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path)?)?;
        let min = match reader.metadata().row_group(0).column(0).statistics() {
            Some(ParquetStatistics::Int64(stats)) => stats.min_opt().copied(),
            _ => None,
        }
        .ok_or_else(|| {
            DeltaTableError::generic(format!(
                "missing int64 min statistic for the timestamp column of {}",
                path.display()
            ))
        })?;
        files.push((min, path));
    }
    files.sort_unstable();
    let provider_elapsed = provider_start.elapsed();

    let exec_start = Instant::now();
    let mut state = SequentialReadState::new(exec_start, params.limit, params.check_order);
    'files: for (_, path) in &files {
        if async_reader {
            let file = tokio::fs::File::open(path).await?;
            let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
            let mask = ProjectionMask::leaves(builder.parquet_schema(), leaves.iter().copied());
            let mut stream = builder
                .with_projection(mask)
                .with_batch_size(8192)
                .build()?;
            while let Some(batch) = stream.next().await {
                if state.consume(batch?)? {
                    break 'files;
                }
            }
        } else {
            let builder = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path)?)?;
            let mask = ProjectionMask::leaves(builder.parquet_schema(), leaves.iter().copied());
            let reader = builder
                .with_projection(mask)
                .with_batch_size(8192)
                .build()?;
            for batch in reader {
                if state.consume(batch?)? {
                    break 'files;
                }
            }
        }
    }
    let total = exec_start.elapsed();

    Ok(SortBenchReport {
        plan: format!(
            "SequentialParquetRead: files={}, columns={}, batch_size=8192, reader={} (parquet crate, bypasses delta-rs scan and DataFusion)",
            files.len(),
            selected + 1,
            if async_reader { "async" } else { "sync" },
        ),
        has_sort_exec: false,
        has_sort_preserving_merge: false,
        has_progressive_eval: false,
        provider: provider_elapsed,
        planning: Duration::ZERO,
        first_batch: state.first_batch,
        total,
        rows: state.rows,
        batches: state.batches,
        sorted: state.check_order.then_some(state.sorted),
        first_violation: state.first_violation,
        peak_rss_mb: peak_rss_mb(),
    })
}

/// Run one sorted streaming query against the table and measure it.
///
/// A fresh session (and hence a cold file-metadata cache) is used every call.
pub async fn run_sort_bench_once(
    table_url: &Url,
    params: &SortBenchParams,
) -> DeltaResult<SortBenchReport> {
    match params.mode {
        SortBenchMode::SequentialRead => {
            return run_sequential_read(table_url, params, false).await;
        }
        SortBenchMode::SequentialReadAsync => {
            return run_sequential_read(table_url, params, true).await;
        }
        _ => {}
    }
    let table = open_table(table_url.clone()).await?;
    let extra_columns = extra_column_names(&table)?;
    let sql = build_query(&extra_columns, params);

    let mut config: SessionConfig = DeltaSessionConfig::default().into();
    if params.mode == SortBenchMode::Declared {
        // Byte-range splitting of file scans defeats the progressive-eval
        // concatenation (the ranges carry whole-file statistics, so scan
        // partitions overlap). Only the declared mode needs it disabled;
        // the other modes keep the DataFusion default.
        config
            .options_mut()
            .set("datafusion.optimizer.repartition_file_scans", "false")?;
    }
    let planner = DeltaPlanner::new();
    let runtime_env = match params.memory_limit_bytes {
        Some(bytes) => DeltaRuntimeEnvBuilder::new()
            .with_max_spill_size(bytes)
            .build(),
        None => DeltaRuntimeEnvBuilder::new().build(),
    };
    let mut state_builder = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime_env)
        .with_query_planner(planner);
    if params.progressive_eval {
        state_builder =
            state_builder.with_physical_optimizer_rule(Arc::new(ProgressiveEvalRule::new()));
    }
    let state = state_builder.build();

    let ctx = SessionContext::new_with_state(state);

    if let Some(partitions) = params.target_partitions {
        ctx.sql(&format!(
            "SET datafusion.execution.target_partitions = {partitions}"
        ))
        .await?
        .collect()
        .await?;
    }
    if let Some(streams) = params.prefetch_streams {
        ctx.sql(&format!(
            "SET delta.progressive_eval_num_prefetch_input_streams = {streams}"
        ))
        .await?
        .collect()
        .await?;
    }

    let provider_start = Instant::now();
    let builder = table.table_provider();
    let builder = match params.mode {
        SortBenchMode::Baseline | SortBenchMode::Unordered => builder,
        SortBenchMode::Declared => {
            builder.with_file_sort_order([FileSortColumn::asc(TIMESTAMP_COLUMN)])
        }
        SortBenchMode::SequentialRead | SortBenchMode::SequentialReadAsync => {
            unreachable!("sequential read modes are handled by run_sequential_read")
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

    let check_order = params.check_order && params.mode != SortBenchMode::Unordered;
    let exec_start = Instant::now();
    let mut stream = execute_stream(plan, ctx.task_ctx())?;
    let mut first_batch = None;
    let mut rows = 0usize;
    let mut batches = 0usize;
    let mut sorted = check_order.then_some(true);
    let mut first_violation = None;
    let mut last_ts: Option<i64> = None;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        if first_batch.is_none() {
            first_batch = Some(exec_start.elapsed());
        }
        batches += 1;
        if check_order {
            let timestamps = batch
                .column(0)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| DeltaTableError::generic("unexpected type for timestamp column"))?
                .values();
            for (row, &ts) in timestamps.iter().enumerate() {
                if let Some(last) = last_ts {
                    if ts < last {
                        sorted = Some(false);
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
        }
        rows += batch.num_rows();
    }
    let total = exec_start.elapsed();

    Ok(SortBenchReport {
        has_sort_exec: rendered.contains("SortExec"),
        has_sort_preserving_merge: rendered.contains("SortPreservingMergeExec"),
        has_progressive_eval: rendered.contains("ProgressiveEvalExec"),
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
