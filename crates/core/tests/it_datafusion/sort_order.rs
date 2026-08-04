//! End-to-end tests of the Delta table provider with a per-file sort order
//! declared via `with_file_sort_order`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, TimestampMicrosecondType};
use arrow_array::{Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::physical_plan::displayable;
use deltalake_core::DeltaTable;
use deltalake_core::delta_datafusion::{FileSortColumn, create_session};
use deltalake_core::kernel::{DataType as DeltaDataType, PrimitiveType, StructField, StructType};
use deltalake_core::protocol::SaveMode;
use deltalake_test::TestResult;

/// Check the query returned all 400 rows in timestamp order.
fn assert_sorted_result(batches: &[RecordBatch]) {
    let timestamps: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<TimestampMicrosecondType>()
                .values()
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(timestamps.len(), 400);
    assert!(
        timestamps.windows(2).all(|pair| pair[0] <= pair[1]),
        "results are not sorted by timestamp"
    );
}

fn delta_write_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("value", DataType::Int64, false),
        Field::new("part", DataType::Utf8, false),
    ]))
}

fn delta_write_batch(part: &str, start: i64, len: i64) -> TestResult<RecordBatch> {
    let timestamps: Vec<i64> = (start..start + len).map(|s| s * 1_000_000).collect();
    let values: Vec<i64> = (start..start + len).collect();
    let parts: Vec<&str> = (0..len).map(|_| part).collect();
    Ok(RecordBatch::try_new(
        delta_write_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(timestamps)),
            Arc::new(Int64Array::from(values)),
            Arc::new(StringArray::from(parts)),
        ],
    )?)
}

/// Create a Delta table partitioned by "part": every file sorted by
/// timestamp, non-overlapping timestamp ranges within a partition,
/// overlapping ranges across partitions. The file count is uneven across
/// partitions so that grouping which ignores statistics always places
/// overlapping files in the same group.
async fn sorted_delta_table() -> TestResult<DeltaTable> {
    let mut table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "timestamp".to_string(),
                DeltaDataType::Primitive(PrimitiveType::TimestampNtz),
                false,
            ),
            StructField::new(
                "value".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
            StructField::new(
                "part".to_string(),
                DeltaDataType::Primitive(PrimitiveType::String),
                false,
            ),
        ])
        .with_partition_columns(vec!["part"])
        .await?;

    let writes = [
        ("A", 0, 100),
        ("A", 100, 100),
        ("A", 200, 100),
        ("B", 50, 100),
    ];
    for (part, start, len) in writes {
        table = table
            .write(vec![delta_write_batch(part, start, len)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }
    assert_eq!(table.snapshot()?.log_data().num_files(), 4);
    Ok(table)
}

/// Querying a Delta table whose files are sorted by timestamp, with that
/// order declared on the table provider, should use the sort pushdown
/// optimization and avoid a full sort.
#[tokio::test]
async fn delta_table_sorted_scan_avoids_sort() -> TestResult<()> {
    let table = sorted_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", value, part FROM test_table ORDER BY \"timestamp\"")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert_sorted_result(&batches);
    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("SortPreservingMergeExec"),
        "expected SortPreservingMergeExec in plan:\n{rendered}"
    );
    Ok(())
}

/// A LIMIT on an ordered query needs no TopK sort when the file sort order is
/// declared: the fetch terminates the pre-ordered scan early instead.
#[tokio::test]
async fn delta_table_sort_order_limit_avoids_topk() -> TestResult<()> {
    let table = sorted_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", value, part FROM test_table ORDER BY \"timestamp\" LIMIT 10")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec (TopK) in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("fetch=10"),
        "expected the limit to be pushed into the plan as fetch=10:\n{rendered}"
    );
    let timestamps = collect_timestamps(&batches);
    let expected: Vec<i64> = (0..10).map(|s| s * 1_000_000).collect();
    assert_eq!(
        timestamps, expected,
        "expected the 10 smallest timestamps in order"
    );
    Ok(())
}

/// When the query does not scan the declared sort column, the declared order
/// cannot be exposed and the query falls back to a full sort with correct
/// results.
#[tokio::test]
async fn delta_table_sort_order_degrades_without_sort_column() -> TestResult<()> {
    let table = sorted_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT value FROM test_table ORDER BY value")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        rendered.contains("SortExec"),
        "expected SortExec in plan:\n{rendered}"
    );
    let values: Vec<i64> = batches
        .iter()
        .flat_map(|batch| batch.column(0).as_primitive::<Int64Type>().values().iter())
        .copied()
        .collect();
    assert_eq!(values.len(), 400);
    assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// Declaring a sort order on a partition column or an unknown column is
/// rejected when building the provider.
#[tokio::test]
async fn delta_table_sort_order_validation() -> TestResult<()> {
    let table = sorted_delta_table().await?;

    let err = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("part")])
        .await
        .expect_err("partition column sort order should be rejected");
    assert!(err.to_string().contains("partition column"), "{err}");

    let err = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("missing")])
        .await
        .expect_err("unknown column sort order should be rejected");
    assert!(err.to_string().contains("does not exist"), "{err}");

    // A dotted name that does not resolve to a nested field is reported as a
    // plain unknown column, not as a nested field reference.
    let err = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("nested.field")])
        .await
        .expect_err("unknown dotted column sort order should be rejected");
    assert!(err.to_string().contains("does not exist"), "{err}");

    let struct_table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "timestamp".to_string(),
                DeltaDataType::Primitive(PrimitiveType::TimestampNtz),
                false,
            ),
            StructField::new(
                "nested".to_string(),
                DeltaDataType::Struct(Box::new(StructType::try_new(vec![StructField::new(
                    "field".to_string(),
                    DeltaDataType::Primitive(PrimitiveType::Long),
                    true,
                )])?)),
                true,
            ),
        ])
        .await?;
    let err = struct_table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("nested.field")])
        .await
        .expect_err("nested field sort order should be rejected");
    assert!(err.to_string().contains("top-level"), "{err}");

    let err = table
        .table_provider()
        .with_file_sort_order([
            FileSortColumn::asc("timestamp"),
            FileSortColumn::asc("timestamp"),
        ])
        .await
        .expect_err("duplicate sort order column should be rejected");
    assert!(err.to_string().contains("more than once"), "{err}");
    Ok(())
}

/// Ordering by the partition key and then the file sort order requires a
/// `SortExec`: partition columns cannot participate in the file sort order
/// (they are injected above the parquet scan), so the scan only exposes the
/// `timestamp` ordering and file groups may interleave partition values.
#[tokio::test]
async fn delta_table_sort_order_degrades_for_partition_key_prefix() -> TestResult<()> {
    let table = sorted_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT part, \"timestamp\", value FROM test_table ORDER BY part, \"timestamp\"")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        rendered.contains("SortExec"),
        "expected SortExec in plan:\n{rendered}"
    );

    let mut keys: Vec<(String, i64)> = Vec::new();
    for batch in &batches {
        // The partition column is produced by the kernel transform and may be
        // dictionary-encoded or a view array; cast to plain Utf8 to read it.
        let parts = arrow_cast::cast(batch.column(0), &DataType::Utf8)?;
        let parts = parts.as_string::<i32>();
        let timestamps = batch
            .column(1)
            .as_primitive::<TimestampMicrosecondType>()
            .values();
        keys.extend(
            parts
                .iter()
                .map(|part| part.unwrap().to_string())
                .zip(timestamps.iter().copied()),
        );
    }
    assert_eq!(keys.len(), 400);
    assert!(
        keys.windows(2).all(|pair| pair[0] <= pair[1]),
        "results are not sorted by (part, timestamp)"
    );
    Ok(())
}

// --- Multi-column sort order ---

/// Each timestamp appears once per object id, so ties on the leading sort
/// column make the secondary column meaningful.
const OBJECTS_PER_TIMESTAMP: i64 = 4;

fn multi_sort_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("object_id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

/// Batch covering `len` timestamps starting at `start` seconds, with
/// [`OBJECTS_PER_TIMESTAMP`] rows per timestamp, sorted by
/// (timestamp, object_id).
fn multi_sort_batch(start: i64, len: i64) -> TestResult<RecordBatch> {
    let rows = (len * OBJECTS_PER_TIMESTAMP) as usize;
    let mut timestamps = Vec::with_capacity(rows);
    let mut object_ids = Vec::with_capacity(rows);
    let mut values = Vec::with_capacity(rows);
    for ts in start..start + len {
        for object_id in 0..OBJECTS_PER_TIMESTAMP {
            timestamps.push(ts * 1_000_000);
            object_ids.push(object_id);
            values.push(ts * OBJECTS_PER_TIMESTAMP + object_id);
        }
    }
    Ok(RecordBatch::try_new(
        multi_sort_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(timestamps)),
            Arc::new(Int64Array::from(object_ids)),
            Arc::new(Int64Array::from(values)),
        ],
    )?)
}

/// Create a Delta table whose files are all sorted by (timestamp, object_id),
/// with non-overlapping timestamp ranges across files.
async fn multi_sorted_delta_table() -> TestResult<DeltaTable> {
    let mut table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "timestamp".to_string(),
                DeltaDataType::Primitive(PrimitiveType::TimestampNtz),
                false,
            ),
            StructField::new(
                "object_id".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
            StructField::new(
                "value".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
        ])
        .await?;

    for start in [0, 50, 100, 150] {
        table = table
            .write(vec![multi_sort_batch(start, 50)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }
    assert_eq!(table.snapshot()?.log_data().num_files(), 4);
    Ok(table)
}

/// A multi-column sort order declared on the provider satisfies an ORDER BY
/// over the same columns without a `SortExec`, and the streamed rows are in
/// (timestamp, object_id) order — the secondary column ordering is exercised
/// by the repeated timestamps within each file. The files are mutually
/// non-overlapping, so the merge is replaced by a `ProgressiveEvalExec`
/// concatenation.
#[tokio::test]
async fn delta_table_multi_column_sort_order_avoids_sort() -> TestResult<()> {
    let table = multi_sorted_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([
            FileSortColumn::asc("timestamp"),
            FileSortColumn::asc("object_id"),
        ])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql(
            "SELECT \"timestamp\", object_id, value FROM test_table \
             ORDER BY \"timestamp\", object_id",
        )
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );

    let mut keys: Vec<(i64, i64)> = Vec::new();
    for batch in &batches {
        let timestamps = batch
            .column(0)
            .as_primitive::<TimestampMicrosecondType>()
            .values();
        let object_ids = batch.column(1).as_primitive::<Int64Type>().values();
        keys.extend(timestamps.iter().copied().zip(object_ids.iter().copied()));
    }
    assert_eq!(keys.len(), (4 * 50 * OBJECTS_PER_TIMESTAMP) as usize);
    assert!(
        keys.windows(2).all(|pair| pair[0] <= pair[1]),
        "results are not sorted by (timestamp, object_id)"
    );
    Ok(())
}

/// Batch sorted by (t, object_id) covering `t_start..=t_end` (inclusive, so
/// adjacent files can share a boundary value) with one row per (t, id) for
/// the given ids. The first sort column is an integer: delta-rs pads
/// timestamp max statistics up to the next millisecond, so an exactly equal
/// boundary is only provable from statistics for non-timestamp columns.
fn shared_boundary_batch(t_start: i64, t_end: i64, ids: &[i64]) -> TestResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("t", DataType::Int64, false),
        Field::new("object_id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let mut ts = Vec::new();
    let mut object_ids = Vec::new();
    let mut values = Vec::new();
    for t in t_start..=t_end {
        for &id in ids {
            ts.push(t);
            object_ids.push(id);
            values.push(t * 100 + id);
        }
    }
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ts)),
            Arc::new(Int64Array::from(object_ids)),
            Arc::new(Int64Array::from(values)),
        ],
    )?)
}

/// Create a table of two files sorted by (t, object_id) that share the
/// boundary value t = 100: the first file holds object ids {0, 1} and the
/// second ids {2, 3}, so at the boundary the files are ordered only by
/// virtue of the second sort column.
async fn shared_boundary_delta_table() -> TestResult<DeltaTable> {
    let mut table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "t".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
            StructField::new(
                "object_id".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
            StructField::new(
                "value".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
        ])
        .await?;

    for (t_start, t_end, ids) in [(0i64, 100i64, [0i64, 1]), (100, 200, [2, 3])] {
        table = table
            .write(vec![shared_boundary_batch(t_start, t_end, &ids)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }
    assert_eq!(table.snapshot()?.log_data().num_files(), 2);
    Ok(table)
}

/// Two files share the boundary value of the first sort column, so it alone
/// cannot prove the scan partitions ordered; the disjoint object-id ranges
/// can. The equal-boundary check falls through to the second sort column and
/// the merge is still replaced by the `ProgressiveEvalExec` concatenation.
#[tokio::test]
async fn delta_table_shared_boundary_value_ordered_by_second_column() -> TestResult<()> {
    let table = shared_boundary_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("t"), FileSortColumn::asc("object_id")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT t, object_id, value FROM test_table ORDER BY t, object_id")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );

    let mut keys: Vec<(i64, i64)> = Vec::new();
    for batch in &batches {
        let ts = batch.column(0).as_primitive::<Int64Type>().values();
        let object_ids = batch.column(1).as_primitive::<Int64Type>().values();
        keys.extend(ts.iter().copied().zip(object_ids.iter().copied()));
    }
    // Two files x 101 values of t x 2 object ids.
    assert_eq!(keys.len(), 2 * 101 * 2);
    assert!(
        keys.windows(2).all(|pair| pair[0] <= pair[1]),
        "results are not sorted by (t, object_id)"
    );
    Ok(())
}

/// An ORDER BY over a prefix of the declared multi-column sort order is
/// satisfied by the declared ordering without a `SortExec`, and the
/// non-overlapping files allow the `ProgressiveEvalExec` concatenation.
#[tokio::test]
async fn delta_table_sort_order_prefix_query_avoids_sort() -> TestResult<()> {
    let table = multi_sorted_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([
            FileSortColumn::asc("timestamp"),
            FileSortColumn::asc("object_id"),
        ])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", object_id, value FROM test_table ORDER BY \"timestamp\"")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );
    let timestamps = collect_timestamps(&batches);
    assert_eq!(timestamps.len(), (4 * 50 * OBJECTS_PER_TIMESTAMP) as usize);
    assert!(
        timestamps.windows(2).all(|pair| pair[0] <= pair[1]),
        "results are not sorted by timestamp"
    );
    Ok(())
}

// --- Descending sort order ---

fn desc_write_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("value", DataType::Int64, false),
    ]))
}

/// Batch covering `[start, start + len)` seconds with rows in *descending*
/// timestamp order.
fn desc_write_batch(start: i64, len: i64) -> TestResult<RecordBatch> {
    let timestamps: Vec<i64> = (start..start + len).rev().map(|s| s * 1_000_000).collect();
    let values: Vec<i64> = (start..start + len).rev().collect();
    Ok(RecordBatch::try_new(
        desc_write_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(timestamps)),
            Arc::new(Int64Array::from(values)),
        ],
    )?)
}

/// Create a Delta table whose files are all sorted by timestamp *descending*,
/// with non-overlapping timestamp ranges across files.
async fn desc_sorted_delta_table() -> TestResult<DeltaTable> {
    let mut table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "timestamp".to_string(),
                DeltaDataType::Primitive(PrimitiveType::TimestampNtz),
                false,
            ),
            StructField::new(
                "value".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
        ])
        .await?;

    for start in [300, 200, 100, 0] {
        table = table
            .write(vec![desc_write_batch(start, 100)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }
    assert_eq!(table.snapshot()?.log_data().num_files(), 4);
    Ok(table)
}

fn collect_timestamps(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<TimestampMicrosecondType>()
                .values()
                .iter()
                .copied()
        })
        .collect()
}

/// A descending sort order declared on the provider satisfies a matching
/// `ORDER BY ... DESC` without a `SortExec`. The files are mutually
/// non-overlapping, so the merge is further replaced by a
/// `ProgressiveEvalExec` concatenation.
#[tokio::test]
async fn delta_table_descending_sort_order_avoids_sort() -> TestResult<()> {
    let table = desc_sorted_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::desc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", value FROM test_table ORDER BY \"timestamp\" DESC")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );
    let timestamps = collect_timestamps(&batches);
    assert_eq!(timestamps.len(), 400);
    assert!(
        timestamps.windows(2).all(|pair| pair[0] >= pair[1]),
        "results are not sorted by timestamp descending"
    );
    Ok(())
}

/// An ascending query over a table declared as descending cannot use the
/// declared order (files cannot be read backwards) and falls back to a full
/// sort with correct results.
#[tokio::test]
async fn delta_table_descending_sort_order_degrades_for_ascending_query() -> TestResult<()> {
    let table = desc_sorted_delta_table().await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::desc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", value FROM test_table ORDER BY \"timestamp\"")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        rendered.contains("SortExec"),
        "expected SortExec in plan:\n{rendered}"
    );
    let timestamps = collect_timestamps(&batches);
    assert_eq!(timestamps.len(), 400);
    assert!(
        timestamps.windows(2).all(|pair| pair[0] <= pair[1]),
        "results are not sorted by timestamp ascending"
    );
    Ok(())
}

// --- Overlapping file ranges ---

/// Batch with the given timestamps (in seconds, must be sorted ascending) and
/// matching values, using the unpartitioned two-column schema.
fn timestamps_batch(seconds: Vec<i64>) -> TestResult<RecordBatch> {
    let timestamps: Vec<i64> = seconds.iter().map(|s| s * 1_000_000).collect();
    Ok(RecordBatch::try_new(
        desc_write_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(timestamps)),
            Arc::new(Int64Array::from(seconds)),
        ],
    )?)
}

/// Create an unpartitioned Delta table with one file per entry of `files`,
/// each file sorted by timestamp ascending. The timestamp ranges of the files
/// are free to overlap.
async fn overlapping_delta_table(files: Vec<Vec<i64>>) -> TestResult<DeltaTable> {
    let mut table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "timestamp".to_string(),
                DeltaDataType::Primitive(PrimitiveType::TimestampNtz),
                false,
            ),
            StructField::new(
                "value".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
        ])
        .await?;

    let expected_files = files.len();
    for seconds in files {
        table = table
            .write(vec![timestamps_batch(seconds)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }
    assert_eq!(table.snapshot()?.log_data().num_files(), expected_files);
    Ok(table)
}

/// Four files where each file's timestamp range overlaps the start of the
/// next one: [0, 110), [100, 210), [200, 310), [300, 410).
fn chain_overlapping_files() -> Vec<Vec<i64>> {
    [0i64, 100, 200, 300]
        .into_iter()
        .map(|start| (start..start + 110).collect())
        .collect()
}

/// Run `ORDER BY timestamp` with the given target partition count and return
/// the rendered plan and the collected timestamps.
async fn query_sorted_with_target_partitions(
    table: &DeltaTable,
    target_partitions: usize,
) -> TestResult<(String, Vec<i64>)> {
    query_sorted_with_session_options(
        table,
        &[(
            "datafusion.execution.target_partitions",
            &target_partitions.to_string(),
        )],
    )
    .await
}

/// Run `ORDER BY timestamp` with the given session options applied and return
/// the rendered plan and the collected timestamps.
async fn query_sorted_with_session_options(
    table: &DeltaTable,
    options: &[(&str, &str)],
) -> TestResult<(String, Vec<i64>)> {
    let ctx = create_session().into_inner();
    for (key, value) in options {
        ctx.sql(&format!("SET {key} = {value}")).await?;
    }
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", value FROM test_table ORDER BY \"timestamp\"")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
    Ok((rendered, collect_timestamps(&batches)))
}

/// With chain-overlapping files and more than one target partition, the
/// statistics-based grouping interleaves the files into two groups whose
/// members are non-overlapping, so the declared order is preserved and no
/// `SortExec` is needed.
#[tokio::test]
async fn delta_table_chain_overlapping_files_avoid_sort() -> TestResult<()> {
    let table = overlapping_delta_table(chain_overlapping_files()).await?;
    let (rendered, timestamps) = query_sorted_with_target_partitions(&table, 2).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("SortPreservingMergeExec"),
        "expected SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("2 groups"),
        "expected the files to be split into 2 groups:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 440);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// With a single target partition, chain-overlapping files cannot form one
/// ordered group; the grouping overflows the target and still produces two
/// non-overlapping groups, so the sort is still avoided.
#[tokio::test]
async fn delta_table_chain_overlapping_files_single_target_partition() -> TestResult<()> {
    let table = overlapping_delta_table(chain_overlapping_files()).await?;
    let (rendered, timestamps) = query_sorted_with_target_partitions(&table, 1).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("SortPreservingMergeExec"),
        "expected SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("2 groups"),
        "expected the files to be split into 2 groups:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 440);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// When every file overlaps every other file and there are more files than
/// target partitions, no file can share a group with any other: the grouping
/// degrades to one group per file. The declared order is still preserved (a
/// single sorted file per group) so the sort is avoided, at the cost of more
/// scan partitions than requested.
#[tokio::test]
async fn delta_table_fully_overlapping_files_one_group_per_file() -> TestResult<()> {
    // File i holds timestamps i, i + 4, i + 8, ...: every file spans almost
    // the entire [0, 400) range, so all files overlap each other.
    let files: Vec<Vec<i64>> = (0i64..4)
        .map(|file| (0..100).map(|row| row * 4 + file).collect())
        .collect();
    let table = overlapping_delta_table(files).await?;
    let (rendered, timestamps) = query_sorted_with_target_partitions(&table, 2).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("SortPreservingMergeExec"),
        "expected SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("4 groups"),
        "expected one group per file:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 400);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// When overlapping files would need more groups than the cap of
/// `max(64, 2 * target_partitions)` allows, the statistics-based grouping is
/// abandoned: the scan uses the default grouping, DataFusion drops the
/// unprovable ordering, and the query falls back to a full sort with correct
/// results.
#[tokio::test]
async fn delta_table_overlapping_files_beyond_group_cap_fall_back() -> TestResult<()> {
    // 65 mutually overlapping two-row files: file i covers [i, 1000 + i].
    // With target_partitions = 1 the cap is max(64, 2) = 64, and mutual
    // overlap would require one group per file (65).
    let files: Vec<Vec<i64>> = (0i64..65).map(|i| vec![i, 1_000 + i]).collect();
    let table = overlapping_delta_table(files).await?;
    let (rendered, timestamps) = query_sorted_with_target_partitions(&table, 1).await?;

    assert!(
        rendered.contains("SortExec"),
        "expected SortExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("65 groups"),
        "expected the grouping to fall back below one group per file:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 130);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

// --- More target partitions than files ---
//
// With fewer files than target partitions, `EnforceDistribution` has two
// independent mechanisms for raising the scan's partition count to the
// target, and each one would defeat the `ProgressiveEvalExec` optimization:
//
// - `enable_round_robin_repartition` inserts a round-robin `RepartitionExec`
//   above a scan that benefits from input partitioning.
//   `RepartitionExec::partition_statistics` marks all column statistics
//   inexact, so `ProgressiveEvalRule` bails.
//   `DeltaScanExec::benefits_from_input_partitioning` opts out of this
//   when the scan declares an output ordering.
// - `repartition_file_scans` splits single-file groups into byte ranges at
//   the source. Each range carries the whole file's statistics, so
//   partitions reading ranges of the same file have overlapping min/max and
//   `ProgressiveEvalRule` bails. Keeping the `ProgressiveEvalExec`
//   optimization requires disabling this option.

/// Session options simulating large production workloads on a many-core machine:
/// more target partitions than the table has files, and a batch size small enough
/// that the row-count statistics make repartitioning look beneficial.
const MANY_CORE_OPTIONS: &[(&str, &str)] = &[
    ("datafusion.execution.target_partitions", "8"),
    ("datafusion.execution.batch_size", "10"),
];

/// Two non-overlapping sorted files of 100 rows each.
fn two_non_overlapping_files() -> Vec<Vec<i64>> {
    vec![(0..100).collect(), (100..200).collect()]
}

/// With default optimizer options, `repartition_file_scans` splits the
/// scan's single-file groups into byte ranges to reach the target partition
/// count, and the `SortPreservingMergeExec` remains.
#[tokio::test]
async fn delta_table_more_target_partitions_than_files_file_split_defeats_progressive_eval()
-> TestResult<()> {
    let table = overlapping_delta_table(two_non_overlapping_files()).await?;
    let (rendered, timestamps) =
        query_sorted_with_session_options(&table, MANY_CORE_OPTIONS).await?;

    assert!(
        rendered.contains("8 groups"),
        "expected the files to be split into 8 byte-range groups:\n{rendered}"
    );
    assert!(
        rendered.contains("SortPreservingMergeExec"),
        "expected SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("ProgressiveEvalExec"),
        "expected no ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 200);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// With `repartition_file_scans` disabled, the scan keeps fewer partitions
/// than the target: no round-robin `RepartitionExec` is inserted above it
/// (the scan does not benefit from input partitioning), the scan partitions
/// stay non-overlapping, and the merge is replaced by `ProgressiveEvalExec`.
#[tokio::test]
async fn delta_table_more_target_partitions_than_files_uses_progressive_eval() -> TestResult<()> {
    let table = overlapping_delta_table(two_non_overlapping_files()).await?;
    let options: Vec<(&str, &str)> = MANY_CORE_OPTIONS
        .iter()
        .copied()
        .chain([("datafusion.optimizer.repartition_file_scans", "false")])
        .collect();
    let (rendered, timestamps) = query_sorted_with_session_options(&table, &options).await?;

    assert!(
        !rendered.contains("RepartitionExec"),
        "expected no RepartitionExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("2 groups"),
        "expected the file groups to stay intact:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 200);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// Without a declared sort order there are no ordering claims to protect, so
/// the scan keeps the default behaviour and benefits from input partitioning:
/// a round-robin `RepartitionExec` raises the scan's parallelism to the
/// target partition count.
#[tokio::test]
async fn delta_table_unordered_scan_gets_round_robin_repartition() -> TestResult<()> {
    let table = overlapping_delta_table(two_non_overlapping_files()).await?;

    let ctx = create_session().into_inner();
    for (key, value) in MANY_CORE_OPTIONS
        .iter()
        .copied()
        .chain([("datafusion.optimizer.repartition_file_scans", "false")])
    {
        ctx.sql(&format!("SET {key} = {value}")).await?;
    }
    let provider = table.table_provider().await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", value FROM test_table ORDER BY \"timestamp\"")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        rendered.contains("RoundRobinBatch"),
        "expected round-robin RepartitionExec in plan:\n{rendered}"
    );

    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
    let timestamps = collect_timestamps(&batches);
    assert_eq!(timestamps.len(), 200);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

// --- Nullable sort column ---

/// A batch over the nullable-timestamp schema: the given seconds ascending,
/// followed by `nulls` null timestamps (nulls last).
fn nullable_batch(seconds: Vec<i64>, nulls: usize) -> TestResult<RecordBatch> {
    let mut timestamps: Vec<Option<i64>> = seconds.iter().map(|s| Some(s * 1_000_000)).collect();
    let mut values: Vec<i64> = seconds;
    for i in 0..nulls {
        timestamps.push(None);
        values.push(-(i as i64) - 1);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("value", DataType::Int64, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(timestamps)),
            Arc::new(Int64Array::from(values)),
        ],
    )?)
}

/// Create a table with a nullable timestamp column and two files, each
/// internally sorted ascending with one null last, with non-overlapping
/// non-null ranges. Reading the files back-to-back would put the first file's
/// null before the second file's values.
async fn nullable_sorted_delta_table() -> TestResult<DeltaTable> {
    let mut table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "timestamp".to_string(),
                DeltaDataType::Primitive(PrimitiveType::TimestampNtz),
                true,
            ),
            StructField::new(
                "value".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
        ])
        .await?;

    for start in [0i64, 100] {
        table = table
            .write(vec![nullable_batch((start..start + 100).collect(), 1)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }
    assert_eq!(table.snapshot()?.log_data().num_files(), 2);
    Ok(table)
}

/// Run `ORDER BY timestamp` (ascending nulls last) against the nullable table
/// with the given target partition count and return the rendered plan and the
/// collected (nullable) timestamps.
async fn query_nullable_sorted(
    table: &DeltaTable,
    target_partitions: usize,
) -> TestResult<(String, Vec<Option<i64>>)> {
    let ctx = create_session().into_inner();
    ctx.sql(&format!(
        "SET datafusion.execution.target_partitions = {target_partitions}"
    ))
    .await?;
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", value FROM test_table ORDER BY \"timestamp\"")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
    let timestamps = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<TimestampMicrosecondType>()
                .iter()
        })
        .collect();
    Ok((rendered, timestamps))
}

/// Check the nullable query returned all 202 rows ascending with the two
/// nulls at the end (the ORDER BY default of ascending nulls last).
fn assert_sorted_nulls_last(timestamps: &[Option<i64>]) {
    assert_eq!(timestamps.len(), 202);
    assert_eq!(timestamps.iter().filter(|ts| ts.is_none()).count(), 2);
    let keys: Vec<(bool, Option<i64>)> = timestamps.iter().map(|ts| (ts.is_none(), *ts)).collect();
    assert!(
        keys.windows(2).all(|pair| pair[0] <= pair[1]),
        "results are not sorted ascending nulls last: {timestamps:?}"
    );
}

/// When several files containing nulls in the sort column share a scan
/// partition, the declared ordering cannot be used: nulls sort at one end of
/// *each file*, so concatenating files ordered by min/max interleaves the
/// nulls into the middle of the stream, which min/max statistics cannot
/// detect. The query must fall back to a full sort.
#[tokio::test]
async fn delta_table_sort_order_with_nulls_degrades() -> TestResult<()> {
    let table = nullable_sorted_delta_table().await?;
    // A single target partition forces both files into one group.
    let (rendered, timestamps) = query_nullable_sorted(&table, 1).await?;

    assert!(
        rendered.contains("SortExec"),
        "expected SortExec in plan:\n{rendered}"
    );
    assert_sorted_nulls_last(&timestamps);
    Ok(())
}

/// With one file per scan partition, each partition upholds the declared
/// ordering even though the sort column contains nulls (nulls last within a
/// single sorted file), so the merge can still replace the full sort.
#[tokio::test]
async fn delta_table_sort_order_with_nulls_single_file_groups_avoids_sort() -> TestResult<()> {
    let table = nullable_sorted_delta_table().await?;
    let (rendered, timestamps) = query_nullable_sorted(&table, 2).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("SortPreservingMergeExec"),
        "expected SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert_sorted_nulls_last(&timestamps);
    Ok(())
}

/// Create a table like [`nullable_sorted_delta_table`] but with the nulls
/// confined to the second (last) file.
async fn nulls_in_last_file_delta_table() -> TestResult<DeltaTable> {
    let mut table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "timestamp".to_string(),
                DeltaDataType::Primitive(PrimitiveType::TimestampNtz),
                true,
            ),
            StructField::new(
                "value".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
        ])
        .await?;

    for (start, nulls) in [(0i64, 0usize), (100, 2)] {
        table = table
            .write(vec![nullable_batch((start..start + 100).collect(), nulls)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }
    assert_eq!(table.snapshot()?.log_data().num_files(), 2);
    Ok(table)
}

/// Nulls in the sort column are compatible with the `ProgressiveEvalExec`
/// concatenation when they are confined to the last partition of an
/// ascending nulls-last ordering: every partition boundary is still provably
/// ordered.
#[tokio::test]
async fn delta_table_nulls_in_last_partition_use_progressive_eval() -> TestResult<()> {
    let table = nulls_in_last_file_delta_table().await?;
    let (rendered, timestamps) = query_nullable_sorted(&table, 2).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert_sorted_nulls_last(&timestamps);
    Ok(())
}

/// Batch over (t: not-null Long, object_id: nullable Long), pre-sorted by
/// (t ASC, object_id ASC NULLS LAST).
fn t_id_batch(rows: &[(i64, Option<i64>)]) -> TestResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("t", DataType::Int64, false),
        Field::new("object_id", DataType::Int64, true),
    ]));
    let ts: Vec<i64> = rows.iter().map(|(t, _)| *t).collect();
    let ids: Vec<Option<i64>> = rows.iter().map(|(_, id)| *id).collect();
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ts)),
            Arc::new(Int64Array::from(ids)),
        ],
    )?)
}

/// Nulls in the *second* sort column of a *middle* file must prevent the
/// `ProgressiveEvalExec` concatenation.
///
/// The middle file ends with (20, NULL), which sorts after every non-null
/// object_id at t = 20 (nulls last), while the last file begins with
/// (20, 5): statistics min/max exclude nulls, so the middle file's non-null
/// object_id max (4) < the last file's min (5) makes the boundary look
/// ordered even though the concatenation emits (20, NULL) before (20, 5).
/// The merge must be kept for the rows to stream in order.
#[tokio::test]
async fn delta_table_nulls_in_second_sort_column_of_middle_file_keep_merge() -> TestResult<()> {
    let mut table = DeltaTable::new_in_memory()
        .create()
        .with_columns(vec![
            StructField::new(
                "t".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                false,
            ),
            StructField::new(
                "object_id".to_string(),
                DeltaDataType::Primitive(PrimitiveType::Long),
                true,
            ),
        ])
        .await?;

    let files: Vec<Vec<(i64, Option<i64>)>> = vec![
        // Strictly before the middle file on t, so the middle file's second
        // sort column is never inspected.
        (0..10).map(|t| (t, Some(t))).collect(),
        vec![
            (10, Some(0)),
            (12, Some(1)),
            (14, Some(2)),
            (16, Some(3)),
            (20, Some(4)),
            (20, None),
        ],
        vec![(20, Some(5)), (21, Some(6)), (25, Some(7)), (30, Some(8))],
    ];
    for rows in &files {
        table = table
            .write(vec![t_id_batch(rows)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }
    assert_eq!(table.snapshot()?.log_data().num_files(), 3);

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("t"), FileSortColumn::asc("object_id")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT t, object_id FROM test_table ORDER BY t, object_id")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    // Sort key mirroring ORDER BY t, object_id (ASC NULLS LAST).
    let mut keys: Vec<(i64, bool, i64)> = Vec::new();
    for batch in &batches {
        let ts = batch.column(0).as_primitive::<Int64Type>().values();
        let ids = batch.column(1).as_primitive::<Int64Type>();
        for (row, t) in ts.iter().enumerate() {
            let id = ids.is_valid(row).then(|| ids.value(row));
            keys.push((*t, id.is_none(), id.unwrap_or_default()));
        }
    }
    assert_eq!(keys.len(), 20);
    assert!(
        keys.windows(2).all(|pair| pair[0] <= pair[1]),
        "results are not sorted by (t, object_id nulls last): {keys:?}"
    );

    assert!(
        !rendered.contains("ProgressiveEvalExec"),
        "expected no ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("SortPreservingMergeExec"),
        "expected SortPreservingMergeExec in plan:\n{rendered}"
    );
    Ok(())
}

// --- Column-mapped table ---

/// Physical parquet column names for the hand-written column-mapped table.
const CM_TS_PHYSICAL: &str = "col-4b74c04d-ts";
const CM_VALUE_PHYSICAL: &str = "col-8aa03c9e-value";

/// Write one data file of the column-mapped table, covering timestamps
/// `[start, start + len)`, sorted ascending. Returns the file size in bytes.
fn write_column_mapped_file(path: &std::path::Path, start: i64, len: i64) -> TestResult<u64> {
    use deltalake_core::parquet::arrow::ArrowWriter;

    let schema = Arc::new(Schema::new(vec![
        Field::new(CM_TS_PHYSICAL, DataType::Int64, true),
        Field::new(CM_VALUE_PHYSICAL, DataType::Int64, true),
    ]));
    let timestamps: Vec<i64> = (start..start + len).collect();
    let values: Vec<i64> = timestamps.iter().map(|ts| ts * 10).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Int64Array::from(values)),
        ],
    )?;
    let mut writer = ArrowWriter::try_new(std::fs::File::create(path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(std::fs::metadata(path)?.len())
}

/// Build a Delta table with `delta.columnMapping.mode = name` whose files are
/// sorted by the logical `ts` column with non-overlapping ranges.
///
/// delta-rs does not support writing to column-mapped tables, so the parquet
/// files (using physical column names) and the delta log are written by hand,
/// mirroring the layout of the `table_with_column_mapping` fixture. File
/// statistics are keyed by physical column names, as Delta requires.
fn create_column_mapped_table(root: &std::path::Path) -> TestResult<()> {
    let schema_string = serde_json::json!({
        "type": "struct",
        "fields": [
            {
                "name": "ts",
                "type": "long",
                "nullable": true,
                "metadata": {
                    "delta.columnMapping.id": 1,
                    "delta.columnMapping.physicalName": CM_TS_PHYSICAL,
                },
            },
            {
                "name": "value",
                "type": "long",
                "nullable": true,
                "metadata": {
                    "delta.columnMapping.id": 2,
                    "delta.columnMapping.physicalName": CM_VALUE_PHYSICAL,
                },
            },
        ],
    })
    .to_string();

    let mut actions = vec![
        serde_json::json!({"protocol": {"minReaderVersion": 2, "minWriterVersion": 5}}),
        serde_json::json!({"metaData": {
            "id": "11111111-2222-3333-4444-555555555555",
            "format": {"provider": "parquet", "options": {}},
            "schemaString": schema_string,
            "partitionColumns": [],
            "configuration": {
                "delta.columnMapping.mode": "name",
                "delta.columnMapping.maxColumnId": "2",
            },
            "createdTime": 1700000000000i64,
        }}),
    ];

    for (index, start) in [0i64, 100, 200].into_iter().enumerate() {
        let len = 100i64;
        let name = format!("part-{index:05}.parquet");
        let size = write_column_mapped_file(&root.join(&name), start, len)?;
        let stats = format!(
            r#"{{"numRecords":{len},"minValues":{{"{CM_TS_PHYSICAL}":{min}}},"maxValues":{{"{CM_TS_PHYSICAL}":{max}}},"nullCount":{{"{CM_TS_PHYSICAL}":0}}}}"#,
            min = start,
            max = start + len - 1,
        );
        actions.push(serde_json::json!({"add": {
            "path": name,
            "partitionValues": {},
            "size": size,
            "modificationTime": 1700000000000i64,
            "dataChange": true,
            "stats": stats,
        }}));
    }

    let log_dir = root.join("_delta_log");
    std::fs::create_dir_all(&log_dir)?;
    let commit: String = actions.iter().map(|action| format!("{action}\n")).collect();
    std::fs::write(log_dir.join("00000000000000000000.json"), commit)?;
    Ok(())
}

/// A sort order declared with logical column names works on a column-mapped
/// table: the ordering is translated to physical parquet column names for the
/// scan, and back to logical names on the scan output. The files are
/// non-overlapping, so the merge is replaced by a `ProgressiveEvalExec`
/// concatenation.
#[tokio::test]
async fn delta_table_column_mapped_sort_order_avoids_sort() -> TestResult<()> {
    let temp_dir = tempfile::tempdir()?;
    create_column_mapped_table(temp_dir.path())?;
    let table_url = url::Url::from_directory_path(temp_dir.path().canonicalize()?).unwrap();
    let table = deltalake_core::open_table(table_url).await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("ts")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT ts, value FROM test_table ORDER BY ts")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for batch in &batches {
        let timestamps = batch.column(0).as_primitive::<Int64Type>().values();
        let values = batch.column(1).as_primitive::<Int64Type>().values();
        rows.extend(timestamps.iter().copied().zip(values.iter().copied()));
    }
    assert_eq!(rows.len(), 300);
    assert!(
        rows.windows(2).all(|pair| pair[0].0 <= pair[1].0),
        "results are not sorted by ts"
    );
    // Both columns must have been mapped correctly from physical names.
    assert!(rows.iter().all(|(ts, value)| *value == ts * 10));
    Ok(())
}

// --- ProgressiveEval concatenation of non-overlapping partitions ---

/// Sum a named metric across a named node type in an executed plan tree.
fn sum_metric(
    plan: &dyn datafusion::physical_plan::ExecutionPlan,
    node: &str,
    metric: &str,
) -> Option<usize> {
    if plan.name() == node {
        return plan
            .metrics()?
            .sum_by_name(metric)
            .map(|value| value.as_usize());
    }
    plan.children()
        .into_iter()
        .find_map(|child| sum_metric(child.as_ref(), node, metric))
}

/// Four mutually non-overlapping files: [0, 100), [100, 200), [200, 300),
/// [300, 400).
fn disjoint_files() -> Vec<Vec<i64>> {
    [0i64, 100, 200, 300]
        .into_iter()
        .map(|start| (start..start + 100).collect())
        .collect()
}

/// With mutually non-overlapping files, the scan partitions are contiguous
/// range-ordered chunks of the ordered file list and the merge is replaced by
/// a `ProgressiveEvalExec` that concatenates them, reporting the partition
/// ranges in the plan.
#[tokio::test]
async fn delta_table_non_overlapping_files_use_progressive_eval() -> TestResult<()> {
    let table = overlapping_delta_table(disjoint_files()).await?;
    let (rendered, timestamps) = query_sorted_with_target_partitions(&table, 2).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec: input_ranges="),
        "expected ProgressiveEvalExec with input ranges in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("2 groups"),
        "expected two contiguous file groups:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 400);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// The contiguous chunking must produce exactly the target partition count
/// even when the file count does not divide evenly (here 6 files into 4
/// groups): a shortfall makes DataFusion repartition the scan for
/// parallelism, splitting files across partitions and losing both the
/// declared ordering (full-sort fallback) and the concatenation.
#[tokio::test]
async fn delta_table_progressive_eval_with_uneven_file_count() -> TestResult<()> {
    let files: Vec<Vec<i64>> = (0i64..6)
        .map(|file| (file * 100..file * 100 + 100).collect())
        .collect();
    let table = overlapping_delta_table(files).await?;
    let (rendered, timestamps) = query_sorted_with_target_partitions(&table, 4).await?;

    assert!(
        !rendered.contains("SortExec"),
        "expected no SortExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("4 groups"),
        "expected exactly the target partition count of file groups:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 600);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// Touching boundaries count as overlap for the concatenation: a shared value
/// on the group boundary is not provably ordered by min/max statistics alone,
/// so the merge is kept.
#[tokio::test]
async fn delta_table_touching_files_keep_merge() -> TestResult<()> {
    // File boundaries share the values 99, 199 and 299.
    let files: Vec<Vec<i64>> = [0i64, 99, 199, 299]
        .into_iter()
        .map(|start| (start..start + 100).collect())
        .collect();
    let table = overlapping_delta_table(files).await?;
    let (rendered, timestamps) = query_sorted_with_target_partitions(&table, 2).await?;

    assert!(
        !rendered.contains("ProgressiveEvalExec"),
        "expected no ProgressiveEvalExec in plan:\n{rendered}"
    );
    assert!(
        rendered.contains("SortPreservingMergeExec"),
        "expected SortPreservingMergeExec in plan:\n{rendered}"
    );
    assert_eq!(timestamps.len(), 400);
    assert!(timestamps.windows(2).all(|pair| pair[0] <= pair[1]));
    Ok(())
}

/// A LIMIT over the concatenated partitions is pushed into
/// `ProgressiveEvalExec` as a fetch and terminates the query after the first
/// partition: the later partitions' files are never read.
#[tokio::test]
async fn delta_table_progressive_eval_limit_reads_only_needed_partitions() -> TestResult<()> {
    let table = overlapping_delta_table(disjoint_files()).await?;

    let ctx = create_session().into_inner();
    ctx.sql("SET datafusion.execution.target_partitions = 4")
        .await?;
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql("SELECT \"timestamp\", value FROM test_table ORDER BY \"timestamp\" LIMIT 10")
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        rendered.contains("ProgressiveEvalExec: fetch=10"),
        "expected ProgressiveEvalExec with fetch=10 in plan:\n{rendered}"
    );
    assert!(
        !rendered.contains("SortPreservingMergeExec"),
        "expected no SortPreservingMergeExec in plan:\n{rendered}"
    );

    let batches = datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;
    let timestamps = collect_timestamps(&batches);
    let expected: Vec<i64> = (0..10).map(|s| s * 1_000_000).collect();
    assert_eq!(timestamps, expected);

    // Ten rows fit in the first of the four partitions; only it and the
    // prefetched next partition may have been executed.
    let num_read = sum_metric(plan.as_ref(), "ProgressiveEvalExec", "num_read_inputs")
        .expect("expected num_read_inputs metric on ProgressiveEvalExec");
    assert!(
        num_read <= 2,
        "expected at most 2 of 4 partitions to be executed, got {num_read}"
    );
    Ok(())
}

/// A filtered ordered query still uses the concatenation, and the filter is
/// applied within it.
///
/// The target partition count is kept at the file-group count: with a
/// pushed-down filter and spare target partitions, DataFusion otherwise
/// splits or round-robins the scan input for parallelism, which invalidates
/// the non-overlapping-partitions claim and (correctly) keeps a merge or sort
/// instead.
#[tokio::test]
async fn delta_table_progressive_eval_with_filter() -> TestResult<()> {
    let table = overlapping_delta_table(disjoint_files()).await?;

    let ctx = create_session().into_inner();
    // The predicate prunes one of the four files at planning time.
    ctx.sql("SET datafusion.execution.target_partitions = 3")
        .await?;
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("test_table", provider)?;

    let df = ctx
        .sql(
            "SELECT \"timestamp\", value FROM test_table \
             WHERE value >= 150 ORDER BY \"timestamp\"",
        )
        .await?;
    let plan = df.create_physical_plan().await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in plan:\n{rendered}"
    );
    let timestamps = collect_timestamps(&batches);
    let expected: Vec<i64> = (150..400).map(|s| s * 1_000_000).collect();
    assert_eq!(timestamps, expected);
    Ok(())
}

/// The concatenated results match the plain full-sort baseline (provider
/// without a declared sort order) exactly.
#[tokio::test]
async fn delta_table_progressive_eval_matches_sort_baseline() -> TestResult<()> {
    let table = overlapping_delta_table(disjoint_files()).await?;

    let ctx = create_session().into_inner();
    let provider = table
        .table_provider()
        .with_file_sort_order([FileSortColumn::asc("timestamp")])
        .await?;
    ctx.register_table("optimized", provider)?;
    let baseline_provider = table.table_provider().await?;
    ctx.register_table("baseline", baseline_provider)?;

    let query =
        |name: &str| format!("SELECT \"timestamp\", value FROM {name} ORDER BY \"timestamp\"");
    let optimized = ctx.sql(&query("optimized")).await?;
    let optimized_plan = optimized.create_physical_plan().await?;
    let rendered = displayable(optimized_plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        rendered.contains("ProgressiveEvalExec"),
        "expected ProgressiveEvalExec in optimized plan:\n{rendered}"
    );
    let optimized_batches =
        datafusion::physical_plan::collect(optimized_plan, ctx.task_ctx()).await?;

    let baseline_batches = ctx.sql(&query("baseline")).await?.collect().await?;

    let optimized_rows = collect_timestamps(&optimized_batches);
    let baseline_rows = collect_timestamps(&baseline_batches);
    assert_eq!(optimized_rows, baseline_rows);
    Ok(())
}
