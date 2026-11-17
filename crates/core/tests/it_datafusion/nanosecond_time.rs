// Test basic operations on a table that uses the nanosecond-time feature.
//
// Values carry sub-microsecond digits throughout to check that nothing truncates them.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Time64NanosecondType};
use arrow_array::{Int32Array, ListArray, RecordBatch, Time64NanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{col, lit};
use datafusion::prelude::SessionContext;
use delta_kernel::Predicate;
use delta_kernel::expressions::{Expression, Scalar};
use delta_kernel::table_features::TableFeature;
use deltalake_core::delta_datafusion::create_session;
use deltalake_core::{DeltaResult, DeltaTable};
use futures::TryStreamExt;

const TIME: DataType = DataType::Time64(TimeUnit::Nanosecond);
const MIDNIGHT: i64 = 0;
// 13:06:07.123456789
const AFTERNOON: i64 = 47_167_123_456_789;
// 23:59:59.999999999
const LAST_NANOSECOND: i64 = 86_399_999_999_999;

fn arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("t", TIME, true),
        Field::new_list("ts", Field::new_list_field(TIME, true), true),
    ]))
}

fn make_batch(ids: Vec<i32>, times: Vec<Option<i64>>) -> DeltaResult<RecordBatch> {
    let lists = times.iter().map(|time| Some(vec![*time]));
    Ok(RecordBatch::try_new(
        arrow_schema(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Time64NanosecondArray::from(times.clone())),
            Arc::new(ListArray::from_iter_primitive::<Time64NanosecondType, _, _>(lists)),
        ],
    )?)
}

async fn setup_table() -> DeltaResult<DeltaTable> {
    let batch = make_batch(
        vec![0, 1, 2, 3],
        vec![Some(MIDNIGHT), Some(AFTERNOON), Some(LAST_NANOSECOND), None],
    )?;
    DeltaTable::new_in_memory().write(vec![batch]).await
}

async fn query(table: DeltaTable, sql: &str) -> Vec<RecordBatch> {
    let ctx: SessionContext = create_session().into();
    table.update_datafusion_session(&ctx.state()).unwrap();
    ctx.register_table("test", table.table_provider().await.unwrap())
        .unwrap();
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

/// Collect the `t` column from a set of batches, ordered by the `id` column.
fn sorted_times(batches: &[RecordBatch]) -> Vec<Option<i64>> {
    let mut rows: Vec<(i32, Option<i64>)> = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let times = batch
            .column_by_name("t")
            .unwrap()
            .as_primitive::<Time64NanosecondType>();
        for (id, time) in ids.values().iter().zip(times.iter()) {
            rows.push((*id, time));
        }
    }
    rows.sort_by_key(|(id, _)| *id);
    rows.into_iter().map(|(_, time)| time).collect()
}

#[tokio::test]
async fn write_enables_the_time_nanos_table_feature() -> DeltaResult<()> {
    let table = setup_table().await?;

    let protocol = table.snapshot()?.protocol();
    for features in [protocol.reader_features(), protocol.writer_features()] {
        assert!(
            features
                .unwrap_or_default()
                .contains(&TableFeature::TimeNanos)
        );
    }
    Ok(())
}

#[tokio::test]
async fn read_returns_the_written_time_columns() -> DeltaResult<()> {
    let written = make_batch(
        vec![0, 1, 2, 3],
        vec![Some(MIDNIGHT), Some(AFTERNOON), Some(LAST_NANOSECOND), None],
    )?;
    let table = DeltaTable::new_in_memory()
        .write(vec![written.clone()])
        .await?;

    let batches = query(table, "select * from test order by id").await;

    let read = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
    let column = |batch: &RecordBatch, name: &str| batch.column_by_name(name).unwrap().clone();
    assert_eq!(&column(&read, "t"), &column(&written, "t"));
    // The stored list names its element field differently, so compare the elements.
    assert_eq!(
        column(&read, "ts").as_list::<i32>().values(),
        column(&written, "ts").as_list::<i32>().values()
    );
    Ok(())
}

#[tokio::test]
async fn append_time_nanos() -> DeltaResult<()> {
    let table = setup_table().await?;

    let table = table
        .write(vec![make_batch(vec![4], vec![Some(AFTERNOON + 1)])?])
        .await?;
    assert_eq!(table.version(), Some(1));

    let batches = query(table, "select * from test").await;
    assert_eq!(
        sorted_times(&batches),
        vec![
            Some(MIDNIGHT),
            Some(AFTERNOON),
            Some(LAST_NANOSECOND),
            None,
            Some(AFTERNOON + 1)
        ]
    );
    Ok(())
}

/// Count the files that may contain rows matching a predicate after stats-based skipping.
async fn num_files_matching(table: &DeltaTable, predicate: Predicate) -> DeltaResult<usize> {
    let files: Vec<_> = table
        .snapshot()?
        .snapshot()
        .file_views(&table.log_store(), Some(Arc::new(predicate)))
        .try_collect()
        .await?;
    Ok(files.len())
}

#[tokio::test]
async fn skip_files_with_time_nanos_predicate() -> DeltaResult<()> {
    let morning = make_batch(vec![0, 1], vec![Some(MIDNIGHT), Some(1)])?;
    // The two files differ only below the microsecond.
    let afternoon = make_batch(vec![2, 3], vec![Some(AFTERNOON), Some(AFTERNOON + 1)])?;
    let table = DeltaTable::new_in_memory().write(vec![morning]).await?;
    let table = table.write(vec![afternoon]).await?;

    let predicate = Expression::column(["t"]).gt(Scalar::TimeNanos(AFTERNOON - 1));
    assert_eq!(num_files_matching(&table, predicate).await?, 1);
    let predicate = Expression::column(["t"]).lt(Scalar::TimeNanos(2));
    assert_eq!(num_files_matching(&table, predicate).await?, 1);
    let predicate = Expression::column(["t"]).gt(Scalar::TimeNanos(0));
    assert_eq!(num_files_matching(&table, predicate).await?, 2);

    let sql = "select * from test where t > arrow_cast('13:06:07.123456789', 'Time64(Nanosecond)')";
    let batches = query(table, sql).await;
    assert_eq!(sorted_times(&batches), vec![Some(AFTERNOON + 1)]);
    Ok(())
}

#[tokio::test]
async fn delete_with_time_nanos_predicate() -> DeltaResult<()> {
    let table = setup_table().await?;

    let (table, metrics) = table
        .delete()
        .with_predicate(col("t").gt_eq(lit(ScalarValue::Time64Nanosecond(Some(AFTERNOON)))))
        .await?;
    assert_eq!(table.version(), Some(1));
    assert!(metrics.num_deleted_rows.is_some_and(|r| r == 2));

    let batches = query(table, "select * from test").await;
    assert_eq!(sorted_times(&batches), vec![Some(MIDNIGHT), None]);
    Ok(())
}

#[tokio::test]
async fn partition_by_time_nanos() -> DeltaResult<()> {
    let batch = make_batch(
        vec![0, 1, 2, 3],
        vec![Some(MIDNIGHT), Some(AFTERNOON), Some(LAST_NANOSECOND), None],
    )?;

    let table = DeltaTable::new_in_memory()
        .write(vec![batch])
        .with_partition_columns(["t"])
        .await?;
    assert_eq!(table.snapshot()?.log_data().num_files(), 4);

    let batches = query(table.clone(), "select * from test").await;
    assert_eq!(batches[0].schema().field_with_name("t")?.data_type(), &TIME);
    assert_eq!(
        sorted_times(&batches),
        vec![Some(MIDNIGHT), Some(AFTERNOON), Some(LAST_NANOSECOND), None]
    );

    let sql = "select id from test where t > arrow_cast('13:06:07', 'Time64(Nanosecond)')";
    let batches = query(table, sql).await;
    let mut ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2]);
    Ok(())
}
