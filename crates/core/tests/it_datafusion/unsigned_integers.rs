// Test basic operations on a table that uses the unsigned-integers feature.
//
// Values above the corresponding signed maximum are used throughout to check that they are
// never reinterpreted as negative numbers, e.g. when computing statistics or skipping files.

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{Int32Array, ListArray, RecordBatch, UInt8Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{col, lit};
use datafusion::prelude::SessionContext;
use delta_kernel::Predicate;
use delta_kernel::expressions::{Expression, Scalar};
use deltalake_core::delta_datafusion::create_session;
use deltalake_core::{DeltaResult, DeltaTable};
use futures::TryStreamExt;
use std::sync::Arc;

const LARGE: u64 = 1 << 63;

fn id_array(values: Vec<i32>) -> Arc<Int32Array> {
    Arc::new(Int32Array::from(values))
}

fn uint64_array(values: Vec<u64>) -> Arc<UInt64Array> {
    Arc::new(UInt64Array::from(values))
}

fn uint32_list_array(values: Vec<Vec<u32>>) -> Arc<ListArray> {
    let data = values
        .into_iter()
        .map(|inner| Some(inner.into_iter().map(Some).collect::<Vec<_>>()));
    Arc::new(ListArray::from_iter_primitive::<UInt32Type, _, _>(data))
}

fn make_batch(
    schema: &Arc<Schema>,
    ids: Vec<i32>,
    values: Vec<u64>,
    list_values: Vec<Vec<u32>>,
) -> DeltaResult<RecordBatch> {
    Ok(RecordBatch::try_new(
        schema.clone(),
        vec![
            id_array(ids),
            uint64_array(values),
            uint32_list_array(list_values),
        ],
    )?)
}

fn arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("x", DataType::UInt64, true),
        Field::new_list("xs", Field::new_list_field(DataType::UInt32, true), true),
    ]))
}

async fn setup_table() -> DeltaResult<(DeltaTable, Arc<Schema>)> {
    let arrow_schema = arrow_schema();

    let batch = make_batch(
        &arrow_schema,
        vec![0, 1, 2],
        vec![1, LARGE, u64::MAX],
        vec![vec![1, u32::MAX], vec![2], vec![3, 4, 5]],
    )?;

    let table: DeltaTable = DeltaTable::new_in_memory().write(vec![batch]).await?;

    Ok((table, arrow_schema))
}

async fn get_data(table: DeltaTable) -> Vec<RecordBatch> {
    query(table, "select * from test").await
}

async fn query(table: DeltaTable, sql: &str) -> Vec<RecordBatch> {
    let ctx: SessionContext = create_session().into();
    table.update_datafusion_session(&ctx.state()).unwrap();
    ctx.register_table("test", table.table_provider().await.unwrap())
        .unwrap();
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

/// Collect the `x` column from a set of batches, ordered by the `id` column.
fn sorted_values(batches: &Vec<RecordBatch>) -> Vec<u64> {
    let mut rows: Vec<(i32, u64)> = Vec::new();
    for batch in batches {
        let id_column = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let x_column = batch
            .column_by_name("x")
            .unwrap()
            .as_primitive::<UInt64Type>();
        for i in 0..batch.num_rows() {
            rows.push((id_column.value(i), x_column.value(i)));
        }
    }
    rows.sort_by_key(|(id, _)| *id);
    rows.into_iter().map(|(_, x)| x).collect()
}

/// Collect the `xs` list column from a set of batches, ordered by the `id` column.
fn sorted_list_values(batches: &Vec<RecordBatch>) -> Vec<Vec<u32>> {
    let mut rows: Vec<(i32, Vec<u32>)> = Vec::new();
    for batch in batches {
        let id_column = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let list_column = batch.column_by_name("xs").unwrap().as_list::<i32>();
        for i in 0..batch.num_rows() {
            let inner = list_column.value(i);
            let values = inner.as_primitive::<UInt32Type>().values().to_vec();
            rows.push((id_column.value(i), values));
        }
    }
    rows.sort_by_key(|(id, _)| *id);
    rows.into_iter().map(|(_, xs)| xs).collect()
}

#[tokio::test]
async fn read_unsigned_integer_table() -> DeltaResult<()> {
    let (table, _) = setup_table().await?;
    let batches = get_data(table).await;

    let schema = batches[0].schema();
    assert_eq!(schema.field_with_name("x")?.data_type(), &DataType::UInt64);

    let DataType::List(inner) = schema.field_with_name("xs")?.data_type() else {
        panic!("expected `xs` to be a list column");
    };
    assert_eq!(inner.data_type(), &DataType::UInt32);

    assert_eq!(sorted_values(&batches), vec![1, LARGE, u64::MAX]);
    assert_eq!(
        sorted_list_values(&batches),
        vec![vec![1, u32::MAX], vec![2], vec![3, 4, 5]]
    );

    Ok(())
}

#[tokio::test]
async fn append_unsigned_integers() -> DeltaResult<()> {
    let (table, arrow_schema) = setup_table().await?;

    let new_batch = make_batch(
        &arrow_schema,
        vec![3, 4],
        vec![0, u64::MAX - 1],
        vec![vec![], vec![u32::MAX - 1]],
    )?;
    let table = table.write(vec![new_batch]).await?;
    assert_eq!(table.version(), Some(1));

    let batches = get_data(table).await;
    assert_eq!(
        sorted_values(&batches),
        vec![1, LARGE, u64::MAX, 0, u64::MAX - 1]
    );
    assert_eq!(
        sorted_list_values(&batches),
        vec![
            vec![1, u32::MAX],
            vec![2],
            vec![3, 4, 5],
            vec![],
            vec![u32::MAX - 1],
        ]
    );

    Ok(())
}

#[tokio::test]
async fn filter_unsigned_integers() -> DeltaResult<()> {
    let (table, _) = setup_table().await?;

    let batches = query(table, &format!("select * from test where x >= {LARGE}")).await;
    assert_eq!(sorted_values(&batches), vec![LARGE, u64::MAX]);

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
async fn skip_files_with_unsigned_integer_predicate() -> DeltaResult<()> {
    let arrow_schema = arrow_schema();
    let small = make_batch(&arrow_schema, vec![0, 1], vec![1, 2], vec![vec![], vec![]])?;
    let large = make_batch(
        &arrow_schema,
        vec![2, 3],
        vec![LARGE, u64::MAX],
        vec![vec![], vec![]],
    )?;
    let table = DeltaTable::new_in_memory().write(vec![small]).await?;
    let table = table.write(vec![large]).await?;

    // If unsigned statistics were interpreted as signed, the file containing large values
    // would have negative min/max values and be skipped here.
    let predicate = Expression::column(["x"]).gt(Scalar::Uint64(LARGE - 1));
    assert_eq!(num_files_matching(&table, predicate).await?, 1);
    let predicate = Expression::column(["x"]).lt(Scalar::Uint64(3));
    assert_eq!(num_files_matching(&table, predicate).await?, 1);
    let predicate = Expression::column(["x"]).gt(Scalar::Uint64(0));
    assert_eq!(num_files_matching(&table, predicate).await?, 2);

    let batches = query(table, "select * from test where x > 2").await;
    assert_eq!(sorted_values(&batches), vec![LARGE, u64::MAX]);

    Ok(())
}

#[tokio::test]
async fn delete_with_unsigned_integer_predicate() -> DeltaResult<()> {
    let (table, _) = setup_table().await?;

    let (table, metrics) = table
        .delete()
        .with_predicate(col("x").gt_eq(lit(LARGE)))
        .await?;
    assert_eq!(table.version(), Some(1));
    assert!(metrics.num_deleted_rows.is_some_and(|r| r == 2));

    let batches = get_data(table).await;
    assert_eq!(sorted_values(&batches), vec![1]);

    Ok(())
}

#[tokio::test]
async fn update_with_unsigned_integers() -> DeltaResult<()> {
    let (table, _) = setup_table().await?;

    let (table, metrics) = table
        .update()
        .with_predicate(col("x").gt_eq(lit(LARGE)))
        .with_update("x", lit(u64::MAX - 1))
        .await?;
    assert_eq!(table.version(), Some(1));
    assert_eq!(metrics.num_updated_rows, 2);

    let batches = get_data(table).await;
    assert_eq!(sorted_values(&batches), vec![1, u64::MAX - 1, u64::MAX - 1]);

    Ok(())
}

#[tokio::test]
async fn merge_with_unsigned_integers() -> DeltaResult<()> {
    let (table, arrow_schema) = setup_table().await?;

    let matched_update = u64::MAX - 1;
    let inserted = LARGE + 1;

    // Source row 1 matches a target row (LARGE), row 2 does not.
    let source_batch = make_batch(
        &arrow_schema,
        vec![1, 3],
        vec![LARGE, inserted],
        vec![vec![], vec![]],
    )?;

    let ctx: SessionContext = create_session().into();
    let source = ctx.read_batch(source_batch)?;

    let (table, metrics) = table
        .merge(source, col("target.x").eq(col("source.x")))
        .with_source_alias("source")
        .with_target_alias("target")
        .when_matched_update(|update| {
            update.update("x", lit(ScalarValue::UInt64(Some(matched_update))))
        })?
        .when_not_matched_insert(|insert| {
            insert.set("id", col("source.id")).set("x", col("source.x"))
        })?
        .await?;

    assert_eq!(table.version(), Some(1));
    assert_eq!(metrics.num_target_rows_updated, 1);
    assert_eq!(metrics.num_target_rows_inserted, 1);

    let batches = get_data(table).await;
    assert_eq!(
        sorted_values(&batches),
        vec![1, matched_update, u64::MAX, inserted]
    );

    Ok(())
}

#[tokio::test]
async fn partition_by_unsigned_integer() -> DeltaResult<()> {
    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("p", DataType::UInt8, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![
            id_array(vec![0, 1, 2]),
            Arc::new(UInt8Array::from(vec![1, 200, u8::MAX])),
        ],
    )?;

    let table = DeltaTable::new_in_memory()
        .write(vec![batch])
        .with_partition_columns(["p"])
        .await?;
    assert_eq!(table.snapshot()?.log_data().num_files(), 3);

    let batches = query(table.clone(), "select * from test").await;
    let mut rows: Vec<(i32, u8)> = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let partitions = batch
            .column_by_name("p")
            .unwrap()
            .as_primitive::<UInt8Type>();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), partitions.value(i)));
        }
    }
    rows.sort();
    assert_eq!(rows, vec![(0, 1), (1, 200), (2, u8::MAX)]);

    let batches = query(table, "select id from test where p > 127").await;
    let mut ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2]);

    Ok(())
}
