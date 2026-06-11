// Test basic operations on a table that uses the float16 (half-precision) feature.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Int32Type};
use arrow_array::{FixedSizeListArray, Float16Array, Int32Array, ListArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use deltalake_core::delta_datafusion::create_session;
use deltalake_core::{DeltaResult, DeltaTable};
use half::f16;
use std::sync::Arc;

fn id_array(values: Vec<i32>) -> Arc<Int32Array> {
    Arc::new(Int32Array::from(values))
}

fn float16_array(values: Vec<f32>) -> Arc<Float16Array> {
    Arc::new(Float16Array::from(
        values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
    ))
}

fn float16_list_array(values: Vec<Vec<f32>>) -> Arc<ListArray> {
    let data = values.into_iter().map(|inner| {
        Some(
            inner
                .into_iter()
                .map(|v| Some(f16::from_f32(v)))
                .collect::<Vec<_>>(),
        )
    });
    Arc::new(ListArray::from_iter_primitive::<Float16Type, _, _>(data))
}

fn make_batch(
    schema: &Arc<Schema>,
    ids: Vec<i32>,
    values: Vec<f32>,
    list_values: Vec<Vec<f32>>,
) -> DeltaResult<RecordBatch> {
    Ok(RecordBatch::try_new(
        schema.clone(),
        vec![
            id_array(ids),
            float16_array(values),
            float16_list_array(list_values),
        ],
    )?)
}

async fn setup_table() -> DeltaResult<(DeltaTable, Arc<Schema>)> {
    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("x", DataType::Float16, true),
        Field::new_list("xs", Field::new_list_field(DataType::Float16, true), true),
    ]));

    let batch = make_batch(
        &arrow_schema,
        vec![0, 1, 2],
        vec![1.0, 2.5, 4.0],
        vec![vec![1.0, 2.5], vec![4.0], vec![8.0, 16.0, 32.0]],
    )?;

    let table: DeltaTable = DeltaTable::new_in_memory().write(vec![batch]).await?;

    Ok((table, arrow_schema))
}

async fn get_data(table: DeltaTable) -> Vec<RecordBatch> {
    let ctx: SessionContext = create_session().into();
    table.update_datafusion_session(&ctx.state()).unwrap();
    ctx.register_table("test", table.table_provider().await.unwrap())
        .unwrap();
    ctx.sql("select * from test")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
}

/// Collect the float16 `x` column from a set of batches as `f32` values, ordered by the `id` column.
fn sorted_values(batches: &Vec<RecordBatch>) -> Vec<f32> {
    let mut rows: Vec<(i32, f32)> = Vec::new();
    for batch in batches {
        let id_column = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let float_column = batch
            .column_by_name("x")
            .unwrap()
            .as_primitive::<Float16Type>();
        for i in 0..batch.num_rows() {
            rows.push((id_column.value(i), float_column.value(i).to_f32()));
        }
    }
    rows.sort_by_key(|(id, _)| *id);
    rows.into_iter().map(|(_, x)| x).collect()
}

/// Collect the `xs` list-of-float16 column from a set of batches as `f32` values, ordered by the
/// `id` column.
fn sorted_list_values(batches: &Vec<RecordBatch>) -> Vec<Vec<f32>> {
    let mut rows: Vec<(i32, Vec<f32>)> = Vec::new();
    for batch in batches {
        let id_column = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int32Type>();
        let list_column = batch.column_by_name("xs").unwrap().as_list::<i32>();
        for i in 0..batch.num_rows() {
            let inner = list_column.value(i);
            let float_column = inner.as_primitive::<Float16Type>();
            let values = (0..float_column.len())
                .map(|j| float_column.value(j).to_f32())
                .collect();
            rows.push((id_column.value(i), values));
        }
    }
    rows.sort_by_key(|(id, _)| *id);
    rows.into_iter().map(|(_, xs)| xs).collect()
}

#[tokio::test]
async fn read_float16_table() -> DeltaResult<()> {
    let (table, _) = setup_table().await?;
    let batches = get_data(table).await;

    let schema = batches[0].schema();
    let float_field = schema.field_with_name("x")?;
    assert_eq!(float_field.data_type(), &DataType::Float16);

    assert_eq!(sorted_values(&batches), vec![1.0, 2.5, 4.0]);

    let list_field = schema.field_with_name("xs")?;
    let DataType::List(inner) = list_field.data_type() else {
        panic!("expected `xs` to be a list column");
    };
    assert_eq!(inner.data_type(), &DataType::Float16);

    assert_eq!(
        sorted_list_values(&batches),
        vec![vec![1.0, 2.5], vec![4.0], vec![8.0, 16.0, 32.0]]
    );

    Ok(())
}

#[tokio::test]
async fn append_float16() -> DeltaResult<()> {
    let (table, arrow_schema) = setup_table().await?;

    let new_batch = make_batch(
        &arrow_schema,
        vec![3, 4],
        vec![8.0, 16.0],
        vec![vec![64.0], vec![128.0, 256.0]],
    )?;
    let table = table.write(vec![new_batch]).await?;
    assert_eq!(table.version(), Some(1));

    let batches = get_data(table).await;
    assert_eq!(sorted_values(&batches), vec![1.0, 2.5, 4.0, 8.0, 16.0]);
    assert_eq!(
        sorted_list_values(&batches),
        vec![
            vec![1.0, 2.5],
            vec![4.0],
            vec![8.0, 16.0, 32.0],
            vec![64.0],
            vec![128.0, 256.0],
        ]
    );

    Ok(())
}

#[tokio::test]
async fn delete_with_float16_predicate() -> DeltaResult<()> {
    let (table, _) = setup_table().await?;

    let (table, metrics) = table.delete().with_predicate("x >= 2.5").await?;
    assert_eq!(table.version(), Some(1));
    assert!(metrics.num_deleted_rows.is_some_and(|r| r == 2));

    let batches = get_data(table).await;
    assert_eq!(sorted_values(&batches), vec![1.0]);

    Ok(())
}

#[tokio::test]
async fn update_with_float16() -> DeltaResult<()> {
    let (table, _) = setup_table().await?;

    let (table, metrics) = table
        .update()
        .with_predicate("x >= 2.5")
        .with_update("x", "8.0")
        .await?;
    assert_eq!(table.version(), Some(1));
    assert_eq!(metrics.num_updated_rows, 2);

    let batches = get_data(table).await;
    assert_eq!(sorted_values(&batches), vec![1.0, 8.0, 8.0]);

    Ok(())
}

#[tokio::test]
async fn read_fixed_size_list_float16_as_list() -> DeltaResult<()> {
    // F16 is commonly used in fixed-size arrays in ML workloads,
    // so test this scenario.
    // Delta doesn't support fixed size lists, so a column written as an Arrow `FixedSizeList`
    // of float16 should be read back as a variable size `List`.
    let list_size = 2;
    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new_fixed_size_list(
            "xs",
            Field::new_list_field(DataType::Float16, true),
            list_size,
            true,
        ),
    ]));

    let make_batch = |ids: Vec<i32>, list_values: &[Vec<f32>]| -> DeltaResult<RecordBatch> {
        let fixed_size_list = Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float16Type, _, _>(
                list_values.iter().map(|inner| {
                    Some(
                        inner
                            .iter()
                            .map(|v| Some(f16::from_f32(*v)))
                            .collect::<Vec<_>>(),
                    )
                }),
                list_size,
            ),
        );
        Ok(RecordBatch::try_new(
            arrow_schema.clone(),
            vec![id_array(ids), fixed_size_list],
        )?)
    };

    let initial_values: Vec<Vec<f32>> = vec![vec![1.0, 2.5], vec![4.0, 8.0], vec![16.0, 32.0]];
    let table: DeltaTable = DeltaTable::new_in_memory()
        .write(vec![make_batch(vec![0, 1, 2], &initial_values)?])
        .await?;

    // Append more rows, which must also be cast from a fixed size list to a variable size list.
    let appended_values: Vec<Vec<f32>> = vec![vec![64.0, 128.0], vec![256.0, 512.0]];
    let table = table
        .write(vec![make_batch(vec![3, 4], &appended_values)?])
        .await?;
    assert_eq!(table.version(), Some(1));

    let batches = get_data(table).await;

    let schema = batches[0].schema();
    let list_field = schema.field_with_name("xs")?;
    let DataType::List(inner) = list_field.data_type() else {
        panic!("expected `xs` to be read back as a variable size list column");
    };
    assert_eq!(inner.data_type(), &DataType::Float16);

    let expected: Vec<Vec<f32>> = initial_values.into_iter().chain(appended_values).collect();
    assert_eq!(sorted_list_values(&batches), expected);

    Ok(())
}

#[tokio::test]
async fn merge_with_float16() -> DeltaResult<()> {
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::{col, lit};

    let (table, arrow_schema) = setup_table().await?;

    let matched_update: f32 = 16.0;
    let inserted: f32 = 8.0;

    // Source row 1 matches a target row (2.5), row 2 does not (8.0).
    let source_batch = make_batch(
        &arrow_schema,
        vec![1, 3],
        vec![2.5, inserted],
        vec![vec![2.5], vec![8.0]],
    )?;

    let ctx: SessionContext = create_session().into();
    let source = ctx.read_batch(source_batch)?;

    let (table, metrics) = table
        .merge(source, col("target.x").eq(col("source.x")))
        .with_source_alias("source")
        .with_target_alias("target")
        .when_matched_update(|update| {
            update.update(
                "x",
                lit(ScalarValue::Float16(Some(f16::from_f32(matched_update)))),
            )
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
        vec![1.0, matched_update, 4.0, inserted]
    );

    Ok(())
}
