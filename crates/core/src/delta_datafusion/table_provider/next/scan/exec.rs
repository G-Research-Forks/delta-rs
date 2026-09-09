//! Physical execution for Delta table scans.
//!
//! This module implements [`DeltaScanExec`], the core execution plan that reads Parquet files
//! and applies Delta Lake protocol transformations to produce logical table data.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{RecordBatch, StringArray};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{FieldRef, Schema, SchemaRef, UInt16Type};
use arrow_array::StringViewArray;
use arrow_array::{Array, ArrayRef, BooleanArray, UInt64Array};
use dashmap::DashMap;
use datafusion::common::config::ConfigOptions;
use datafusion::common::error::{DataFusionError, Result};
use datafusion::common::stats::Precision;
use datafusion::common::{
    ColumnStatistics, HashMap, internal_datafusion_err, internal_err, plan_err,
};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::{
    Distribution, EquivalenceProperties, LexOrdering, Partitioning, PhysicalSortExpr,
};
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::execution_plan::{CardinalityEffect, PlanProperties};
use datafusion::physical_plan::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, SortOrderPushdownResult, Statistics,
};
use datafusion_datasource::{
    PartitionedFile, compute_all_files_statistics,
    file_scan_config::{FileScanConfig, FileScanConfigBuilder},
    source::DataSourceExec,
};
use datafusion_physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapterFactory,
};
use delta_kernel::schema::DataType as KernelDataType;
use delta_kernel::table_features::TableFeature;
use delta_kernel::{EvaluationHandler, ExpressionRef};
use futures::stream::{Stream, StreamExt};
use indexmap::IndexMap;
use object_store::path::Path as ObjectStorePath;

use super::plan::KernelScanPlan;
use crate::delta_datafusion::file_id::file_id_field;
use crate::kernel::ARROW_HANDLER;
use crate::kernel::arrow::engine_ext::ExpressionEvaluatorExt;

#[derive(Debug, PartialEq)]
pub(crate) struct DvMaskResult {
    pub selection: Option<Vec<bool>>,
    pub should_remove: bool,
}

/// Consume the per-file deletion-vector keep-mask for the current batch.
///
/// The keep-mask is stored once per file and consumed incrementally as parquet
/// batches are produced:
/// - If the mask is shorter than the batch, missing trailing entries are
///   treated as `true` (keep row).
/// - If the mask is longer than the batch, the remainder is preserved for the
///   next batch from the same file.
///
/// This function intentionally does not error when `selection_vector.len()` is
/// greater than `batch_num_rows`; that is expected when one file spans multiple
/// input batches.
pub(crate) fn consume_dv_mask(
    selection_vector: &mut Vec<bool>,
    batch_num_rows: usize,
) -> DvMaskResult {
    if selection_vector.is_empty() {
        return DvMaskResult {
            selection: None,
            should_remove: true,
        };
    }

    if selection_vector.len() >= batch_num_rows {
        let sv: Vec<bool> = selection_vector.drain(0..batch_num_rows).collect();
        let is_empty = selection_vector.is_empty();
        DvMaskResult {
            selection: Some(sv),
            should_remove: is_empty,
        }
    } else {
        let mut sv: Vec<bool> = std::mem::take(selection_vector);
        sv.resize(batch_num_rows, true);
        DvMaskResult {
            selection: Some(sv),
            should_remove: true,
        }
    }
}

/// Map a physical parquet column name used by the inner scan back to the
/// logical column name exposed by [`DeltaScanExec`].
fn input_to_logical_column_name(scan_plan: &KernelScanPlan, physical_name: &str) -> Option<String> {
    let table_config = scan_plan.table_configuration();
    if table_config.is_feature_enabled(&TableFeature::ColumnMapping) {
        let mode = table_config.column_mapping_mode();
        scan_plan.scan.logical_schema().fields().find_map(|field| {
            let physical = field.make_physical(mode).ok()?;
            (physical.name() == physical_name).then(|| field.name().to_string())
        })
    } else {
        Some(physical_name.to_string())
    }
}

/// Derive the orderings this exec's output satisfies from the orderings of its
/// input. The per-file kernel transforms (partition value injection, column
/// mapping, deletion vector filtering) preserve row order, so any input
/// ordering whose columns survive into the output schema carries over,
/// possibly truncated to a prefix.
fn derive_output_orderings(
    scan_plan: &KernelScanPlan,
    input: &Arc<dyn ExecutionPlan>,
) -> Vec<LexOrdering> {
    let output_schema = &scan_plan.contract.output_schema;
    let mut orderings = Vec::new();
    for ordering in input
        .properties()
        .equivalence_properties()
        .oeq_class()
        .iter()
    {
        let mut mapped = Vec::new();
        for sort_expr in ordering.iter() {
            let Some(column) = sort_expr.expr.downcast_ref::<Column>() else {
                break;
            };
            let Some(logical_name) = input_to_logical_column_name(scan_plan, column.name()) else {
                break;
            };
            let Ok(index) = output_schema.index_of(&logical_name) else {
                break;
            };
            mapped.push(PhysicalSortExpr::new(
                Arc::new(Column::new(&logical_name, index)),
                sort_expr.options,
            ));
        }
        if let Some(ordering) = LexOrdering::new(mapped) {
            orderings.push(ordering);
        }
    }
    orderings
}

/// Whether `stat` describes a column holding exactly one non-null value:
/// a singleton in DataFusion's sense, with no nulls beside it.
fn is_single_valued(stat: &ColumnStatistics) -> bool {
    stat.null_count == Precision::Exact(0) && stat.is_singleton()
}

/// Descend through wrappers that hand their input's partitions on untouched -
/// same rows, same partition, same order - to whatever they wrap.
fn pass_through_leaf(plan: &Arc<dyn ExecutionPlan>) -> &Arc<dyn ExecutionPlan> {
    let mut plan = plan;
    while let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        plan = cooperative.input();
    }
    plan
}

/// A file's path and the byte range of it that a scan reads.
type FileExtent<'a> = (&'a ObjectStorePath, (u64, u64));

/// The ordered file extents of each of a parquet scan's file groups, or `None`
/// when the plan is not a single parquet scan (possibly behind pass-through
/// wrappers) whose grouping can be read.
fn file_group_extents(plan: &Arc<dyn ExecutionPlan>) -> Option<Vec<Vec<FileExtent<'_>>>> {
    let file_scan = pass_through_leaf(plan)
        .downcast_ref::<DataSourceExec>()?
        .data_source()
        .as_ref()
        .downcast_ref::<FileScanConfig>()?;
    Some(
        file_scan
            .file_groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|file| (&file.object_meta.location, file.range()))
                    .collect()
            })
            .collect(),
    )
}

/// Whether `new` reads exactly the file groups `old` did, in the same order.
///
/// Required for a [`PushedSort`] to remain compatible when the input changes.
fn grouping_is_unchanged(old: &Arc<dyn ExecutionPlan>, new: &Arc<dyn ExecutionPlan>) -> bool {
    match (file_group_extents(old), file_group_extents(new)) {
        (Some(old), Some(new)) => old == new,
        _ => false,
    }
}

/// See through a round-robin repartition to the plan that actually reads files.
fn file_reading_input(input: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    match input.downcast_ref::<RepartitionExec>() {
        Some(repartition) => matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
            .then(|| repartition.input()),
        None => Some(input),
    }
}

/// Put byte-range pieces of the same file back together, keeping first-appearance
/// order. Returns `None` if any files are only partially read.
///
/// This is required to restore correct min/max stats and be able to prove files
/// are non-overlapping, because chunks from a split file inherit the min/max stats
/// of the full file.
///
/// Each file is represented by its first piece, borrowed as it came: nothing
/// is copied for a pushdown that is then refused, and the piece's range is
/// cleared when the file is materialized (see `plan_sort_pushdown`).
fn coalesce_file_ranges<'a>(pieces: &[&'a PartitionedFile]) -> Option<Vec<&'a PartitionedFile>> {
    let mut by_path: IndexMap<&ObjectStorePath, Vec<&'a PartitionedFile>> = IndexMap::new();
    for &piece in pieces {
        by_path
            .entry(&piece.object_meta.location)
            .or_default()
            .push(piece);
    }

    by_path
        .into_values()
        .map(|group| {
            let mut ranges: Vec<(u64, u64)> = group.iter().map(|piece| piece.range()).collect();
            ranges.sort_unstable();
            let mut covered = 0;
            for (start, end) in ranges {
                if start != covered {
                    return None;
                }
                covered = end;
            }
            (covered == group[0].object_meta.size).then_some(group[0])
        })
        .collect()
}

/// Physical execution plan for scanning Delta tables.
///
/// Wraps a Parquet reader execution plan and applies Delta Lake protocol transformations
/// to produce the logical table data. This includes:
///
/// - **Column mapping**: Translates physical column names to logical names
/// - **Partition values**: Materializes partition column values from file paths
/// - **Deletion vectors**: Filters out deleted rows using per-file selection vectors
/// - **Schema evolution**: Handles missing columns and type coercion
///
/// # Data Flow
///
/// 1. Inner [`input`](Self::input) plan reads raw Parquet data
/// 2. Per-file [`transforms`](Self::transforms) convert physical to logical schema
/// 3. [`selection_vectors`](Self::selection_vectors) filter deleted rows
/// 4. Result is cast to the projected scan contract's result schema
#[derive(Clone, Debug)]
pub struct DeltaScanExec {
    scan_plan: Arc<KernelScanPlan>,
    /// Execution plan yielding the raw data read from data files.
    input: Arc<dyn ExecutionPlan>,
    /// Transforms to be applied to data eminating from individual files
    transforms: Arc<HashMap<String, ExpressionRef>>,
    /// Selection vectors to be applied to data read from individual files
    selection_vectors: Arc<DashMap<String, Vec<bool>>>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// File id column name carried by the input batches for per file correlation.
    input_file_id_column: String,
    /// User-visible file-id column name when projected in the output.
    file_id_column: Option<String>,
    /// plan properties
    properties: Arc<PlanProperties>,
    /// Aggregated partition column statistics
    partition_stats: HashMap<String, ColumnStatistics>,
    /// Resolved per-file sort order over the parquet read schema, when the table
    /// declares one (`DeltaScanConfig::file_sort_order`).
    file_sort_order: Option<LexOrdering>,
    /// Result of a successful sort pushdown, or `None` without one. Valid only
    /// for the file grouping it was computed against.
    pushed: Option<Arc<PushedSort>>,
    /// Whether any scanned file carries a deletion vector
    has_selection_vectors: bool,
}

/// Result of a successful sort pushdown, which regroups files to achieve
/// an ordered output.
#[derive(Debug)]
struct PushedSort {
    /// `[partition prefix…, file sort order prefix…]`. The file groups are
    /// mutually non-overlapping and range-ordered on it.
    ordering: LexOrdering,
    /// Statistics for the ordering's leading partition columns, keyed by logical
    /// column name and indexed by execution partition.
    per_partition_stats: Vec<HashMap<String, ColumnStatistics>>,
}

impl DisplayAs for DeltaScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // TODO: actually implement formatting according to the type
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "DeltaScanExec")?;
                if let Some(file_id_column) = &self.file_id_column {
                    write!(f, ": file_id_column={file_id_column}")?;
                }
                if let Some(row_index_field) = self.scan_plan.contract.retained_row_index_field() {
                    write!(f, ": row_index_column={}", row_index_field.name())?;
                }
                Ok(())
            }
        }
    }
}

impl DeltaScanExec {
    pub(crate) fn new(
        scan_plan: Arc<KernelScanPlan>,
        input: Arc<dyn ExecutionPlan>,
        transforms: Arc<HashMap<String, ExpressionRef>>,
        selection_vectors: Arc<DashMap<String, Vec<bool>>>,
        partition_stats: HashMap<String, ColumnStatistics>,
        metrics: ExecutionPlanMetricsSet,
    ) -> Self {
        let input_file_id_column = scan_plan.contract.file_id_field.name().to_owned();
        let file_id_column = scan_plan
            .contract
            .retain_file_id
            .then(|| scan_plan.contract.file_id_field.name().to_owned());
        let properties = Self::build_properties(&scan_plan, &input, None);
        let has_selection_vectors = !selection_vectors.is_empty();
        Self {
            scan_plan,
            input,
            transforms,
            selection_vectors,
            partition_stats,
            metrics,
            input_file_id_column,
            file_id_column,
            properties,
            file_sort_order: None,
            pushed: None,
            has_selection_vectors,
        }
    }

    /// Declare the resolved per-file sort order (over the parquet read schema).
    pub(crate) fn with_file_sort_order(mut self, file_sort_order: Option<LexOrdering>) -> Self {
        self.file_sort_order = file_sort_order;
        self
    }

    /// Build [`PlanProperties`], optionally advertising a pushed-down output
    /// ordering ahead of the orderings derived from the input.
    fn build_properties(
        scan_plan: &KernelScanPlan,
        input: &Arc<dyn ExecutionPlan>,
        pushed: Option<&PushedSort>,
    ) -> Arc<PlanProperties> {
        let mut orderings = Vec::new();
        if let Some(pushed) = pushed {
            orderings.push(pushed.ordering.clone());
        }
        orderings.extend(derive_output_orderings(scan_plan, input));
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new_with_orderings(
                Arc::clone(&scan_plan.contract.output_schema),
                orderings,
            ),
            input.properties().partitioning.clone(),
            input.properties().emission_type,
            input.properties().boundedness,
        ))
    }

    /// Offer an ordering this exec cannot serve itself to the scan underneath.
    ///
    /// The per-file work done here - partition values, column transforms,
    /// deletion vectors - keeps rows in the order they arrive, so this node is
    /// transparent to ordering and the trait asks it to delegate and re-wrap.
    /// That reaches DataFusion's own file regrouping and its `Inexact`
    /// row-group reordering, which a reversed request can use to read the
    /// interesting end of each file first.
    ///
    /// Deletion vectors must stay above it, though: they are applied by
    /// position within a file, so the `Inexact` path's row-group reordering
    /// would line the mask up against the wrong rows. That returns the wrong
    /// *data*, not merely the wrong order, and no `SortExec` above can repair
    /// it. (The tables `try_pushdown_sort` refuses outright never get here.)
    ///
    /// This is not free. `PushdownSort` offers every `SortExec` to its input,
    /// and the parquet scan answers `Inexact` for any ordering that leads with
    /// a file column, rebuilding its `FileScanConfig` - a copy of every
    /// `PartitionedFile` - and ranking the files by statistics to do so. The
    /// reordering only pays off under a `LIMIT`, where the TopK sort above can
    /// finish early; a full sort gains nothing from it. The fetch is not
    /// visible here (the rule applies it to the answer), so the cost cannot be
    /// gated on it and is accepted as plan time proportional to the file count.
    fn delegate_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        if self.has_selection_vectors {
            return Ok(SortOrderPushdownResult::Unsupported);
        }
        // A column that does not survive the crossing into the child's schema
        // - a partition column above all, which is materialised here and
        // absent from the parquet child - cannot be asked of it.
        let Some(child_order) = super::sort_pushdown::rebind_ordering(
            order,
            &self.scan_plan.contract.output_schema,
            &self.input.schema(),
        ) else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        // Push down the sort to the child, rebuilding this exec's input with the re-ordered result
        Ok(self
            .input
            .try_pushdown_sort(&child_order)?
            .map(|inner| self.with_input(inner, None)))
    }

    /// Rebuild this exec around a new input plan, recomputing plan properties.
    ///
    /// A pushed-down sort describes the file grouping `try_pushdown_sort`
    /// built, so it survives a child that was rebuilt around the same groups
    /// and nothing else. By the time a child is swapped, `PushdownSort` has
    /// already deleted the `SortExec` on the strength of that grouping, so a
    /// child whose grouping differs or cannot be read is an error: dropping
    /// the claim instead would return unsorted rows without a word.
    fn with_new_input(&self, input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(pushed) = &self.pushed
            && !Arc::ptr_eq(&self.input, &input)
            && !grouping_is_unchanged(&self.input, &input)
        {
            return internal_err!(
                "DeltaScanExec: the input was replaced by a plan whose file grouping \
                 does not match the regrouping made for the pushed-down sort `{}`; the \
                 sort above it has already been removed, so the ordering can no longer \
                 be guaranteed. Physical optimizer rules that run after `PushdownSort` \
                 must leave the parquet scan's file groups unchanged",
                pushed.ordering
            );
        }
        Ok(self.with_input(input, self.pushed.clone()))
    }

    /// Rebuild this exec around `input`, claiming `pushed` as the sort its
    /// file grouping serves, and recompute the plan properties from both.
    fn with_input(
        &self,
        input: Arc<dyn ExecutionPlan>,
        pushed: Option<Arc<PushedSort>>,
    ) -> Arc<dyn ExecutionPlan> {
        let properties = Self::build_properties(&self.scan_plan, &input, pushed.as_deref());
        Arc::new(Self {
            input,
            properties,
            pushed,
            ..self.clone()
        })
    }

    /// Transform the statistics from the inner physical parquet read plan to the logical
    /// schema we expose via the table provider. We do not attempt to provide meaningful
    /// statistics for metadata columns as we do not expect these to be useful in planning.
    /// - predicates on metadata columns (like file id) are not really useful (random etc.)
    fn map_statistics(
        &self,
        mut stats: Statistics,
        partition: Option<usize>,
    ) -> Result<Statistics> {
        // Column statistics include stats for the added file id column, so we expect the
        // number of physical schema fields + 1 to match the number of column statistics.
        // We validate this to en sure we can safely remap the statistics below.
        if self.scan_plan.scan.physical_schema().fields().len() > stats.column_statistics.len() {
            return internal_err!(
                "mismatched number of column statistics: expected {}, got {}",
                self.scan_plan.scan.physical_schema().fields().len(),
                stats.column_statistics.len()
            );
        }

        let config = self.scan_plan.table_configuration();
        let mut new_stats = Vec::with_capacity(self.schema().fields().len());

        // After a sort pushdown, prefer the exact per-execution-partition
        // partition-column statistics computed while regrouping the files.
        let per_partition = partition
            .zip(self.pushed.as_ref())
            .and_then(|(idx, pushed)| pushed.per_partition_stats.get(idx));
        // The aggregated stats span every scanned file: valid bounds for a
        // single execution partition, but exact only when there is just one -
        // or when the column holds a single value, so that no partition can
        // differ from the whole.
        let aggregate_is_exact =
            partition.is_none() || self.properties.partitioning.partition_count() == 1;
        let partition_stat = |name: &str| -> Option<ColumnStatistics> {
            if let Some(stat) = per_partition.and_then(|map| map.get(name)) {
                return Some(stat.clone());
            }
            let stat = self.partition_stats.get(name).cloned()?;
            Some(if aggregate_is_exact || is_single_valued(&stat) {
                stat
            } else {
                stat.to_inexact()
            })
        };

        if config.is_feature_enabled(&TableFeature::ColumnMapping) {
            let get_index = |name| {
                if let Some(logical) = self.scan_plan.scan.logical_schema().field(name) {
                    let physical = logical.make_physical(config.column_mapping_mode()).ok()?;
                    self.input.schema().index_of(physical.name()).ok()
                } else {
                    None
                }
            };

            for field in self.schema().fields() {
                if let Some(index) = get_index(field.name()) {
                    new_stats.push(stats.column_statistics[index].clone());
                } else if let Some(part_stat) = partition_stat(field.name()) {
                    new_stats.push(part_stat);
                } else {
                    new_stats.push(Default::default());
                }
            }
        } else {
            for field in self.schema().fields() {
                if let Some((index, _)) = self
                    .scan_plan
                    .scan
                    .physical_schema()
                    .field_with_index(field.name())
                {
                    new_stats.push(stats.column_statistics[index].clone());
                } else if let Some(part_stat) = partition_stat(field.name()) {
                    new_stats.push(part_stat);
                } else {
                    new_stats.push(Default::default());
                }
            }
        }

        stats.column_statistics = new_stats;
        Ok(stats)
    }

    /// The default physical expr adapter rewrites missing nullable columns to null literals.
    /// That rewrite supports schema evolution. Delta materialized columns live above the
    /// Parquet child, including partition values, file id, and row index. Dynamic filters
    /// on those columns must stay bound to this exec's output schema; rewriting them
    /// against the Parquet child can turn them into null literals and drop rows.
    fn references_delta_materialized_column(&self, filter: &Arc<dyn PhysicalExpr>) -> bool {
        let input_schema = self.input.schema();
        collect_columns(filter).iter().any(|column| {
            self.scan_plan
                .contract
                .result_schema
                .field_with_name(column.name())
                .is_ok()
                && input_schema.field_with_name(column.name()).is_err()
        })
    }
}

impl ExecutionPlan for DeltaScanExec {
    fn name(&self) -> &'static str {
        "DeltaScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        if self.scan_plan.contract.retained_row_index_field().is_some() {
            // Retained row indexes depend on one stream seeing each file's rows.
            vec![Distribution::SinglePartition]
        } else {
            vec![Distribution::UnspecifiedDistribution]
        }
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // When this scan declares an output ordering, partitioning the input further
        // is not desirable as this loses exact statistics and ordering information,
        // which can prevent certain optimizations such as using a sort-preserving
        // merge or progressive evaluation to handle sorted reads.
        // Unordered scans keep the default behaviour and benefit from the added
        // parallelism of round-robin repartitioning.
        let has_ordering = !self
            .properties
            .equivalence_properties()
            .oeq_class()
            .is_empty();
        vec![!has_ordering]
    }

    // TODO: setting this will fail certain tests, but why
    // fn maintains_input_order(&self) -> Vec<bool> {
    //     vec![true]
    // }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return plan_err!("DeltaScan: wrong number of children {}", children.len());
        }
        self.with_new_input(children[0].clone())
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        config: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if self.scan_plan.contract.retained_row_index_field().is_some() {
            // Each DeltaScanStream keeps row ordinal counters for one execution partition.
            // Repartitioning can split one file across streams and break ordinal contiguity.
            return Ok(None);
        }

        if self.pushed.is_some() {
            // The file groups were formed to be non-overlapping and range-ordered
            // for a pushed-down sort; re-splitting them would break that.
            return Ok(None);
        }

        if self.has_selection_vectors {
            // A deletion vector's keep mask is one shared sequence per file,
            // consumed from the front as that file's batches arrive. Byte-range
            // pieces of the file read on different streams would each drain it
            // for their own rows, applying positions that belong to the other
            // piece: deleted rows returned and live rows dropped.
            return Ok(None);
        }

        if let Some(input) = self.input.repartitioned(target_partitions, config)? {
            // Rebuild the cached properties: the new input's partitioning can
            // differ from the one this exec was built around.
            Ok(Some(self.with_new_input(input)?))
        } else {
            Ok(None)
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Normal planning enforces this through EnforceDistribution. Keep this check for
        // callers that build DeltaScanExec directly or replace its child plan.
        if self.scan_plan.contract.retained_row_index_field().is_some() {
            let input_partition_count = self.input.properties().partitioning.partition_count();
            if input_partition_count > 1 {
                return plan_err!(
                    "DeltaScanExec retained row indexes require a single input partition, got {input_partition_count}"
                );
            }
        }

        Ok(Box::pin(DeltaScanStream {
            scan_plan: Arc::clone(&self.scan_plan),
            kernel_type: Arc::clone(self.scan_plan.scan.logical_schema()).into(),
            input: self.input.execute(partition, context)?,
            baseline_metrics: BaselineMetrics::new(&self.metrics, partition),
            transforms: Arc::clone(&self.transforms),
            selection_vectors: Arc::clone(&self.selection_vectors),
            input_file_id_column: self.input_file_id_column.clone(),
            file_id_column: self.file_id_column.clone(),
            row_index_field: self.scan_plan.contract.retained_row_index_field(),
            row_index_by_file: HashMap::new(),
            pending: VecDeque::new(),
            schema_adapter: super::SchemaAdapter::new(Arc::clone(
                &self.scan_plan.contract.result_schema,
            )),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn supports_limit_pushdown(&self) -> bool {
        !self.has_selection_vectors && self.input.supports_limit_pushdown()
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn fetch(&self) -> Option<usize> {
        self.input.fetch()
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        if self.has_selection_vectors {
            // A fetch handed to the parquet scan counts raw rows, but deletion
            // vectors are applied above it, so a scan carrying any would come up
            // short. Refusing the fetch leaves the limit to a node above this one.
            return None;
        }
        // A child that took the fetch keeps its file groups; should one ever
        // not, refusing the fetch here leaves the limit above the scan.
        self.with_new_input(self.input.with_fetch(limit)?).ok()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let stats = self.input.partition_statistics(partition)?;
        self.map_statistics(Arc::unwrap_or_clone(stats), partition)
            .map(Arc::new)
    }

    /// Satisfy `ORDER BY <partition columns…>, <declared file sort order…>` by
    /// regrouping the underlying parquet files: bucket them by partition-column
    /// value, order each bucket on the file sort order, and concatenate the
    /// buckets in partition-column order. The resulting groups are mutually
    /// non-overlapping and range-ordered on the requested ordering, so the
    /// `SortExec` can be removed. See [`super::sort_pushdown`].
    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        // Neither regrouping nor delegation can serve these tables. Retained
        // row indexes are positional: they require a single input partition,
        // so there is no regrouping to do, and the row-group reordering the
        // scan below may answer with would misnumber rows. Column mapping
        // renames columns between this exec's schema and the child's; neither
        // the file-sort-order columns nor the ordering handed down are
        // translated yet.
        if self.scan_plan.contract.retained_row_index_field().is_some()
            || self
                .scan_plan
                .table_configuration()
                .is_feature_enabled(&TableFeature::ColumnMapping)
        {
            return Ok(SortOrderPushdownResult::Unsupported);
        }

        // Whatever this exec cannot serve by regrouping, the parquet scan below
        // may still be able to optimise for.
        let unsupported = || self.delegate_pushdown_sort(order);

        let Some(ordering) = LexOrdering::new(order.to_vec()) else {
            return unsupported();
        };

        // Check whether the requested ordering starts with partition columns
        // and can be handled by a regrouping.
        let Some(shape) = super::sort_pushdown::analyze_order(
            &ordering,
            &self.scan_plan.contract.output_schema,
            &self.scan_plan.parquet_read_schema,
            self.scan_plan
                .table_configuration()
                .metadata()
                .partition_columns(),
            self.file_sort_order.as_ref(),
        ) else {
            return unsupported();
        };

        // Only a single-store parquet scan is handled; multi-store unions and
        // any other wrapper are left alone.
        let Some(source) = file_reading_input(&self.input)
            .and_then(|input| input.downcast_ref::<DataSourceExec>())
        else {
            return unsupported();
        };
        let Some(file_scan) = source
            .data_source()
            .as_ref()
            .downcast_ref::<FileScanConfig>()
        else {
            return unsupported();
        };

        let pieces: Vec<&PartitionedFile> = file_scan
            .file_groups
            .iter()
            .flat_map(|group| group.iter())
            .collect();
        // A byte-range piece stands in for its whole file until the files are
        // materialized: nothing in between reads the range.
        let files: Vec<&PartitionedFile> = if pieces.iter().any(|piece| piece.range.is_some()) {
            match coalesce_file_ranges(&pieces) {
                Some(files) => files,
                None => return unsupported(),
            }
        } else {
            pieces
        };
        let parquet_table_schema = file_scan
            .file_source()
            .table_schema()
            .table_schema()
            .clone();
        let target_groups = self.input.properties().partitioning.partition_count();

        // Try to form new groups ordered by the partition columns.
        // Try to avoid expensive metadata copying before this point.
        let Some(plan) = super::sort_pushdown::plan_sort_pushdown(
            &shape,
            &self.scan_plan.parquet_read_schema,
            &files,
            target_groups,
        )?
        else {
            return unsupported();
        };

        let (file_groups, statistics) =
            compute_all_files_statistics(plan.file_groups, parquet_table_schema, true, false)?;
        // Rebuild the FileScanConfig field by field rather than through
        // `FileScanConfigBuilder::from(file_scan.clone())`: cloning the config
        // deep-copies every `PartitionedFile` only for `with_file_groups` to
        // drop them again. This code needs to be updated if a new field is added.
        let new_file_scan = FileScanConfigBuilder::new(
            file_scan.object_store_url.clone(),
            Arc::clone(file_scan.file_source()),
        )
        .with_limit(file_scan.limit)
        .with_constraints(file_scan.constraints.clone())
        .with_file_compression_type(file_scan.file_compression_type)
        .with_batch_size(file_scan.batch_size)
        .with_expr_adapter(file_scan.expr_adapter_factory.clone())
        .with_partitioned_by_file_group(file_scan.partitioned_by_file_group)
        .with_file_groups(file_groups)
        .with_statistics(statistics)
        // The regrouped files are ordered by the partition prefix first, which the
        // parquet child cannot express (partition columns are not in its schema),
        // so it declares no ordering; the combined order is advertised by this
        // exec instead. A scan that declares nothing is not order-sensitive by
        // default, and an order-insensitive scan lets DataFusion pool every file
        // into one queue for the sibling streams to share, read them in any
        // order, and let a pushed fetch prune earlier row groups - so say so
        // explicitly.
        //
        // The flag costs the work stealing and byte-range balancing an
        // order-insensitive scan gets, which is only a loss when the regrouped
        // scan ends up under a sort anyway. That happens with
        // `repartition_sorts` off: `CoalescePartitionsExec` delegates the
        // pushdown here, downgrades the `Exact` answer to `Inexact` because
        // its input had several partitions, and `PushdownSort` keeps the
        // `SortExec` over the regrouped scan. Clearing the flag would break
        // the `Exact` claim on the default path, so the trade-off stands until
        // upstream keeps the original child when it downgrades an answer it
        // cannot use.
        .with_preserve_order(true)
        .build();
        let new_input = DataSourceExec::from_data_source(new_file_scan) as Arc<dyn ExecutionPlan>;

        let pushed = Arc::new(PushedSort {
            ordering,
            per_partition_stats: plan.per_partition_stats,
        });
        Ok(SortOrderPushdownResult::Exact {
            inner: self.with_input(new_input, Some(pushed)),
        })
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        // Parent filters are bound against the logical output schema. For column mapped tables
        // the child parquet schema uses physical column names, so pushing the parent filter
        // through this exec again can rewrite it against the wrong child field. Provider level
        // predicate planning already handles the safe parquet pushdown path for these tables.
        if self
            .scan_plan
            .table_configuration()
            .is_feature_enabled(&TableFeature::ColumnMapping)
        {
            return Ok(FilterDescription::all_unsupported(
                &parent_filters,
                &self.children(),
            ));
        }

        let adapter_factory = DefaultPhysicalExprAdapterFactory {};
        let adapted_filters = adapter_factory
            .create(
                Arc::clone(&self.scan_plan.contract.result_schema),
                self.input.schema(),
            )
            .and_then(|adapter| {
                parent_filters
                    .iter()
                    .map(|filter| {
                        if self.references_delta_materialized_column(filter) {
                            // Leave filters that reference Delta materialized columns in the parent
                            // schema. `FilterDescription::from_children` marks them unsupported for
                            // the Parquet child because those columns are absent from the child schema.
                            Ok(Arc::clone(filter))
                        } else {
                            adapter.rewrite(Arc::clone(filter))
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            });

        match adapted_filters {
            Ok(filters) => FilterDescription::from_children(filters, &self.children()),
            Err(_) => Ok(FilterDescription::all_unsupported(
                &parent_filters,
                &self.children(),
            )),
        }
    }
}

/// Stream that produces logical RecordBatches from a Delta table scan.
///
/// Consumes raw Parquet data from the input stream and applies Delta Lake transformations
/// per-file to yield logical table data. Handles:
///
/// - Deletion vectors: Filters rows marked as deleted
/// - Column transforms: Applies partition value injection and column mapping
/// - Schema projection: Projects to requested columns only
/// - Type casting: Ensures output matches expected logical schema
///
/// Input batches may contain rows from multiple file IDs (e.g., due to upstream coalescing).
/// The stream splits such batches by contiguous file-id runs and applies per-file transforms.
struct DeltaScanStream {
    scan_plan: Arc<KernelScanPlan>,
    /// Kernel data type for the data after transformations
    kernel_type: KernelDataType,
    /// Input stream yielding raw data read from data files.
    input: SendableRecordBatchStream,
    /// Execution metrics
    baseline_metrics: BaselineMetrics,
    /// Transforms to be applied to data read from individual files
    transforms: Arc<HashMap<String, ExpressionRef>>,
    /// Selection vectors to be applied to data read from individual files
    selection_vectors: Arc<DashMap<String, Vec<bool>>>,
    /// File id column name carried by the input batches for per file correlation.
    input_file_id_column: String,
    /// User-visible file-id column name when projected in the output.
    file_id_column: Option<String>,
    /// Row index field included in projected output.
    row_index_field: Option<FieldRef>,
    /// Per file ordinal state for this execution partition.
    ///
    /// `DataSourceExec` assigns whole `PartitionedFile`s to file groups. Each physical file has
    /// one scan stream partition owner.
    row_index_by_file: HashMap<String, u64>,
    pending: VecDeque<RecordBatch>,
    /// Cached schema adapter for efficient batch adaptation across batches
    schema_adapter: super::SchemaAdapter,
}

impl DeltaScanStream {
    fn batch_project(&mut self, batch: RecordBatch) -> Result<Vec<RecordBatch>> {
        // Clone the metric so the timer guard does not immutably borrow `self`,
        // which would conflict with the `&mut self` calls below.
        let elapsed = self.baseline_metrics.elapsed_compute().clone();
        let _timer = elapsed.timer();

        if batch.num_rows() == 0 {
            return Ok(vec![RecordBatch::new_empty(self.schema())]);
        }

        let file_id_idx = file_id_column_idx(&batch, &self.input_file_id_column)?;
        let file_runs = split_by_file_id_runs(&batch, file_id_idx)?;

        let mut results = Vec::with_capacity(file_runs.len());
        for (file_id, slice) in file_runs {
            results.push(self.batch_project_single_file(slice, file_id, file_id_idx)?);
        }
        Ok(results)
    }

    fn batch_project_single_file(
        &mut self,
        batch: RecordBatch,
        file_id: String,
        file_id_idx: usize,
    ) -> Result<RecordBatch> {
        let dv_result = if let Some(mut selection_vector) = self.selection_vectors.get_mut(&file_id)
        {
            consume_dv_mask(&mut selection_vector, batch.num_rows())
        } else {
            DvMaskResult {
                selection: None,
                should_remove: false,
            }
        };

        if dv_result.should_remove {
            self.selection_vectors.remove(&file_id);
        }

        let mut batch = if let Some(selection) = dv_result.selection {
            filter_record_batch(&batch, &BooleanArray::from(selection))?
        } else {
            batch
        };

        let file_id_col = batch.remove_column(file_id_idx);

        let result = if let Some(transform) = self.transforms.get(&file_id) {
            let evaluator = ARROW_HANDLER
                .new_expression_evaluator(
                    self.scan_plan.scan.physical_schema().clone(),
                    transform.clone(),
                    self.kernel_type.clone(),
                )
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            evaluator
                .evaluate_arrow(batch)
                .map_err(|e| DataFusionError::External(Box::new(e)))?
        } else {
            batch
        };

        let result = if let Some(file_id_column) = &self.file_id_column {
            super::finalize_transformed_batch(
                result,
                &self.scan_plan,
                Some((file_id_col, file_id_field(Some(file_id_column)))),
                &mut self.schema_adapter,
            )
        } else {
            super::finalize_transformed_batch(
                result,
                &self.scan_plan,
                None,
                &mut self.schema_adapter,
            )
        }?;

        self.append_row_index(result, &file_id)
    }

    fn append_row_index(&mut self, batch: RecordBatch, file_id: &str) -> Result<RecordBatch> {
        let Some(row_index_field) = self.row_index_field.clone() else {
            return Ok(batch);
        };

        let row_count = u64::try_from(batch.num_rows()).map_err(|_| {
            internal_datafusion_err!("batch row count does not fit u64 while assigning row indexes")
        })?;
        let next_row_index = self
            .row_index_by_file
            .entry(file_id.to_string())
            .or_default();
        let end = next_row_index.checked_add(row_count).ok_or_else(|| {
            internal_datafusion_err!(
                "row index overflow while assigning row indexes for file '{file_id}'"
            )
        })?;

        let values = if row_count == 0 {
            Vec::new()
        } else {
            ((*next_row_index + 1)..=end).collect()
        };
        *next_row_index = end;

        let row_index: ArrayRef = Arc::new(UInt64Array::from(values));
        let mut columns = batch.columns().to_vec();
        columns.push(row_index);
        let mut fields = batch.schema().fields().to_vec();
        fields.push(row_index_field);

        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }
}

impl Stream for DeltaScanStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(batch) = self.pending.pop_front() {
            return self
                .baseline_metrics
                .record_poll(Poll::Ready(Some(Ok(batch))));
        }

        let poll = self.input.poll_next_unpin(cx).map(|x| match x {
            Some(Ok(batch)) => {
                let projected = match self.batch_project(batch) {
                    Ok(outputs) => {
                        let mut outputs = outputs.into_iter();
                        match outputs.next() {
                            Some(first) => {
                                self.pending.extend(outputs);
                                Ok(first)
                            }
                            None => {
                                Err(internal_datafusion_err!("batch_project returned no output"))
                            }
                        }
                    }
                    Err(err) => Err(err),
                };
                Some(projected)
            }
            other => other,
        });

        self.baseline_metrics.record_poll(poll)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, _high) = self.input.size_hint();
        (self.pending.len() + low, None)
    }
}

impl RecordBatchStream for DeltaScanStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.scan_plan.contract.output_schema)
    }
}

#[inline]
fn file_id_column_idx(batch: &RecordBatch, file_id_column: &str) -> Result<usize> {
    batch
        .schema_ref()
        .fields()
        .iter()
        .position(|f| f.name() == file_id_column)
        .ok_or_else(|| {
            internal_datafusion_err!(
                "Expected column '{}' to be present in the input",
                file_id_column
            )
        })
}

/// Split batch into contiguous runs by file ID. Compares dictionary keys for efficiency.
/// Returns zero-copy slices. Errors on null file IDs or unexpected column type.
fn split_by_file_id_runs(
    batch: &RecordBatch,
    file_id_idx: usize,
) -> Result<Vec<(String, RecordBatch)>> {
    if batch.num_rows() == 0 {
        return Ok(Vec::new());
    }

    let dict = batch
        .column(file_id_idx)
        .as_any()
        .downcast_ref::<arrow_array::DictionaryArray<UInt16Type>>()
        .ok_or_else(|| {
            internal_datafusion_err!(
                "Expected file id column '{}' to be Dictionary<UInt16, Utf8|Utf8View>, got {:?}",
                batch.schema_ref().field(file_id_idx).name(),
                batch.column(file_id_idx).data_type()
            )
        })?;

    // Parquet reads may yield Utf8 or Utf8View depending on DataFusion settings.
    // Accept either for the synthetic file id column.
    let keys = dict.keys();

    enum FileIdValues<'a> {
        Utf8(&'a StringArray),
        Utf8View(&'a StringViewArray),
    }

    let values = if let Some(values) = dict
        .values()
        .as_ref()
        .as_any()
        .downcast_ref::<StringArray>()
    {
        FileIdValues::Utf8(values)
    } else if let Some(values) = dict
        .values()
        .as_ref()
        .as_any()
        .downcast_ref::<StringViewArray>()
    {
        FileIdValues::Utf8View(values)
    } else {
        return Err(internal_datafusion_err!(
            "Expected file id column '{}' to be Dictionary<UInt16, Utf8|Utf8View>, got {:?}",
            batch.schema_ref().field(file_id_idx).name(),
            batch.column(file_id_idx).data_type()
        ));
    };

    let file_id_for_row = |row: usize| -> String {
        let key = keys.value(row) as usize;
        match values {
            FileIdValues::Utf8(arr) => arr.value(key).to_string(),
            FileIdValues::Utf8View(arr) => arr.value(key).to_string(),
        }
    };

    if dict.is_null(0) {
        return Err(internal_datafusion_err!("file id value must not be null"));
    }

    let mut prev_key = keys.value(0);
    let mut start = 0usize;
    let mut runs = Vec::new();

    for i in 1..batch.num_rows() {
        if dict.is_null(i) {
            return Err(internal_datafusion_err!("file id value must not be null"));
        }
        let key = keys.value(i);
        if key != prev_key {
            let file_id = file_id_for_row(start);
            runs.push((file_id, batch.slice(start, i - start)));
            start = i;
            prev_key = key;
        }
    }

    let file_id = file_id_for_row(start);
    runs.push((file_id, batch.slice(start, batch.num_rows() - start)));

    Ok(runs)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::AsArray;
    use arrow::datatypes::{DataType, Field, Schema, UInt64Type};
    use arrow_array::Array;
    use arrow_array::ArrayAccessor;
    use datafusion::{
        common::{ToDFSchema, stats::Precision},
        datasource::TableProvider,
        logical_expr::Operator,
        physical_expr::expressions::{
            BinaryExpr, Column, DynamicFilterPhysicalExpr, lit as physical_lit,
        },
        physical_expr::{Distribution, Partitioning},
        physical_plan::{
            PhysicalExpr, collect, collect_partitioned,
            filter_pushdown::{FilterPushdownPhase, PushedDown},
            repartition::RepartitionExec,
        },
        prelude::{col, lit},
        scalar::ScalarValue,
    };

    use super::*;
    use crate::{
        assert_batches_sorted_eq,
        delta_datafusion::{
            DeltaScanConfig, session::create_session, table_provider::next::FILE_ID_COLUMN_DEFAULT,
        },
        test_utils::{TestResult, TestTables, open_fs_path},
    };

    #[tokio::test]
    async fn test_scan_nested() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/nested_types/delta");
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());

        let scan = provider.scan(&session.state(), None, &[], None).await?;

        let batches = collect(scan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-----------------------------+-----------------+--------------------------+",
            "| pk | struct                      | array           | map                      |",
            "+----+-----------------------------+-----------------+--------------------------+",
            "| 0  | {float64: 0.0, bool: true}  | [0]             | {}                       |",
            "| 1  | {float64: 1.0, bool: false} | [0, 1]          | {0: 0}                   |",
            "| 2  | {float64: 2.0, bool: true}  | [0, 1, 2]       | {0: 0, 1: 1}             |",
            "| 3  | {float64: 3.0, bool: false} | [0, 1, 2, 3]    | {0: 0, 1: 1, 2: 2}       |",
            "| 4  | {float64: 4.0, bool: true}  | [0, 1, 2, 3, 4] | {0: 0, 1: 1, 2: 2, 3: 3} |",
            "+----+-----------------------------+-----------------+--------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = Arc::new(create_session().into_inner());

        let scan = provider
            .scan(&session.state(), None, &[col("letter").eq(lit("b"))], None)
            .await?;

        let downcast = scan.downcast_ref::<DeltaScanExec>();
        assert!(downcast.is_some());
        assert_eq!(downcast.unwrap().file_id_column.as_deref(), Some("file_id"));

        let data = collect_partitioned(scan, session.task_ctx())
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        // Verify that file_id column is present in the result
        assert!(data[0].schema().column_with_name("file_id").is_some());
        assert_eq!(data[0].num_rows(), 1);

        // Verify file_id column has the correct type
        let schema = data[0].schema();
        let file_id_field = schema.column_with_name("file_id").unwrap().1;
        match file_id_field.data_type() {
            DataType::Dictionary(_, value_type)
                if value_type.as_ref() == &DataType::Utf8
                    || value_type.as_ref() == &DataType::Utf8View =>
            {
                // ok
            }
            other => panic!("unexpected file_id dtype: {other:?}"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id_projection() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = Arc::new(create_session().into_inner());

        // Select only data and file_id columns (both can be satisfied from metadata)
        let data_idx = provider.schema().index_of("data").unwrap();
        let file_id_idx = provider.schema().index_of("file_id").unwrap();

        let scan = provider
            .scan(
                &session.state(),
                Some(&vec![data_idx, file_id_idx]),
                &[col("letter").eq(lit("b"))],
                None,
            )
            .await?;

        // Scan could be either DeltaScanExec or DeltaScanMetaExec depending on whether
        // data column requires physical file access
        let data = collect_partitioned(scan, session.task_ctx())
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        // Should have 2 columns: data and file_id
        assert_eq!(data[0].num_columns(), 2);
        assert!(data[0].schema().column_with_name("data").is_some());
        assert!(data[0].schema().column_with_name("file_id").is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id_provider_does_not_force_output_when_unprojected() -> TestResult
    {
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = Arc::new(create_session().into_inner());
        let id_idx = provider.schema().index_of("id").unwrap();

        let scan = provider
            .scan(&session.state(), Some(&vec![id_idx]), &[], None)
            .await?;

        let downcast = scan.downcast_ref::<DeltaScanExec>();
        assert!(downcast.is_some());
        assert!(downcast.unwrap().file_id_column.is_none());

        let data = collect_partitioned(scan, session.task_ctx())
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        assert!(!data.is_empty());
        assert_eq!(data[0].num_columns(), 1);
        assert!(data[0].schema().column_with_name("id").is_some());
        assert!(data[0].schema().column_with_name("file_id").is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_gather_filters_for_pushdown_adapts_override_schema_predicates() -> TestResult {
        let mut table =
            open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        table.load().await?;

        let provider = crate::delta_datafusion::table_provider::next::DeltaScan::new(
            table.snapshot()?.snapshot().clone(),
            DeltaScanConfig::default().with_schema(
                crate::delta_datafusion::table_provider::next::test_multi_partitioned_override_schema(),
            ),
        )?
        .with_log_store(table.log_store());

        let session = Arc::new(create_session().into_inner());
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");

        let filter = session.state().create_physical_expr(
            col("number").lt(lit(ScalarValue::TimestampMillisecond(Some(7), None))),
            &exec.schema().clone().to_dfschema()?,
        )?;

        let description = exec.gather_filters_for_pushdown(
            FilterPushdownPhase::Pre,
            vec![filter],
            session.state().config().options(),
        )?;

        let child_filters = description.parent_filters();
        assert_eq!(child_filters.len(), 1);
        assert_eq!(child_filters[0].len(), 1);
        assert!(matches!(child_filters[0][0].discriminant, PushedDown::Yes));

        let input_batches = collect(Arc::clone(&exec.input), session.task_ctx()).await?;
        assert!(!input_batches.is_empty());
        child_filters[0][0].predicate.evaluate(&input_batches[0])?;

        Ok(())
    }

    #[tokio::test]
    async fn test_gather_filters_for_pushdown_rejects_delta_materialized_dynamic_filters()
    -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");

        let letter_idx = exec.schema().index_of("letter")?;
        assert!(exec.schema().field_with_name("letter").is_ok());
        assert!(exec.input.schema().field_with_name("letter").is_err());

        let letter: Arc<dyn PhysicalExpr> = Arc::new(Column::new("letter", letter_idx));
        let predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&letter),
            Operator::Eq,
            physical_lit("a"),
        ));
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(vec![letter], predicate));

        let description = exec.gather_filters_for_pushdown(
            FilterPushdownPhase::Post,
            vec![filter],
            session.state().config().options(),
        )?;

        let child_filters = description.parent_filters();
        assert_eq!(child_filters.len(), 1);
        assert_eq!(child_filters[0].len(), 1);
        assert!(matches!(child_filters[0][0].discriminant, PushedDown::No));

        Ok(())
    }

    #[tokio::test]
    async fn test_gather_filters_for_pushdown_rejects_mixed_delta_materialized_dynamic_filters()
    -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");

        let letter_idx = exec.schema().index_of("letter")?;
        let number_idx = exec.schema().index_of("number")?;
        assert!(exec.schema().field_with_name("letter").is_ok());
        assert!(exec.schema().field_with_name("number").is_ok());
        assert!(exec.input.schema().field_with_name("letter").is_err());
        assert!(exec.input.schema().field_with_name("number").is_ok());

        let letter: Arc<dyn PhysicalExpr> = Arc::new(Column::new("letter", letter_idx));
        let number: Arc<dyn PhysicalExpr> = Arc::new(Column::new("number", number_idx));
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![letter, number],
            physical_lit(true),
        ));

        let description = exec.gather_filters_for_pushdown(
            FilterPushdownPhase::Post,
            vec![filter],
            session.state().config().options(),
        )?;

        let child_filters = description.parent_filters();
        assert_eq!(child_filters.len(), 1);
        assert_eq!(child_filters[0].len(), 1);
        assert!(matches!(child_filters[0][0].discriminant, PushedDown::No));

        Ok(())
    }

    #[tokio::test]
    async fn test_gather_filters_for_pushdown_keeps_parquet_dynamic_filters() -> TestResult {
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");

        let id_idx = exec.schema().index_of("id")?;
        assert!(exec.schema().field_with_name("id").is_ok());
        assert!(exec.input.schema().field_with_name("id").is_ok());

        let data: Arc<dyn PhysicalExpr> = Arc::new(Column::new("id", id_idx));
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![data],
            physical_lit(true),
        ));

        let description = exec.gather_filters_for_pushdown(
            FilterPushdownPhase::Post,
            vec![filter],
            session.state().config().options(),
        )?;

        let child_filters = description.parent_filters();
        assert_eq!(child_filters.len(), 1);
        assert_eq!(child_filters[0].len(), 1);
        assert!(matches!(child_filters[0][0].discriminant, PushedDown::Yes));

        Ok(())
    }

    #[tokio::test]
    async fn test_gather_filters_for_pushdown_skips_column_mapping_parent_filters() -> TestResult {
        let mut table = open_fs_path("../test/tests/data/table_with_column_mapping");
        table.load().await?;

        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");

        let filter = session.state().create_physical_expr(
            col(r#""Super Name""#).eq(lit(ScalarValue::Utf8View(Some("Timothy Lamb".to_string())))),
            &exec.schema().clone().to_dfschema()?,
        )?;

        let description = exec.gather_filters_for_pushdown(
            FilterPushdownPhase::Pre,
            vec![filter],
            session.state().config().options(),
        )?;

        let child_filters = description.parent_filters();
        assert_eq!(child_filters.len(), 1);
        assert_eq!(child_filters[0].len(), 1);
        assert!(matches!(child_filters[0][0].discriminant, PushedDown::No));

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id_groupby() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = Arc::new(create_session().into_inner());

        session.register_table("delta_table", provider).unwrap();

        // Query that groups by file_id to verify each file's contribution
        let df = session
            .sql("SELECT file_id, COUNT(*) as count FROM delta_table GROUP BY file_id ORDER BY file_id")
            .await
            .unwrap();
        let batches = df.collect().await?;

        // Should have 2 or more groups (one for each file)
        assert!(batches[0].num_rows() >= 2);
        assert_eq!(batches[0].num_columns(), 2);

        // Verify file_id column is present
        assert!(batches[0].schema().column_with_name("file_id").is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_without_file_id() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().await?;
        let session = create_session().into_inner();

        let scan = provider
            .scan(&session.state(), None, &[col("letter").eq(lit("b"))], None)
            .await?;

        let downcast = scan.downcast_ref::<DeltaScanExec>();
        assert!(downcast.is_some());
        assert!(downcast.unwrap().file_id_column.is_none());

        let data = collect_partitioned(scan, session.task_ctx())
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        // Verify that file_id column is NOT present when not requested
        assert!(data[0].schema().column_with_name("file_id").is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id_all_data() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = create_session().into_inner();

        session.register_table("delta_table", provider).unwrap();

        // Query to verify file_id is present for all rows
        let df = session
            .sql("SELECT data, letter, file_id FROM delta_table WHERE letter = 'b'")
            .await
            .unwrap();
        let batches = df.collect().await?;

        // Verify the result has the expected structure
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[0].num_columns(), 3);

        // Verify all expected columns are present
        assert!(batches[0].schema().column_with_name("data").is_some());
        assert!(batches[0].schema().column_with_name("letter").is_some());
        assert!(batches[0].schema().column_with_name("file_id").is_some());

        // Verify file_id column has a value (full file path)
        let file_id_col = batches[0].column_by_name("file_id").unwrap();
        assert_eq!(file_id_col.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id_filter_omits_unprojected_file_id_from_final_output()
    -> TestResult {
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = Arc::new(create_session().into_inner());

        session
            .register_table("delta_table", provider.clone())
            .unwrap();

        let file_id_batches = session
            .sql("SELECT CAST(file_id AS STRING) AS file_id FROM delta_table LIMIT 1")
            .await
            .unwrap()
            .collect()
            .await?;
        let file_id = file_id_batches[0].column(0).as_string_view().value(0);
        let file_id = file_id.replace('\'', "''");

        let df = session
            .sql(&format!(
                "SELECT id FROM delta_table WHERE file_id = '{file_id}'"
            ))
            .await
            .unwrap();
        let batches = df.collect().await?;

        assert_eq!(batches[0].num_columns(), 1);
        assert!(batches[0].schema().column_with_name("id").is_some());
        assert!(batches[0].schema().column_with_name("file_id").is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id_extract_filename() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = Arc::new(create_session().into_inner());

        session.register_table("delta_table", provider).unwrap();

        // Extract just the filename from the full path using SQL
        // Use REVERSE and STRPOS to find the last '/' and extract everything after it
        let df = session
            .sql(
                "SELECT
                    data,
                    letter,
                    REVERSE(SUBSTRING(REVERSE(file_id), 1, STRPOS(REVERSE(file_id), '/') - 1)) as filename
                 FROM delta_table
                 WHERE letter = 'b'"
            )
            .await
            .unwrap();
        let batches = df.collect().await?;

        // Verify the filename contains expected patterns (UUID and .parquet extension)
        let expected = vec![
            "+----------+--------+---------------------------------------------------------------------+",
            "| data     | letter | filename                                                            |",
            "+----------+--------+---------------------------------------------------------------------+",
            "| f09f9888 | b      | part-00000-b300ccc0-7096-4f4f-acf9-3811211dca3e.c000.snappy.parquet |",
            "+----------+--------+---------------------------------------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id_multiple_files() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = Arc::new(create_session().into_inner());

        session.register_table("delta_table", provider).unwrap();

        // Query all data and extract filenames
        let df = session
            .sql(
                "SELECT
                    letter,
                    COUNT(*) as count,
                    REVERSE(SUBSTRING(REVERSE(file_id), 1, STRPOS(REVERSE(file_id), '/') - 1)) as filename
                 FROM delta_table
                 GROUP BY letter, file_id
                 ORDER BY letter, filename"
            )
            .await
            .unwrap();
        let batches = df.collect().await?;

        // Should have multiple groups (one for each unique file)
        assert!(batches[0].num_rows() >= 2);

        // Verify columns are present
        assert!(batches[0].schema().column_with_name("letter").is_some());
        assert!(batches[0].schema().column_with_name("count").is_some());
        assert!(batches[0].schema().column_with_name("filename").is_some());

        // Verify each group has a valid parquet filename
        let filename_col = batches[0]
            .column_by_name("filename")
            .unwrap()
            .as_string_view();

        for i in 0..filename_col.len() {
            let filename = filename_col.value(i);
            assert!(
                filename.ends_with(".parquet"),
                "Filename should end with .parquet: {}",
                filename
            );
            assert!(
                filename.contains("part-"),
                "Filename should contain 'part-': {}",
                filename
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_file_id_data_validation() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = Arc::new(create_session().into_inner());

        session.register_table("delta_table", provider).unwrap();

        // Query to validate that file_id is present for each partition
        let df = session
            .sql(
                "SELECT
                    letter,
                    data,
                    REVERSE(SUBSTRING(REVERSE(file_id), 1, STRPOS(REVERSE(file_id), '/') - 1)) as filename
                 FROM delta_table
                 WHERE letter = 'b'
                 ORDER BY letter"
            )
            .await
            .unwrap();
        let batches = df.collect().await?;

        let expected = vec![
            "+--------+----------+---------------------------------------------------------------------+",
            "| letter | data     | filename                                                            |",
            "+--------+----------+---------------------------------------------------------------------+",
            "| b      | f09f9888 | part-00000-b300ccc0-7096-4f4f-acf9-3811211dca3e.c000.snappy.parquet |",
            "+--------+----------+---------------------------------------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_statistics() -> TestResult {
        let mut table =
            open_fs_path("../../dat/v0.0.3/reader_tests/generated/all_primitive_types/delta");
        table.load().await?;
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());

        // for scans without prodicates, we gather only top level statistic
        // and omit collecting column level statistics
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let statistics = scan.partition_statistics(None)?;
        assert_eq!(statistics.num_rows, Precision::Exact(5));
        assert_eq!(statistics.total_byte_size, Precision::Inexact(3240));
        for col_stat in statistics.column_statistics.iter() {
            assert_eq!(col_stat.null_count, Precision::Absent);
            assert_eq!(col_stat.min_value, Precision::Absent);
            assert_eq!(col_stat.max_value, Precision::Absent);
        }

        // for scans with predicates, we gather full statistics
        let predicates = table
            .snapshot()?
            .schema()
            .field_names()
            .map(|c| col(c).is_not_null())
            .collect::<Vec<_>>();
        let scan = provider
            .scan(&session.state(), None, &predicates, None)
            .await?;
        let statistics = scan.partition_statistics(None)?;
        for (col_stat, field) in statistics
            .column_statistics
            .iter()
            .zip(provider.schema().fields())
        {
            // skip boolean and binary columns as they do not have min/max stats
            if matches!(
                field.data_type(),
                &DataType::Boolean | &DataType::Binary | &DataType::BinaryView
            ) {
                assert!(matches!(col_stat.null_count, Precision::Inexact(_)));
                continue;
            }
            assert!(matches!(col_stat.null_count, Precision::Inexact(_)));
            assert!(matches!(col_stat.min_value, Precision::Inexact(_)));
            assert!(matches!(col_stat.max_value, Precision::Inexact(_)));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_statistics_column_mapping() -> TestResult {
        let mut table =
            open_fs_path("../../dat/v0.0.3/reader_tests/generated/column_mapping/delta");
        table.load().await?;
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());

        // for scans with predicates, we gather full statistics
        let predicates = table
            .snapshot()?
            .schema()
            .field_names()
            .map(|c| col(c).is_not_null())
            .collect::<Vec<_>>();
        let scan = provider
            .scan(&session.state(), None, &predicates, None)
            .await?;
        let statistics = scan.partition_statistics(None)?;
        assert_eq!(
            statistics.column_statistics.len(),
            provider.schema().fields().len()
        );
        for col_stat in statistics.column_statistics.iter() {
            assert!(matches!(col_stat.null_count, Precision::Inexact(_)));
            assert!(matches!(col_stat.min_value, Precision::Inexact(_)));
            assert!(matches!(col_stat.max_value, Precision::Inexact(_)));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_statistics_partitioned() -> TestResult {
        let mut table =
            open_fs_path("../../dat/v0.0.3/reader_tests/generated/multi_partitioned/delta");
        table.load().await?;
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());

        // for scans with predicates, we gather full statistics
        let predicates = table
            .snapshot()?
            .schema()
            .field_names()
            .map(|c| col(c).is_not_null())
            .collect::<Vec<_>>();
        let scan = provider
            .scan(&session.state(), None, &predicates, None)
            .await?;
        let statistics = scan.partition_statistics(None)?;
        for (col_stat, _field) in statistics
            .column_statistics
            .iter()
            .zip(provider.schema().fields())
        {
            assert!(matches!(
                col_stat.null_count,
                Precision::Exact(_) | Precision::Inexact(_)
            ));
            assert!(matches!(
                col_stat.min_value,
                Precision::Exact(_) | Precision::Inexact(_)
            ));
            assert!(matches!(
                col_stat.max_value,
                Precision::Exact(_) | Precision::Inexact(_)
            ));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_with_deletion_vectors() -> TestResult {
        let table = open_fs_path("../../dat/v0.0.3/reader_tests/generated/deletion_vectors/delta");
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());

        let scan = provider.scan(&session.state(), None, &[], None).await?;

        let downcast = scan.downcast_ref::<DeltaScanExec>();
        assert!(downcast.is_some(), "Expected DeltaScanExec for DV test");

        let batches = collect(scan, session.task_ctx()).await?;

        let expected = vec![
            "+--------+-----+------------+",
            "| letter | int | date       |",
            "+--------+-----+------------+",
            "| b      | 228 | 1978-12-01 |",
            "+--------+-----+------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    // DV test helpers
    const DV_TABLE_PATH: &str = "../../dat/v0.0.3/reader_tests/generated/deletion_vectors/delta";

    async fn dv_kernel_type_and_int32_scan_plan()
    -> TestResult<(KernelDataType, Arc<KernelScanPlan>)> {
        use arrow::datatypes::{Field, Schema};

        let table = open_fs_path(DV_TABLE_PATH);
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());

        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("Expected DeltaScanExec");

        let kernel_type = Arc::clone(exec.scan_plan.scan.logical_schema()).into();

        let mut scan_plan = exec.scan_plan.as_ref().clone();
        scan_plan.contract.result_schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        scan_plan.contract.output_schema = Arc::clone(&scan_plan.contract.result_schema);
        scan_plan.contract.result_projection = None;
        scan_plan.parquet_read_schema = Arc::clone(&scan_plan.contract.result_schema);

        Ok((kernel_type, Arc::new(scan_plan)))
    }

    async fn int32_scan_plan() -> TestResult<(KernelDataType, Arc<KernelScanPlan>)> {
        use arrow::datatypes::{Field, Schema};

        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());

        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("Expected DeltaScanExec");

        let kernel_type = Arc::clone(exec.scan_plan.scan.logical_schema()).into();

        let mut scan_plan = exec.scan_plan.as_ref().clone();
        scan_plan.contract.result_schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        scan_plan.contract.output_schema = Arc::clone(&scan_plan.contract.result_schema);
        scan_plan.contract.result_projection = None;
        scan_plan.parquet_read_schema = Arc::clone(&scan_plan.contract.result_schema);

        Ok((kernel_type, Arc::new(scan_plan)))
    }

    /// Create a PartitionedFile with a specified range
    fn piece(path: &str, size: u64, range: Option<(i64, i64)>) -> PartitionedFile {
        let mut file = PartitionedFile::new(path.to_string(), size);
        file.range = range.map(|(start, end)| datafusion_datasource::FileRange { start, end });
        file
    }

    /// Wrapper around [`coalesce_file_ranges`] for testing: the path and byte
    /// range of each representative piece.
    fn coalesced_paths(pieces: &[PartitionedFile]) -> Option<Vec<(String, (u64, u64))>> {
        let refs: Vec<&PartitionedFile> = pieces.iter().collect();

        coalesce_file_ranges(&refs).map(|coalesced| {
            coalesced
                .into_iter()
                .map(|file| (file.object_meta.location.to_string(), file.range()))
                .collect()
        })
    }

    /// Pieces that tile their file are represented by the first of them,
    /// keeping the order the first piece of each file appeared in.
    #[test]
    fn test_coalesce_file_ranges_reassembles_tiling_pieces() {
        let pieces = vec![
            piece("a", 100, Some((0, 40))),
            piece("b", 60, Some((0, 60))),
            piece("a", 100, Some((40, 100))),
        ];
        assert_eq!(
            coalesced_paths(&pieces),
            Some(vec![("a".to_string(), (0, 40)), ("b".to_string(), (0, 60))])
        );
    }

    /// Coalescing file ranges should fail if any file is not fully covered
    #[test]
    fn test_coalesce_file_ranges_refuses_a_partial_file() {
        assert_eq!(coalesced_paths(&[piece("a", 100, Some((0, 40)))]), None);
        assert_eq!(
            coalesced_paths(&[
                piece("a", 100, Some((0, 40))),
                piece("a", 100, Some((60, 100))),
            ]),
            None
        );
        // A whole-file entry alongside pieces of the same file should also be rejected,
        // otherwise this would be read twice over.
        assert_eq!(
            coalesced_paths(&[piece("a", 100, None), piece("a", 100, Some((0, 100)))]),
            None
        );
    }

    /// `(path, size, byte range)` of one file in a test scan.
    type FileSpec<'a> = (&'a str, u64, Option<(i64, i64)>);

    /// A parquet scan over `groups`.
    fn parquet_scan(groups: &[&[FileSpec<'_>]]) -> Arc<dyn ExecutionPlan> {
        use datafusion::datasource::physical_plan::ParquetSource;
        use datafusion::execution::object_store::ObjectStoreUrl;
        use datafusion_datasource::TableSchema;
        use datafusion_datasource::file_groups::FileGroup;

        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let source = Arc::new(ParquetSource::new(TableSchema::new(schema, vec![])));
        let file_groups = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|(path, size, range)| piece(path, *size, *range))
                    .collect::<FileGroup>()
            })
            .collect();
        let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
            .with_file_groups(file_groups)
            .build();
        DataSourceExec::from_data_source(config)
    }

    /// A pushed sort describes an arrangement of whole files: the same paths
    /// split into byte ranges, or regrouped, are a different arrangement.
    #[test]
    fn test_grouping_is_unchanged_compares_paths_and_ranges() {
        let whole = parquet_scan(&[&[("a", 100, None)], &[("b", 100, None)]]);
        let same = parquet_scan(&[&[("a", 100, None)], &[("b", 100, None)]]);
        let split = parquet_scan(&[
            &[("a", 100, Some((0, 50))), ("a", 100, Some((50, 100)))],
            &[("b", 100, None)],
        ]);
        let regrouped = parquet_scan(&[&[("a", 100, None), ("b", 100, None)]]);

        assert!(grouping_is_unchanged(&whole, &same));
        assert!(!grouping_is_unchanged(&whole, &split));
        assert!(!grouping_is_unchanged(&whole, &regrouped));
    }

    /// Wrappers that pass partitions through untouched do not hide the
    /// grouping. Any unknown wrappers are assumed to possibly change the grouping.
    #[test]
    fn test_grouping_is_unchanged_looks_through_pass_through_wrappers() {
        let scan = parquet_scan(&[&[("a", 100, None)], &[("b", 100, None)]]);
        let wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(Arc::clone(&scan)));
        let repartitioned: Arc<dyn ExecutionPlan> = Arc::new(
            RepartitionExec::try_new(Arc::clone(&scan), Partitioning::RoundRobinBatch(2)).unwrap(),
        );

        assert!(grouping_is_unchanged(&scan, &wrapped));
        assert!(!grouping_is_unchanged(&scan, &repartitioned));
    }

    fn selection_vectors_f1_f2() -> Arc<DashMap<String, Vec<bool>>> {
        let selection_vectors: Arc<DashMap<String, Vec<bool>>> = Arc::new(DashMap::new());
        selection_vectors.insert("f1".to_string(), vec![true, false]);
        selection_vectors.insert("f2".to_string(), vec![false, true]);
        selection_vectors
    }

    fn value_and_file_id_batch(
        values: &[i32],
        file_ids: &[Option<&str>],
        file_id_nullable: bool,
    ) -> TestResult<RecordBatch> {
        use arrow::datatypes::{Field, Schema};
        use arrow_array::{DictionaryArray, Int32Array};

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new(
                FILE_ID_COLUMN_DEFAULT,
                DataType::Dictionary(DataType::UInt16.into(), DataType::Utf8.into()),
                file_id_nullable,
            ),
        ]));

        let mut file_id_builder =
            arrow_array::builder::StringDictionaryBuilder::<UInt16Type>::new();
        for file_id in file_ids {
            match file_id {
                Some(file_id) => file_id_builder.append_value(file_id),
                None => file_id_builder.append_null(),
            }
        }
        let file_id: DictionaryArray<UInt16Type> = file_id_builder.finish();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(values.to_vec())),
                Arc::new(file_id),
            ],
        )?;

        Ok(batch)
    }

    fn test_scan_stream(
        scan_plan: Arc<KernelScanPlan>,
        kernel_type: KernelDataType,
        selection_vectors: Arc<DashMap<String, Vec<bool>>>,
        input_batches: Vec<RecordBatch>,
        file_id_column: Option<String>,
    ) -> DeltaScanStream {
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;

        let input_schema = input_batches
            .first()
            .map(|b| b.schema())
            .unwrap_or_else(|| Arc::clone(&scan_plan.contract.output_schema));
        let input_file_id_column = scan_plan.contract.file_id_field.name().clone();

        let input = Box::pin(RecordBatchStreamAdapter::new(
            input_schema,
            futures::stream::iter(input_batches.into_iter().map(Ok)),
        ));

        let schema_adapter =
            super::super::SchemaAdapter::new(Arc::clone(&scan_plan.contract.result_schema));
        let row_index_field = scan_plan.contract.retained_row_index_field();
        DeltaScanStream {
            scan_plan,
            kernel_type,
            input,
            baseline_metrics: BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
            transforms: Arc::new(HashMap::new()),
            selection_vectors,
            input_file_id_column,
            file_id_column,
            row_index_field,
            row_index_by_file: HashMap::new(),
            pending: VecDeque::new(),
            schema_adapter,
        }
    }

    fn retain_row_index(scan_plan: Arc<KernelScanPlan>, column: &str) -> Arc<KernelScanPlan> {
        let mut scan_plan = scan_plan.as_ref().clone();
        let row_index_field = Arc::new(Field::new(column, DataType::UInt64, false));
        let mut fields = scan_plan.contract.result_schema.fields().to_vec();
        // Keep the same row index field order as ProjectedScanContract::try_new.
        fields.push(row_index_field.clone());
        scan_plan.contract.output_schema = Arc::new(Schema::new(fields));
        scan_plan.contract.row_index_field = Some(row_index_field);
        scan_plan.contract.retain_row_index = true;
        Arc::new(scan_plan)
    }

    fn row_ordinals(batch: &RecordBatch, column: &str) -> Vec<u64> {
        batch
            .column_by_name(column)
            .expect("row ordinal column")
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec()
    }

    #[tokio::test]
    async fn test_required_input_distribution_tracks_retained_row_index_contract() -> TestResult {
        let (_kernel_type, scan_plan) = int32_scan_plan().await?;
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");

        let distribution = exec.required_input_distribution();
        assert!(
            matches!(
                distribution.as_slice(),
                [Distribution::UnspecifiedDistribution]
            ),
            "unexpected distribution: {distribution:?}"
        );

        let retained_exec = DeltaScanExec::new(
            retain_row_index(scan_plan, "row_ordinal"),
            Arc::clone(&exec.input),
            Arc::clone(&exec.transforms),
            Arc::clone(&exec.selection_vectors),
            exec.partition_stats.clone(),
            exec.metrics.clone(),
        );

        let distribution = retained_exec.required_input_distribution();
        assert!(
            matches!(distribution.as_slice(), [Distribution::SinglePartition]),
            "unexpected distribution: {distribution:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_retained_row_index_execute_rejects_multi_partition_child() -> TestResult {
        let (_kernel_type, scan_plan) = int32_scan_plan().await?;
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().await?;
        let session = Arc::new(create_session().into_inner());
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");

        let repartitioned_input = Arc::new(RepartitionExec::try_new(
            Arc::clone(&exec.input),
            Partitioning::RoundRobinBatch(2),
        )?);

        let retained_exec = DeltaScanExec::new(
            retain_row_index(scan_plan, "row_ordinal"),
            repartitioned_input,
            Arc::clone(&exec.transforms),
            Arc::clone(&exec.selection_vectors),
            exec.partition_stats.clone(),
            exec.metrics.clone(),
        );

        let err = match retained_exec.execute(0, session.task_ctx()) {
            Ok(_) => panic!("retained row-index scans must reject multi-partition children"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("retained row indexes require a single input partition"),
            "unexpected error: {err}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_project_appends_row_ordinals_from_scan_contract() -> TestResult {
        let (kernel_type, scan_plan) = int32_scan_plan().await?;
        let scan_plan = retain_row_index(scan_plan, "row_ordinal");
        let batch = value_and_file_id_batch(&[10, 11], &[Some("f1"), Some("f1")], false)?;

        let mut stream = test_scan_stream(
            scan_plan,
            kernel_type,
            Arc::new(DashMap::new()),
            Vec::new(),
            None,
        );

        assert!(stream.schema().column_with_name("row_ordinal").is_some());
        let outputs = stream.batch_project(batch)?;
        assert_eq!(outputs.len(), 1);
        assert_eq!(row_ordinals(&outputs[0], "row_ordinal"), vec![1, 2]);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_project_resets_row_ordinals_per_file() -> TestResult {
        let (kernel_type, scan_plan) = int32_scan_plan().await?;
        let scan_plan = retain_row_index(scan_plan, "row_ordinal");
        let batch = value_and_file_id_batch(
            &[10, 11, 20, 21],
            &[Some("f1"), Some("f1"), Some("f2"), Some("f2")],
            false,
        )?;

        let mut stream = test_scan_stream(
            scan_plan,
            kernel_type,
            Arc::new(DashMap::new()),
            Vec::new(),
            None,
        );

        let outputs = stream.batch_project(batch)?;
        assert_eq!(outputs.len(), 2);
        assert_eq!(row_ordinals(&outputs[0], "row_ordinal"), vec![1, 2]);
        assert_eq!(row_ordinals(&outputs[1], "row_ordinal"), vec![1, 2]);

        Ok(())
    }

    #[tokio::test]
    async fn test_poll_next_continues_row_ordinals_across_batches_for_same_file() -> TestResult {
        use futures::StreamExt;

        let (kernel_type, scan_plan) = int32_scan_plan().await?;
        let scan_plan = retain_row_index(scan_plan, "row_ordinal");
        let first = value_and_file_id_batch(&[10, 11], &[Some("f1"), Some("f1")], false)?;
        let second = value_and_file_id_batch(&[12, 13], &[Some("f1"), Some("f1")], false)?;

        let mut stream = test_scan_stream(
            scan_plan,
            kernel_type,
            Arc::new(DashMap::new()),
            vec![first, second],
            None,
        );

        let batch1 = stream.next().await.transpose()?.expect("first batch");
        let batch2 = stream.next().await.transpose()?.expect("second batch");
        assert!(stream.next().await.is_none());
        assert_eq!(row_ordinals(&batch1, "row_ordinal"), vec![1, 2]);
        assert_eq!(row_ordinals(&batch2, "row_ordinal"), vec![3, 4]);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_project_splits_mixed_file_batches_for_dv_masks() -> TestResult {
        let (kernel_type, scan_plan) = dv_kernel_type_and_int32_scan_plan().await?;
        let selection_vectors = selection_vectors_f1_f2();

        let batch = value_and_file_id_batch(
            &[10, 11, 20, 21],
            &[Some("f1"), Some("f1"), Some("f2"), Some("f2")],
            false,
        )?;

        let file_id_idx = file_id_column_idx(&batch, FILE_ID_COLUMN_DEFAULT)?;
        let runs = split_by_file_id_runs(&batch, file_id_idx)?;
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].0, "f1");
        assert_eq!(runs[1].0, "f2");
        assert!(selection_vectors.contains_key("f1"));
        assert!(selection_vectors.contains_key("f2"));

        let mut stream = test_scan_stream(
            Arc::clone(&scan_plan),
            kernel_type,
            selection_vectors,
            Vec::new(),
            None,
        );

        let outputs = stream.batch_project(batch)?;
        assert_eq!(outputs.len(), 2);

        let out1 = outputs[0]
            .column(0)
            .as_primitive::<arrow::datatypes::Int32Type>();
        let out2 = outputs[1]
            .column(0)
            .as_primitive::<arrow::datatypes::Int32Type>();
        assert_eq!(out1.values(), &[10]);
        assert_eq!(out2.values(), &[21]);

        Ok(())
    }

    #[tokio::test]
    async fn test_poll_next_buffers_fanout_batches() -> TestResult {
        use futures::StreamExt;

        let (kernel_type, scan_plan) = dv_kernel_type_and_int32_scan_plan().await?;
        let selection_vectors = selection_vectors_f1_f2();

        let batch = value_and_file_id_batch(
            &[10, 11, 20, 21],
            &[Some("f1"), Some("f1"), Some("f2"), Some("f2")],
            false,
        )?;

        let mut stream =
            test_scan_stream(scan_plan, kernel_type, selection_vectors, vec![batch], None);

        let batch1 = stream.next().await.transpose()?.expect("first batch");
        let batch2 = stream.next().await.transpose()?.expect("second batch");
        assert!(stream.next().await.is_none());

        let out1 = batch1
            .column(0)
            .as_primitive::<arrow::datatypes::Int32Type>();
        let out2 = batch2
            .column(0)
            .as_primitive::<arrow::datatypes::Int32Type>();
        assert_eq!(out1.values(), &[10]);
        assert_eq!(out2.values(), &[21]);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_project_handles_interleaved_file_ids() -> TestResult {
        let (kernel_type, scan_plan) = dv_kernel_type_and_int32_scan_plan().await?;
        let selection_vectors = selection_vectors_f1_f2();

        let batch = value_and_file_id_batch(
            &[10, 20, 11, 21],
            &[Some("f1"), Some("f2"), Some("f1"), Some("f2")],
            false,
        )?;

        let mut stream = test_scan_stream(
            Arc::clone(&scan_plan),
            kernel_type,
            selection_vectors,
            Vec::new(),
            None,
        );

        let outputs = stream.batch_project(batch)?;
        let kept: Vec<i32> = outputs
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<arrow::datatypes::Int32Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();

        assert_eq!(kept, vec![10, 21]);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_project_empty_batch_uses_contract_output_schema_without_metadata()
    -> TestResult {
        let (kernel_type, scan_plan) = int32_scan_plan().await?;
        let mut stream = test_scan_stream(
            Arc::clone(&scan_plan),
            kernel_type,
            Arc::new(DashMap::new()),
            Vec::new(),
            None,
        );
        let batches = stream.batch_project(RecordBatch::new_empty(Arc::clone(
            &scan_plan.contract.output_schema,
        )))?;
        let columns = batches[0].columns();

        assert_eq!(batches[0].schema(), scan_plan.contract.output_schema);
        assert_eq!(columns.len(), 1);
        assert_eq!(columns[0].data_type(), &DataType::Int32);
        assert!(columns[0].is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_project_empty_batch_uses_contract_output_schema_with_row_index()
    -> TestResult {
        let (kernel_type, scan_plan) = int32_scan_plan().await?;
        let scan_plan = retain_row_index(scan_plan, "row_ordinal");
        let mut stream = test_scan_stream(
            Arc::clone(&scan_plan),
            kernel_type,
            Arc::new(DashMap::new()),
            Vec::new(),
            None,
        );

        let batches = stream.batch_project(RecordBatch::new_empty(Arc::clone(
            &scan_plan.contract.output_schema,
        )))?;

        assert_eq!(batches[0].schema(), scan_plan.contract.output_schema);
        let schema = batches[0].schema();
        let row_ordinal = schema
            .column_with_name("row_ordinal")
            .expect("row ordinal column")
            .1;
        assert_eq!(row_ordinal.data_type(), &DataType::UInt64);
        assert_eq!(batches[0].num_rows(), 0);
        Ok(())
    }

    #[test]
    fn test_split_by_file_id_runs_invalid_type_returns_error() -> TestResult {
        use arrow::datatypes::{Field, Schema};
        use arrow_array::Int32Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new(FILE_ID_COLUMN_DEFAULT, DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![10, 20])),
            ],
        )?;

        let file_id_idx = file_id_column_idx(&batch, FILE_ID_COLUMN_DEFAULT)?;
        let err = split_by_file_id_runs(&batch, file_id_idx).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("Dictionary<UInt16"));
        assert!(message.contains("Int32"));

        Ok(())
    }

    #[test]
    fn test_split_by_file_id_runs_null_file_id_returns_error() -> TestResult {
        let batch = value_and_file_id_batch(&[1, 2], &[Some("f1"), None], true)?;

        let file_id_idx = file_id_column_idx(&batch, FILE_ID_COLUMN_DEFAULT)?;
        let err = split_by_file_id_runs(&batch, file_id_idx).unwrap_err();
        assert!(err.to_string().contains("file id value must not be null"));

        Ok(())
    }

    #[test]
    fn test_split_by_file_id_runs_preserves_dictionary_key_mapping() -> TestResult {
        use arrow::datatypes::{Field, Schema};
        use arrow_array::{DictionaryArray, Int32Array, StringArray, UInt16Array};

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new(
                FILE_ID_COLUMN_DEFAULT,
                DataType::Dictionary(DataType::UInt16.into(), DataType::Utf8.into()),
                false,
            ),
        ]));

        // Dictionary keys intentionally start with key=1 to ensure run labels are taken from keys,
        // not from row indexes.
        let keys = UInt16Array::from(vec![Some(1), Some(1), Some(0), Some(0), Some(1)]);
        let values = StringArray::from(vec!["f0", "f1"]);
        let file_ids = DictionaryArray::new(keys, Arc::new(values));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![10, 11, 20, 21, 12])),
                Arc::new(file_ids),
            ],
        )?;

        let file_id_idx = file_id_column_idx(&batch, FILE_ID_COLUMN_DEFAULT)?;
        let runs = split_by_file_id_runs(&batch, file_id_idx)?;
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].0, "f1");
        assert_eq!(runs[1].0, "f0");
        assert_eq!(runs[2].0, "f1");

        let run0 = runs[0]
            .1
            .column(0)
            .as_primitive::<arrow::datatypes::Int32Type>();
        let run1 = runs[1]
            .1
            .column(0)
            .as_primitive::<arrow::datatypes::Int32Type>();
        let run2 = runs[2]
            .1
            .column(0)
            .as_primitive::<arrow::datatypes::Int32Type>();
        assert_eq!(run0.values(), &[10, 11]);
        assert_eq!(run1.values(), &[20, 21]);
        assert_eq!(run2.values(), &[12]);

        Ok(())
    }

    #[tokio::test]
    async fn test_poll_next_fanout_preserves_file_ids() -> TestResult {
        use futures::StreamExt;

        let (kernel_type, scan_plan) = dv_kernel_type_and_int32_scan_plan().await?;
        let selection_vectors = selection_vectors_f1_f2();

        let batch = value_and_file_id_batch(
            &[10, 11, 20, 21],
            &[Some("f1"), Some("f1"), Some("f2"), Some("f2")],
            false,
        )?;

        let mut stream = test_scan_stream(
            scan_plan,
            kernel_type,
            selection_vectors,
            vec![batch],
            Some(FILE_ID_COLUMN_DEFAULT.to_string()),
        );

        let batch1 = stream.next().await.transpose()?.expect("first batch");
        let batch2 = stream.next().await.transpose()?.expect("second batch");
        assert!(stream.next().await.is_none());

        assert_eq!(batch1.num_columns(), 2);
        assert_eq!(batch2.num_columns(), 2);

        let file_id1 = batch1
            .column(1)
            .as_dictionary::<UInt16Type>()
            .downcast_dict::<StringArray>()
            .unwrap()
            .value(0)
            .to_string();
        let file_id2 = batch2
            .column(1)
            .as_dictionary::<UInt16Type>()
            .downcast_dict::<StringArray>()
            .unwrap()
            .value(0)
            .to_string();

        assert_eq!(file_id1, "f1");
        assert_eq!(file_id2, "f2");

        Ok(())
    }

    #[test]
    fn test_dv_short_mask_drain_and_pad() {
        use super::{DvMaskResult, consume_dv_mask};

        let mut sv = vec![true, false, true];
        let result = consume_dv_mask(&mut sv, 5);

        assert_eq!(
            result,
            DvMaskResult {
                selection: Some(vec![true, false, true, true, true]),
                should_remove: true,
            }
        );
        assert!(sv.is_empty());
    }

    #[test]
    fn test_dv_mask_exhaustion_across_batches() {
        use super::{DvMaskResult, consume_dv_mask};
        use dashmap::DashMap;

        let selection_vectors: DashMap<String, Vec<bool>> = DashMap::new();
        let file_id = "test_file.parquet".to_string();
        selection_vectors.insert(file_id.clone(), vec![false, true]);

        let result1 = {
            let mut sv = selection_vectors.get_mut(&file_id).unwrap();
            consume_dv_mask(&mut sv, 5)
        };
        assert_eq!(
            result1,
            DvMaskResult {
                selection: Some(vec![false, true, true, true, true]),
                should_remove: true,
            }
        );
        if result1.should_remove {
            selection_vectors.remove(&file_id);
        }

        let result2 = if let Some(mut sv) = selection_vectors.get_mut(&file_id) {
            consume_dv_mask(&mut sv, 5)
        } else {
            DvMaskResult {
                selection: None,
                should_remove: false,
            }
        };
        assert_eq!(
            result2,
            DvMaskResult {
                selection: None,
                should_remove: false,
            }
        );
    }

    #[test]
    fn test_dv_normal_mask_drains_exactly() {
        use super::{DvMaskResult, consume_dv_mask};

        let mut sv = vec![
            true, false, true, false, true, true, false, true, false, true,
        ];

        let result1 = consume_dv_mask(&mut sv, 3);
        assert_eq!(
            result1,
            DvMaskResult {
                selection: Some(vec![true, false, true]),
                should_remove: false,
            }
        );
        assert_eq!(sv.len(), 7);

        let result2 = consume_dv_mask(&mut sv, 3);
        assert_eq!(
            result2,
            DvMaskResult {
                selection: Some(vec![false, true, true]),
                should_remove: false,
            }
        );
        assert_eq!(sv, vec![false, true, false, true]);

        let result3 = consume_dv_mask(&mut sv, 5);
        assert_eq!(
            result3,
            DvMaskResult {
                selection: Some(vec![false, true, false, true, true]),
                should_remove: true,
            }
        );
        assert!(sv.is_empty());

        let result4 = consume_dv_mask(&mut sv, 5);
        assert_eq!(
            result4,
            DvMaskResult {
                selection: None,
                should_remove: true,
            }
        );
    }

    #[test]
    fn test_dv_long_mask_retains_remainder_for_next_batch() {
        use super::{DvMaskResult, consume_dv_mask};

        let mut sv = vec![true, false, false, true];
        let result = consume_dv_mask(&mut sv, 2);

        assert_eq!(
            result,
            DvMaskResult {
                selection: Some(vec![true, false]),
                should_remove: false,
            }
        );
        assert_eq!(sv, vec![false, true]);
    }
}
