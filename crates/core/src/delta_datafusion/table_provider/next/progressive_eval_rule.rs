//! Physical optimizer rule that replaces a merge of ordered partitions with
//! a plain concatenation.
//!
//! A [`SortPreservingMergeExec`] combines its input partitions into one globally
//! ordered stream. That merge compares rows across all partitions and executes every
//! partition up front.
//! If the partitions are non-overlapping with respect to the sort order, then
//! this merge is unnecessary, and we can stream the partitions one after another
//! using a [`ProgressiveEvalExec`].
//!
//! Streaming partitions one after another is only sound when no operator below
//! the merge needs all of its partitions to be running at once; see
//! [`contains_hash_join`] for the one that does.
//!
//! Because this rule is installed session-wide on
//! [`DeltaSessionContext`](crate::delta_datafusion::DeltaSessionContext), it is
//! scoped to merges over [`DeltaScanExec`] subplans (see
//! [`is_delta_scan_plan`]) so that non-Delta tables registered in the same
//! session are left untouched. It also skips merges over blocking operators
//! such as a per-partition `SortExec` (see [`contains_blocking_operator`]), so
//! those operators keep full multi-core parallelism.

use datafusion::common::config::ConfigOptions;
use datafusion::common::stats::Statistics;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{Result, ScalarValue};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, Partitioning, PhysicalSortExpr,
};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::statistics::{StatisticsArgs, StatisticsContext};
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties as _};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;

use super::progressive_eval::ProgressiveEvalExec;
use super::scan::DeltaScanExec;
use std::cmp::Ordering;
use std::sync::Arc;

/// Replace [`SortPreservingMergeExec`] with [`ProgressiveEvalExec`] when its
/// input is a Delta scan whose partitions are mutually non-overlapping,
/// range-ordered slices of the merge ordering, so concatenating them already
/// yields globally ordered output. See the [module docs](self) for background.
#[derive(Debug, Default)]
pub struct ProgressiveEvalRule;

impl ProgressiveEvalRule {
    /// Create the rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for ProgressiveEvalRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        optimize_plan(plan, true)
    }

    fn name(&self) -> &str {
        "ProgressiveEvalRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Core rewrite for [`ProgressiveEvalRule`]. `require_delta_scan` is always
/// `true` in production; unit tests pass `false` to exercise the statistics
/// and operator guards over synthetic plan nodes without a full Delta table.
fn optimize_plan(
    plan: Arc<dyn ExecutionPlan>,
    require_delta_scan: bool,
) -> Result<Arc<dyn ExecutionPlan>> {
    plan.transform_down(|plan| {
        let Some(merge) =
            (plan.as_ref() as &dyn ExecutionPlan).downcast_ref::<SortPreservingMergeExec>()
        else {
            return Ok(Transformed::no(plan));
        };
        let input = merge.input();
        if require_delta_scan && !is_delta_scan_plan(input) {
            return Ok(Transformed::no(plan));
        }
        if contains_hash_join(input)? {
            return Ok(Transformed::no(plan));
        }
        if contains_blocking_operator(input)? {
            return Ok(Transformed::no(plan));
        }
        // A Delta scan may declare the partitions disjoint outright - from
        // a sort pushdown that built them that way, or from the table's
        // assertion that its files never overlap. That is asked first: it
        // is a walk over the plan, whereas the proof materialises every
        // partition's statistics, which the declaration makes redundant.
        // The ranges are then only descriptive, so it does not matter that
        // they may be unavailable.
        let ranges = if partitions_declared_disjoint(input, merge.expr())? {
            leading_partition_ranges(input, merge.expr())
        } else {
            let Some(ranges) = ordered_partition_ranges(input, merge.expr()) else {
                return Ok(Transformed::no(plan));
            };
            Some(ranges)
        };
        let input = preserve_input_order(Arc::clone(input))?;
        let replacement =
            ProgressiveEvalExec::new(input, Some(merge.expr().clone()), ranges, merge.fetch());
        Ok(Transformed::yes(Arc::new(replacement) as _))
    })
    .data()
}

/// Whether every data source feeding `plan` is a [`DeltaScanExec`].
///
/// [`ProgressiveEvalRule`] is registered on every
/// [`DeltaSessionContext`](crate::delta_datafusion::DeltaSessionContext), so
/// scoping it to Delta scans keeps it from altering the execution of non-Delta
/// tables (such as Parquet listing tables) registered in the same session.
fn is_delta_scan_plan(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if (plan.as_ref() as &dyn ExecutionPlan)
        .downcast_ref::<DeltaScanExec>()
        .is_some()
    {
        return true;
    }
    let children = plan.children();
    !children.is_empty() && children.into_iter().all(is_delta_scan_plan)
}

/// Whether `plan` sits on a Delta scan that *declares* - rather than proves
/// from statistics - that its execution partitions are mutually disjoint and
/// arranged in range order, on an ordering that also covers `ordering`. See
/// [`DeltaScanExec::disjoint_ordering`] for where such a declaration comes
/// from.
///
/// The statistics proof beside this one reaches the scan through intervening
/// operators for free, because per-partition statistics propagate up through
/// them. A declaration is not a statistic, so it has to be fetched from the
/// scan itself, and this walks down to find it - rewriting `ordering` over
/// each child's schema on the way, since the columns it names are those of the
/// plan it started from.
///
/// The walk descends only through operators that hand their input's partitions
/// on as they are; see [`pass_through_ordering`]. Anything else ends it, at
/// the cost of a missed optimization rather than a wrong result.
fn partitions_declared_disjoint(
    plan: &Arc<dyn ExecutionPlan>,
    ordering: &LexOrdering,
) -> Result<bool> {
    let node = plan.as_ref() as &dyn ExecutionPlan;
    if let Some(scan) = node.downcast_ref::<DeltaScanExec>() {
        let Some(declared) = scan.disjoint_ordering() else {
            return Ok(false);
        };
        // Concatenating the partitions yields a stream ordered on `declared`.
        // That makes the merge redundant only if `declared` also delivers what
        // the merge was asked for - which it does when `ordering` is a prefix
        // of it, and through equivalences in cases a plain prefix check would
        // miss. Only `declared` is offered: the scan's other orderings
        // describe each partition on its own and say nothing about how they
        // sit relative to each other.
        return EquivalenceProperties::new_with_orderings(plan.schema(), vec![declared.clone()])
            .ordering_satisfy(ordering.iter().cloned());
    }
    let (Some(child_ordering), [child]) =
        (pass_through_ordering(node, ordering), &node.children()[..])
    else {
        return Ok(false);
    };
    partitions_declared_disjoint(child, &child_ordering)
}

/// Rewrite `ordering`, whose columns index `plan`'s output schema, over the
/// schema of its child, or `None` when `plan` is not an operator this can see
/// through.
///
/// Seeing through one means its partitions are its child's: each output
/// partition holds rows of the input partition with the same index, possibly
/// fewer of them, in the order they had there, so a claim about how the
/// child's partitions sit relative to each other still holds of its own. An
/// operator is taken to keep its partitions that way when it
///
/// - has a single child;
/// - reports that it maintains its input order;
/// - has as many partitions as its child and declares no distribution scheme
///   for them. The flag alone does not rule out an order-preserving
///   `RepartitionExec`, which keeps every output partition ordered while
///   dealing rows *between* partitions, and may keep their number. But a
///   repartition always declares the scheme it dealt by, whereas an operator
///   that forwards its input's partitions leaves them `UnknownPartitioning`
///   as the scan declared them; and the other exchanges, merges and
///   coalesces, end at a single partition;
/// - and can carry the sort columns down: unchanged when its schema is its
///   child's, or through its projection when it is a `ProjectionExec` or a
///   `FilterExec` in which `ProjectionPushdown` embedded one. Only a column
///   passed straight through can be carried - one the projection computes is
///   not the column the scan arranged its files on, whatever it is named -
///   and any other change to the schema ends the walk, which is what keeps
///   out the order-maintaining operators that build new rows, such as
///   aggregates and window functions.
///
/// Filters, projections, limits, batch coalescing, the buffer DataFusion's
/// sort pushdown puts under a merge, and cooperative yielding all pass.
/// Operators that repartition, sort, join or aggregate end the walk.
fn pass_through_ordering(plan: &dyn ExecutionPlan, ordering: &LexOrdering) -> Option<LexOrdering> {
    let [child] = plan.children()[..] else {
        return None;
    };
    if !plan
        .maintains_input_order()
        .first()
        .copied()
        .unwrap_or(false)
    {
        return None;
    }
    let partition_count = child.output_partitioning().partition_count();
    let forwards_partitions = matches!(
        plan.output_partitioning(),
        Partitioning::UnknownPartitioning(count) if *count == partition_count
    );
    if !forwards_partitions {
        return None;
    }
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        return map_ordering_columns(ordering, |column| {
            let projected = projection.expr().get(column.index())?;
            projected.expr.downcast_ref::<Column>().cloned()
        });
    }
    if let Some(projection) = plan
        .downcast_ref::<FilterExec>()
        .and_then(|filter| filter.projection().as_ref())
    {
        return map_ordering_columns(ordering, |column| {
            Some(Column::new(column.name(), *projection.get(column.index())?))
        });
    }
    (plan.schema() == child.schema()).then(|| ordering.clone())
}

/// Rewrite `ordering` with each of its columns replaced by the child column
/// `carry` reads it from, or `None` when any of them does not come from one.
fn map_ordering_columns(
    ordering: &LexOrdering,
    carry: impl Fn(&Column) -> Option<Column>,
) -> Option<LexOrdering> {
    let mapped = ordering
        .iter()
        .map(|sort_expr| {
            let column = sort_expr.expr.downcast_ref::<Column>()?;
            Some(PhysicalSortExpr::new(
                Arc::new(carry(column)?),
                sort_expr.options,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    LexOrdering::new(mapped)
}

/// The first sort column's `(start, end)` value per partition, with no ordering
/// check. Reported by [`ProgressiveEvalExec`] as `input_ranges` for readers of
/// the plan; `None` when the statistics do not yield them.
fn leading_partition_ranges(
    plan: &Arc<dyn ExecutionPlan>,
    ordering: &LexOrdering,
) -> Option<Vec<(ScalarValue, ScalarValue)>> {
    // Only the leading column is reported, so statistics missing on a later
    // one must not withhold it.
    let leading = LexOrdering::new([ordering.first().clone()])?;
    let stats_context = StatisticsContext::new();
    (0..plan.output_partitioning().partition_count())
        .map(|partition_idx| {
            let stats = stats_context
                .compute(
                    plan.as_ref(),
                    &StatisticsArgs::new().with_partition(Some(partition_idx)),
                )
                .ok()?;
            let (starts, ends, _) = get_ordering_stats(&stats, &leading)?;
            Some((starts[0].clone(), ends[0].clone()))
        })
        .collect()
}

/// Whether any node of `plan` is a [`HashJoinExec`].
///
/// A [`ProgressiveEvalExec`] executes its input partitions a few at a time,
/// starting the next one only once an earlier one is exhausted. A hash join
/// whose dynamic filter was pushed into the probe-side scan cannot be read
/// that way: every probe partition reports its build-side bounds to a shared
/// accumulator and then waits until *all* probe partitions have reported, so
/// the partitions that are running wait for ones that will only be started
/// after they finish, and the query hangs. The filter is attached by
/// `FilterPushdown::new_post_optimization`, which runs before this rule, so
/// it could be detected precisely; but joins over a scan that was regrouped
/// for a sort are uncommon enough that the merge is simply kept for every
/// hash join, which also does not depend on how the rules are ordered.
fn contains_hash_join(plan: &Arc<dyn ExecutionPlan>) -> Result<bool> {
    plan.exists(|node| {
        Ok((node.as_ref() as &dyn ExecutionPlan)
            .downcast_ref::<HashJoinExec>()
            .is_some())
    })
}

/// Whether any node of `plan` is a blocking operator (must consume all of its
/// input before emitting any output, such as a per-partition `SortExec`).
///
/// A [`ProgressiveEvalExec`] executes only `1 + prefetch` partitions at a time
/// (default 2). Rewriting a merge over a blocking operator serializes
/// per-partition work that [`SortPreservingMergeExec`] would otherwise run
/// across all partitions in parallel.
fn contains_blocking_operator(plan: &Arc<dyn ExecutionPlan>) -> Result<bool> {
    plan.exists(|node| Ok(node.properties().emission_type == EmissionType::Final))
}

/// Mark the data sources under `plan` order-sensitive, so that they keep the
/// partition-to-data mapping their per-partition statistics describe.
///
/// A file scan that declares no output ordering is free to let its sibling
/// streams steal unopened files from one another at execution time, which is
/// what `datafusion.execution.enable_file_stream_work_stealing` does by
/// default. The partition boundaries proven above are then meaningless: a
/// partition can emit rows planned for another. Pinning order-sensitivity here
/// ensures every file stays on its planned partition.
///
/// Only leaves are asked: the nodes above forward `with_preserve_order` to
/// their children and rebuild themselves, which the walk already does for the
/// ancestors of a changed leaf. A file scan that keeps its files local is
/// left as it is - `FileScanConfig::with_preserve_order` copies every file
/// even when the flag already matches, and a scan regrouped for a pushed-down
/// sort is already order-sensitive.
fn preserve_input_order(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    plan.transform_down(|plan| {
        if !plan.children().is_empty() || scan_keeps_files_local(&plan) {
            return Ok(Transformed::no(plan));
        }
        match plan.with_preserve_order(true) {
            Some(pinned) => Ok(Transformed::yes(pinned)),
            None => Ok(Transformed::no(plan)),
        }
    })
    .data()
}

/// Whether `plan` is a file scan whose streams cannot take each other's files:
/// order-sensitive, or declaring an output partitioning, which disables the
/// sharing as well.
fn scan_keeps_files_local(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<DataSourceExec>()
        .and_then(|scan| scan.data_source().as_ref().downcast_ref::<FileScanConfig>())
        .is_some_and(|config| config.preserve_order || config.output_partitioning.is_some())
}

/// To be able to convert to a ProgressiveEval, we need the partitions to
/// be ordered with respect to each other.
/// If this is the case, return a vector of the (start, end) range for the first
/// sort term from each partition, otherwise return None.
///
/// Each partition boundary is checked by walking the sort columns: a strictly
/// ordered column proves the boundary (later columns are irrelevant), an equal
/// column defers to the next one, and an out-of-order column rejects. The
/// min/max statistics used for these comparisons exclude nulls, so a
/// comparison at a column is only valid when nulls cannot hide on the wrong
/// side of the boundary: nulls sorting last may lurk anywhere in the earlier
/// partition (including among rows tied on all preceding sort columns, which
/// they would follow), and nulls sorting first anywhere in the later one.
/// Columns after a strict break are never relied on, so their nulls are
/// harmless.
///
/// NaN is excluded from the statistics in the same way, but unlike nulls
/// nothing reports how many there are, so there is no proving a float column
/// free of them. A boundary that would have to be decided on a float column
/// is rejected.
fn ordered_partition_ranges(
    plan: &Arc<dyn ExecutionPlan>,
    ordering: &LexOrdering,
) -> Option<Vec<(ScalarValue, ScalarValue)>> {
    let partition_count = plan.output_partitioning().partition_count();
    let mut prev_ends: Vec<ScalarValue> = Vec::new();
    let mut prev_null_counts: Vec<usize> = Vec::new();
    let mut first_col_ordering = Vec::with_capacity(partition_count);
    // One context for all partitions, so statistics of shared subplans are
    // computed once.
    let stats_context = StatisticsContext::new();
    for partition_idx in 0..partition_count {
        let partition_stats = stats_context
            .compute(
                plan.as_ref(),
                &StatisticsArgs::new().with_partition(Some(partition_idx)),
            )
            .ok()?;
        let (starts, ends, null_counts) = get_ordering_stats(&partition_stats, ordering)?;
        if partition_idx != 0 {
            // Check partition is ordered correctly with respect to the previous partition
            for (i, sort_expr) in ordering.iter().enumerate() {
                // Reject nulls that could sort onto the wrong side of this
                // boundary; see the function docs.
                let boundary_null_count = if sort_expr.options.nulls_first {
                    null_counts[i]
                } else {
                    prev_null_counts[i]
                };
                if boundary_null_count != 0 {
                    return None;
                }
                // Min/max statistics leave NaN out as well, and nothing counts
                // them, so a float column's bounds cannot place a boundary.
                if starts[i].data_type().is_floating() {
                    return None;
                }
                // Incomparable values (partial_cmp is None) bail out rather
                // than being silently treated as equal.
                let cmp = starts[i].partial_cmp(&prev_ends[i])?;
                let cmp = if sort_expr.options.descending {
                    cmp.reverse()
                } else {
                    cmp
                };
                match cmp {
                    // In order
                    Ordering::Greater => break,
                    // Out of order
                    Ordering::Less => return None,
                    // Equal, need to check next sort expression
                    Ordering::Equal => continue,
                }
            }
        }
        first_col_ordering.push((starts[0].clone(), ends[0].clone()));
        prev_ends = ends;
        prev_null_counts = null_counts;
    }
    Some(first_col_ordering)
}

fn get_ordering_stats(
    stats: &Arc<Statistics>,
    ordering: &LexOrdering,
) -> Option<(Vec<ScalarValue>, Vec<ScalarValue>, Vec<usize>)> {
    let mut starts = Vec::with_capacity(ordering.len());
    let mut ends = Vec::with_capacity(ordering.len());
    let mut null_counts = Vec::with_capacity(ordering.len());

    for sort_expr in ordering.iter() {
        let column = sort_expr.expr.downcast_ref::<Column>()?;
        let col_stats = stats.column_statistics.get(column.index())?;
        // We require exact stats to guarantee no overlap in partition ranges.
        if !(col_stats.null_count.is_exact()?
            && col_stats.min_value.is_exact()?
            && col_stats.max_value.is_exact()?)
        {
            return None;
        }
        let (start, end) = if sort_expr.options.descending {
            (
                col_stats.max_value.get_value()?,
                col_stats.min_value.get_value()?,
            )
        } else {
            (
                col_stats.min_value.get_value()?,
                col_stats.max_value.get_value()?,
            )
        };
        // Stats may be null for all-null or empty partitions.
        // For now, don't try to optimize this case:
        if start.is_null() || end.is_null() {
            return None;
        }
        starts.push(start.clone());
        ends.push(end.clone());
        null_counts.push(*col_stats.null_count.get_value()?);
    }

    Some((starts, ends, null_counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
    use datafusion::common::stats::{ColumnStatistics, Precision};
    use datafusion::common::tree_node::TreeNodeRecursion;
    use datafusion::common::{JoinType, NullEquality};
    use datafusion::execution::{SendableRecordBatchStream, TaskContext};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion::physical_plan::buffer::BufferExec;
    use datafusion::physical_plan::execution_plan::Boundedness;
    use datafusion::physical_plan::filter::FilterExecBuilder;
    use datafusion::physical_plan::joins::PartitionMode;
    use datafusion::physical_plan::limit::LocalLimitExec;
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::physical_plan::{DisplayAs, DisplayFormatType, PlanProperties};

    /// Plan reporting fixed per-partition statistics; never executed. It is a
    /// leaf unless given an `input`, which it reports regardless of.
    #[derive(Debug)]
    struct StatsExec {
        stats: Vec<Statistics>,
        input: Option<Arc<dyn ExecutionPlan>>,
        cache: Arc<PlanProperties>,
    }

    impl StatsExec {
        fn plan(stats: Vec<Statistics>) -> Arc<dyn ExecutionPlan> {
            Self::build(stats, None, None)
        }

        fn over(
            stats: Vec<Statistics>,
            input: Option<Arc<dyn ExecutionPlan>>,
        ) -> Arc<dyn ExecutionPlan> {
            Self::build(stats, input, None)
        }

        /// A leaf that also declares `ordering` as the order of each of its
        /// partitions.
        fn declaring(
            stats: Vec<Statistics>,
            ordering: Option<LexOrdering>,
        ) -> Arc<dyn ExecutionPlan> {
            Self::build(stats, None, ordering)
        }

        fn build(
            stats: Vec<Statistics>,
            input: Option<Arc<dyn ExecutionPlan>>,
            ordering: Option<LexOrdering>,
        ) -> Arc<dyn ExecutionPlan> {
            let cache = Arc::new(PlanProperties::new(
                EquivalenceProperties::new_with_orderings(test_schema(), ordering),
                Partitioning::UnknownPartitioning(stats.len()),
                EmissionType::Incremental,
                Boundedness::Bounded,
            ));
            Arc::new(Self {
                stats,
                input,
                cache,
            })
        }
    }

    impl DisplayAs for StatsExec {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            write!(f, "StatsExec")
        }
    }

    impl ExecutionPlan for StatsExec {
        fn name(&self) -> &'static str {
            "StatsExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.cache
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            self.input.iter().collect()
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            unimplemented!("StatsExec is only used for planning")
        }

        fn statistics_from_inputs(
            &self,
            _input_stats: &[Arc<Statistics>],
            args: &StatisticsArgs,
        ) -> Result<Arc<Statistics>> {
            match args.partition() {
                Some(idx) => Ok(Arc::new(self.stats[idx].clone())),
                // A single partition's statistics are also the whole plan's.
                None if self.stats.len() == 1 => Ok(Arc::new(self.stats[0].clone())),
                None => Ok(Arc::new(Statistics::new_unknown(&self.schema()))),
            }
        }
    }

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("t", DataType::Int64, true),
            Field::new("id", DataType::Int64, true),
        ]))
    }

    fn sort_expr(index: usize, name: &str, options: SortOptions) -> PhysicalSortExpr {
        PhysicalSortExpr::new(Arc::new(Column::new(name, index)), options)
    }

    fn asc(index: usize, name: &str) -> PhysicalSortExpr {
        sort_expr(
            index,
            name,
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )
    }

    fn desc(index: usize, name: &str) -> PhysicalSortExpr {
        sort_expr(
            index,
            name,
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        )
    }

    fn exact_i64(min: i64, max: i64, null_count: usize) -> ColumnStatistics {
        ColumnStatistics {
            null_count: Precision::Exact(null_count),
            min_value: Precision::Exact(ScalarValue::Int64(Some(min))),
            max_value: Precision::Exact(ScalarValue::Int64(Some(max))),
            ..Default::default()
        }
    }

    fn partition(column_statistics: Vec<ColumnStatistics>) -> Statistics {
        Statistics {
            num_rows: Precision::Exact(10),
            total_byte_size: Precision::Exact(100),
            column_statistics,
        }
    }

    #[test]
    fn equal_first_column_boundary_ordered_by_second_column() {
        // The partitions share t = 100 on the boundary; the disjoint id
        // ranges disambiguate, so the check falls through to the second
        // sort column and accepts.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 100, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(100, 200, 0), exact_i64(11, 20, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        let ranges = ordered_partition_ranges(&plan, &ordering).expect("expected ranges");
        assert_eq!(
            ranges,
            vec![
                (ScalarValue::Int64(Some(0)), ScalarValue::Int64(Some(100))),
                (ScalarValue::Int64(Some(100)), ScalarValue::Int64(Some(200))),
            ]
        );
    }

    #[test]
    fn equal_first_column_boundary_overlapping_second_column() {
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 100, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(100, 200, 0), exact_i64(5, 20, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn boundary_equal_on_all_sort_columns_is_accepted() {
        // min == prev max on every sort column bounds all rows of both
        // partitions, so concatenation is still (non-strictly) ordered.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 100, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(100, 200, 0), exact_i64(10, 20, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_some());
    }

    #[test]
    fn nulls_only_allowed_in_last_partition_for_nulls_last() {
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();

        // Nulls in the last partition sort after all values: accepted.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 99, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(100, 200, 2), exact_i64(0, 10, 0)]),
        ]);
        assert!(ordered_partition_ranges(&plan, &ordering).is_some());

        // Nulls in the first partition would surface mid-stream: rejected.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 99, 2), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(100, 200, 0), exact_i64(0, 10, 0)]),
        ]);
        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn nulls_only_allowed_in_first_partition_for_nulls_first() {
        let ordering = LexOrdering::new(vec![desc(0, "t")]).unwrap();

        // Descending partitions with nulls leading in the first: accepted.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(100, 200, 2), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(0, 99, 0), exact_i64(0, 10, 0)]),
        ]);
        assert!(ordered_partition_ranges(&plan, &ordering).is_some());

        // Nulls in the last partition sort before its values: rejected.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(100, 200, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(0, 99, 2), exact_i64(0, 10, 0)]),
        ]);
        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn nulls_in_deeper_sort_column_hidden_by_first_column_break() {
        // The middle partition carries a null in the second sort column. Its
        // boundary with the first partition is strict on the first column, so
        // the second column is never inspected for it. The boundary with the
        // last partition is equal on the first column and falls through to
        // the second: the middle partition's non-null max (4) < the last
        // partition's min (5) looks ordered, but the middle partition's null
        // rows sort after every non-null value (nulls last), so a row like
        // (20, NULL) would precede (20, 5) in the concatenation.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 9, 0), exact_i64(0, 9, 0)]),
            partition(vec![exact_i64(10, 20, 0), exact_i64(0, 4, 1)]),
            partition(vec![exact_i64(20, 30, 0), exact_i64(5, 8, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn nulls_in_deeper_sort_column_harmless_when_boundaries_strict_on_first() {
        // Both of the middle partition's boundaries are strict on the first
        // column, so the second sort column is never relied on and its nulls
        // cannot surface out of order.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 9, 0), exact_i64(0, 9, 0)]),
            partition(vec![exact_i64(10, 19, 0), exact_i64(0, 4, 1)]),
            partition(vec![exact_i64(20, 30, 0), exact_i64(5, 8, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_some());
    }

    #[test]
    fn nulls_in_first_partition_with_all_equal_values_rejected() {
        // Every partition shares the same value on both sort columns, so no
        // boundary breaks early and every column is inspected. The first
        // partition's nulls (sorting last) would surface before the later
        // partitions' rows; the first boundary must reject them.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 1)]),
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 0)]),
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn nulls_in_middle_partition_with_all_equal_values_rejected() {
        // As above, but the nulls sit in the middle partition: its boundary
        // with the *next* partition is the one that must reject them.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 0)]),
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 1)]),
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn nulls_in_last_partition_with_all_equal_values_accepted() {
        // Nulls sorting last in the last partition stream at the very end of
        // the concatenation: correct, and there is no later boundary to
        // invalidate.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 0)]),
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 0)]),
            partition(vec![exact_i64(5, 5, 0), exact_i64(7, 7, 1)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_some());
    }

    #[test]
    fn descending_partitions_out_of_order() {
        // Ascending partition layout under a descending ordering.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 99, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(100, 200, 0), exact_i64(0, 10, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![desc(0, "t")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn incomparable_statistics_types_bail_out() {
        // Mismatched stat types across partitions are incomparable; they
        // must not be treated as an equal boundary.
        let utf8 = |value: &str| ScalarValue::Utf8(Some(value.to_string()));
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 100, 0), exact_i64(0, 10, 0)]),
            partition(vec![
                ColumnStatistics {
                    null_count: Precision::Exact(0),
                    min_value: Precision::Exact(utf8("a")),
                    max_value: Precision::Exact(utf8("b")),
                    ..Default::default()
                },
                exact_i64(0, 10, 0),
            ]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn inexact_statistics_bail_out() {
        let inexact = ColumnStatistics {
            null_count: Precision::Exact(0),
            min_value: Precision::Inexact(ScalarValue::Int64(Some(100))),
            max_value: Precision::Exact(ScalarValue::Int64(Some(200))),
            ..Default::default()
        };
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 99, 0), exact_i64(0, 10, 0)]),
            partition(vec![inexact, exact_i64(0, 10, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn null_statistics_values_bail_out() {
        // All-null or empty partitions report exact but null min/max values;
        // they prove nothing about the partition's range. An all-null first
        // partition under an ascending nulls-first ordering is the dangerous
        // layout: the null guard passes (the later partition has no nulls)
        // and a null scalar compares before any value, so without the
        // explicit bail-out the boundary would look ordered.
        let all_null = ColumnStatistics {
            null_count: Precision::Exact(10),
            min_value: Precision::Exact(ScalarValue::Int64(None)),
            max_value: Precision::Exact(ScalarValue::Int64(None)),
            ..Default::default()
        };
        let plan = StatsExec::plan(vec![
            partition(vec![all_null, exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(0, 99, 0), exact_i64(0, 10, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![sort_expr(
            0,
            "t",
            SortOptions {
                descending: false,
                nulls_first: true,
            },
        )])
        .unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn missing_column_statistics_bail_out() {
        // The ordering references a column index beyond the available
        // statistics; the lookup must bail out rather than panic.
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 99, 0)]),
            partition(vec![exact_i64(100, 200, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn leading_ranges_ignore_statistics_on_later_sort_columns() {
        // The second sort column's statistics are inexact in one partition
        // and missing from the other, neither of which touches the leading
        // column the ranges report.
        let inexact = ColumnStatistics {
            null_count: Precision::Exact(0),
            min_value: Precision::Inexact(ScalarValue::Int64(Some(0))),
            max_value: Precision::Exact(ScalarValue::Int64(Some(10))),
            ..Default::default()
        };
        let plan = StatsExec::plan(vec![
            partition(vec![exact_i64(0, 99, 0), inexact]),
            partition(vec![exact_i64(100, 200, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();

        assert_eq!(
            leading_partition_ranges(&plan, &ordering),
            Some(vec![
                (ScalarValue::Int64(Some(0)), ScalarValue::Int64(Some(99))),
                (ScalarValue::Int64(Some(100)), ScalarValue::Int64(Some(200))),
            ])
        );
    }

    /// Two partitions with disjoint `t` ranges: a merge on `t` over them is
    /// replaceable by a progressive eval.
    fn ordered_partitions() -> Arc<dyn ExecutionPlan> {
        StatsExec::plan(vec![
            partition(vec![exact_i64(0, 100, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(101, 200, 0), exact_i64(11, 20, 0)]),
        ])
    }

    fn merge_on_t(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();
        Arc::new(SortPreservingMergeExec::new(ordering, input))
    }

    fn optimize(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        ProgressiveEvalRule::new()
            .optimize(plan, &ConfigOptions::default())
            .unwrap()
    }

    fn optimize_any(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        optimize_plan(plan, false).unwrap()
    }

    #[test]
    fn merge_over_non_delta_scan_is_kept() {
        let plan = ordered_partitions();
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();
        assert!(
            ordered_partition_ranges(&plan, &ordering).is_some(),
            "the partitions should look replaceable on statistics alone"
        );

        let optimized = optimize(merge_on_t(plan));

        assert!(
            optimized
                .downcast_ref::<SortPreservingMergeExec>()
                .is_some(),
            "expected SortPreservingMergeExec for non-Delta input, got {}",
            optimized.name()
        );
    }

    #[test]
    fn merge_over_ordered_partitions_is_replaced() {
        let optimized = optimize_any(merge_on_t(ordered_partitions()));

        assert!(
            optimized.downcast_ref::<ProgressiveEvalExec>().is_some(),
            "expected ProgressiveEvalExec, got {}",
            optimized.name()
        );
    }

    #[test]
    fn merge_over_sort_exec_is_kept() {
        use datafusion::physical_plan::sorts::sort::SortExec;

        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();
        let sort: Arc<dyn ExecutionPlan> = Arc::new(
            SortExec::new(ordering.clone(), ordered_partitions()).with_preserve_partitioning(true),
        );
        assert!(
            ordered_partition_ranges(&sort, &ordering).is_some(),
            "the per-partition sort should look replaceable on statistics alone"
        );

        let optimized = optimize_any(merge_on_t(Arc::clone(&sort)));
        assert!(
            optimized
                .downcast_ref::<SortPreservingMergeExec>()
                .is_some(),
            "expected SortPreservingMergeExec over unbounded SortExec, got {}",
            optimized.name()
        );

        let limited_merge = merge_on_t(sort).with_fetch(Some(5)).unwrap();
        let optimized = optimize_any(limited_merge);
        assert!(
            optimized
                .downcast_ref::<SortPreservingMergeExec>()
                .is_some(),
            "expected SortPreservingMergeExec over limited SortExec, got {}",
            optimized.name()
        );
    }

    #[test]
    fn merge_over_hash_join_is_kept() {
        // DataFusion reports inexact statistics for a join, which already keeps
        // the merge. Wrap the join in a node reporting exact, ordered statistics,
        // so that only the hash join itself can keep the merge.
        let build = StatsExec::plan(vec![partition(vec![
            exact_i64(0, 0, 0),
            exact_i64(5, 5, 0),
        ])]);
        let join: Arc<dyn ExecutionPlan> = Arc::new(
            HashJoinExec::try_new(
                build,
                ordered_partitions(),
                vec![(
                    Arc::new(Column::new("id", 1)),
                    Arc::new(Column::new("id", 1)),
                )],
                None,
                &JoinType::RightSemi,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap(),
        );
        let ordered_stats = vec![
            partition(vec![exact_i64(0, 100, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(101, 200, 0), exact_i64(11, 20, 0)]),
        ];
        let plan = StatsExec::over(ordered_stats, Some(join));
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();
        assert!(
            ordered_partition_ranges(&plan, &ordering).is_some(),
            "the partitions should look replaceable on statistics alone"
        );

        let optimized = optimize_any(merge_on_t(plan));

        assert!(
            optimized
                .downcast_ref::<SortPreservingMergeExec>()
                .is_some(),
            "expected SortPreservingMergeExec, got {}",
            optimized.name()
        );
    }

    /// A filter over ordered partitions, with a projection embedded in it the
    /// way `ProjectionPushdown` leaves one.
    fn filter_exec(projection: Option<Vec<usize>>) -> FilterExec {
        FilterExecBuilder::new(
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
            ordered_partitions(),
        )
        .apply_projection(projection)
        .unwrap()
        .build()
        .unwrap()
    }

    #[test]
    fn filter_passes_the_ordering_through_unchanged() {
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();

        assert_eq!(
            pass_through_ordering(&filter_exec(None), &ordering),
            Some(ordering)
        );
    }

    #[test]
    fn filter_carries_the_ordering_through_an_embedded_projection() {
        // The filter emits `id` then `t`, so the sort column it is asked about
        // is not the one its child holds under that index.
        let filter = filter_exec(Some(vec![1, 0]));
        assert_eq!(filter.schema().field(1).name(), "t");

        let ordering = LexOrdering::new(vec![asc(1, "t")]).unwrap();

        assert_eq!(
            pass_through_ordering(&filter, &ordering),
            LexOrdering::new(vec![asc(0, "t")])
        );
    }

    /// Operators that hand their input's partitions on as they are, with the
    /// same schema, need no special knowledge to be seen through.
    #[test]
    fn buffer_and_limit_pass_the_ordering_through() {
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();

        let buffer = BufferExec::new(ordered_partitions(), 1024);
        assert_eq!(
            pass_through_ordering(&buffer, &ordering),
            Some(ordering.clone())
        );

        let limit = LocalLimitExec::new(ordered_partitions(), 5);
        assert_eq!(pass_through_ordering(&limit, &ordering), Some(ordering));
    }

    /// A column the projection computes is not the one the scan arranged its
    /// files on, whatever it is called.
    #[test]
    fn computed_projection_column_is_not_carried() {
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();
        let t_plus_zero: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("t", 0)),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int64(Some(0)))),
        ));
        let projection =
            ProjectionExec::try_new(vec![(t_plus_zero, "t".to_string())], ordered_partitions())
                .unwrap();

        assert_eq!(pass_through_ordering(&projection, &ordering), None);
    }

    /// An order-preserving repartition keeps each output partition ordered,
    /// and here keeps the partition count too, so the order-maintenance flag
    /// and the count alone would let it through. It deals rows between the
    /// partitions, and the hash scheme it declares is what keeps it out.
    #[test]
    fn order_preserving_repartition_is_not_seen_through() {
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();
        let input = StatsExec::declaring(
            vec![
                partition(vec![exact_i64(0, 100, 0), exact_i64(0, 10, 0)]),
                partition(vec![exact_i64(101, 200, 0), exact_i64(11, 20, 0)]),
            ],
            Some(ordering.clone()),
        );
        let repartition = RepartitionExec::try_new(
            input,
            Partitioning::Hash(vec![Arc::new(Column::new("id", 1))], 2),
        )
        .unwrap()
        .with_preserve_order();
        assert!(repartition.maintains_input_order()[0]);
        assert_eq!(repartition.properties().partitioning.partition_count(), 2);

        assert_eq!(pass_through_ordering(&repartition, &ordering), None);
    }

    /// A parquet scan over one file, order-sensitive or not.
    fn parquet_scan(preserve_order: bool) -> Arc<dyn ExecutionPlan> {
        use datafusion::datasource::physical_plan::ParquetSource;
        use datafusion::execution::object_store::ObjectStoreUrl;
        use datafusion_datasource::file_groups::FileGroup;
        use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
        use datafusion_datasource::{PartitionedFile, TableSchema};

        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let source = Arc::new(ParquetSource::new(TableSchema::from(schema)));
        let group: FileGroup = [PartitionedFile::new("a", 10)].into_iter().collect();
        let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
            .with_file_groups(vec![group])
            .with_preserve_order(preserve_order)
            .build();
        DataSourceExec::from_data_source(config)
    }

    /// A scan that already keeps its files local is handed back as it is:
    /// asking it would copy every file for nothing.
    #[test]
    fn preserve_input_order_leaves_a_pinned_scan_untouched() {
        let scan = parquet_scan(true);
        let pinned = preserve_input_order(Arc::clone(&scan)).unwrap();
        assert!(Arc::ptr_eq(&scan, &pinned));
    }

    /// An order-insensitive scan is pinned, and the nodes above it rebuilt
    /// around the pinned leaf, whether or not they forward the request
    /// themselves.
    #[test]
    fn preserve_input_order_pins_a_scan_through_its_parents() {
        use datafusion::physical_plan::coop::CooperativeExec;

        let scan = parquet_scan(false);
        assert!(!scan_keeps_files_local(&scan));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(scan));

        let pinned = preserve_input_order(plan).unwrap();

        assert!(pinned.downcast_ref::<CooperativeExec>().is_some());
        assert!(scan_keeps_files_local(pinned.children()[0]));
    }

    #[tokio::test]
    async fn progressive_eval_does_not_advertise_unvalidated_partition_local_ordering() {
        use arrow::array::Int64Array;
        use arrow::record_batch::RecordBatch;
        use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
        use datafusion::physical_plan::collect;
        use datafusion::physical_plan::sorts::sort::SortExec;
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;

        #[derive(Debug)]
        struct PartitionedDataExec {
            batches: Vec<RecordBatch>,
            stats: Vec<Statistics>,
            cache: Arc<PlanProperties>,
        }

        impl DisplayAs for PartitionedDataExec {
            fn fmt_as(
                &self,
                _t: DisplayFormatType,
                f: &mut std::fmt::Formatter<'_>,
            ) -> std::fmt::Result {
                write!(f, "PartitionedDataExec")
            }
        }

        impl ExecutionPlan for PartitionedDataExec {
            fn name(&self) -> &'static str {
                "PartitionedDataExec"
            }

            fn properties(&self) -> &Arc<PlanProperties> {
                &self.cache
            }

            fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
                vec![]
            }

            fn with_new_children(
                self: Arc<Self>,
                _children: Vec<Arc<dyn ExecutionPlan>>,
            ) -> Result<Arc<dyn ExecutionPlan>> {
                Ok(self)
            }

            fn apply_expressions(
                &self,
                _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
            ) -> Result<TreeNodeRecursion> {
                Ok(TreeNodeRecursion::Continue)
            }

            fn execute(
                &self,
                partition: usize,
                _context: Arc<TaskContext>,
            ) -> Result<SendableRecordBatchStream> {
                let batch = self.batches[partition].clone();
                let schema = batch.schema();
                Ok(Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    futures::stream::iter(vec![Ok(batch)]),
                )))
            }

            fn statistics_from_inputs(
                &self,
                _input_stats: &[Arc<Statistics>],
                args: &StatisticsArgs,
            ) -> Result<Arc<Statistics>> {
                match args.partition() {
                    Some(idx) => Ok(Arc::new(self.stats[idx].clone())),
                    None => Ok(Arc::new(Statistics::new_unknown(&self.schema()))),
                }
            }
        }

        let schema = test_schema();
        let make_batch = |ts: Vec<i64>, ids: Vec<i64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ts)),
                    Arc::new(Int64Array::from(ids)),
                ],
            )
            .unwrap()
        };

        // Partition 0: [(1, 100), (2, 101)] — locally sorted on (t ASC, id ASC)
        // Partition 1: [(2, 0),   (3, 1)]   — locally sorted on (t ASC, id ASC)
        // Concatenating them is ordered on `t ASC` (boundary 2 == 2),
        // but NOT ordered on `(t ASC, id ASC)` since (2, 101) precedes (2, 0).
        let local_ordering = LexOrdering::new(vec![asc(0, "t"), asc(1, "id")]).unwrap();
        let source: Arc<dyn ExecutionPlan> = Arc::new(PartitionedDataExec {
            batches: vec![
                make_batch(vec![1, 2], vec![100, 101]),
                make_batch(vec![2, 3], vec![0, 1]),
            ],
            stats: vec![
                partition(vec![exact_i64(1, 2, 0), exact_i64(100, 101, 0)]),
                partition(vec![exact_i64(2, 3, 0), exact_i64(0, 1, 0)]),
            ],
            cache: Arc::new(PlanProperties::new(
                EquivalenceProperties::new_with_orderings(
                    Arc::clone(&schema),
                    vec![local_ordering.clone()],
                ),
                Partitioning::UnknownPartitioning(2),
                EmissionType::Incremental,
                Boundedness::Bounded,
            )),
        });

        // Inner merge only establishes global order on `t ASC`, while an outer
        // SortExec requires `(t ASC, id ASC)`.
        let merge = merge_on_t(source);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(local_ordering.clone(), merge));

        // 1. ProgressiveEvalRule replaces SortPreservingMergeExec(t) with ProgressiveEvalExec.
        let after_progressive = optimize_any(plan);
        let t_ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();
        let progressive = after_progressive.children()[0];
        assert!(
            progressive.downcast_ref::<ProgressiveEvalExec>().is_some(),
            "expected ProgressiveEvalExec under SortExec, got {}",
            progressive.name()
        );
        assert!(
            progressive
                .equivalence_properties()
                .ordering_satisfy(t_ordering.iter().cloned())
                .unwrap(),
            "ProgressiveEvalExec must advertise the validated merge ordering (t)"
        );
        assert!(
            !progressive
                .equivalence_properties()
                .ordering_satisfy(local_ordering.iter().cloned())
                .unwrap(),
            "ProgressiveEvalExec must not advertise unvalidated partition-local ordering (t, id)"
        );

        // 2. A subsequent DataFusion physical optimization pass must keep the
        //    outer SortExec([t ASC, id ASC]) and yield globally sorted rows.
        let config = ConfigOptions::default();
        let mut after_optimizer = after_progressive;
        for rule in PhysicalOptimizer::new().rules {
            after_optimizer = rule.optimize(after_optimizer, &config).unwrap();
        }
        assert!(
            after_optimizer
                .exists(|node| Ok(node.downcast_ref::<SortExec>().is_some()))
                .unwrap(),
            "outer SortExec on (t, id) must be retained"
        );

        let batches = collect(after_optimizer, Arc::new(TaskContext::default()))
            .await
            .unwrap();
        let rows: Vec<(i64, i64)> = batches
            .iter()
            .flat_map(|b| {
                let ts = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                let ids = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                (0..b.num_rows()).map(move |i| (ts.value(i), ids.value(i)))
            })
            .collect();

        assert_eq!(rows, vec![(1, 100), (2, 0), (2, 101), (3, 1)]);
    }

    #[tokio::test]
    async fn progressive_eval_clears_per_partition_constants() {
        use arrow::array::{Int64Array, StringArray};
        use arrow::record_batch::RecordBatch;
        use datafusion::physical_expr::{AcrossPartitions, ConstExpr};
        use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
        use datafusion::physical_plan::collect;
        use datafusion::physical_plan::sorts::sort::SortExec;
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use datafusion::physical_plan::union::UnionExec;

        #[derive(Debug)]
        struct ConstLeafExec {
            batch: RecordBatch,
            stats: Statistics,
            cache: Arc<PlanProperties>,
        }

        impl DisplayAs for ConstLeafExec {
            fn fmt_as(
                &self,
                _t: DisplayFormatType,
                f: &mut std::fmt::Formatter<'_>,
            ) -> std::fmt::Result {
                write!(f, "ConstLeafExec")
            }
        }

        impl ExecutionPlan for ConstLeafExec {
            fn name(&self) -> &'static str {
                "ConstLeafExec"
            }

            fn properties(&self) -> &Arc<PlanProperties> {
                &self.cache
            }

            fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
                vec![]
            }

            fn with_new_children(
                self: Arc<Self>,
                _children: Vec<Arc<dyn ExecutionPlan>>,
            ) -> Result<Arc<dyn ExecutionPlan>> {
                Ok(self)
            }

            fn apply_expressions(
                &self,
                _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
            ) -> Result<TreeNodeRecursion> {
                Ok(TreeNodeRecursion::Continue)
            }

            fn execute(
                &self,
                partition: usize,
                _context: Arc<TaskContext>,
            ) -> Result<SendableRecordBatchStream> {
                assert_eq!(partition, 0);
                let batch = self.batch.clone();
                let schema = batch.schema();
                Ok(Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    futures::stream::iter(vec![Ok(batch)]),
                )))
            }

            fn statistics_from_inputs(
                &self,
                _input_stats: &[Arc<Statistics>],
                _args: &StatisticsArgs,
            ) -> Result<Arc<Statistics>> {
                Ok(Arc::new(self.stats.clone()))
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("t", DataType::Int64, false),
            Field::new("p", DataType::Utf8, false),
        ]));
        let t_ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();

        let make_leaf = |ts: Vec<i64>, p_val: &str| -> Arc<dyn ExecutionPlan> {
            let min_t = *ts.first().unwrap();
            let max_t = *ts.last().unwrap();
            let p_scalar = ScalarValue::Utf8(Some(p_val.to_string()));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ts.clone())),
                    Arc::new(StringArray::from(vec![p_val; ts.len()])),
                ],
            )
            .unwrap();
            let stats = Statistics {
                num_rows: Precision::Exact(ts.len()),
                total_byte_size: Precision::Exact(100),
                column_statistics: vec![
                    exact_i64(min_t, max_t, 0),
                    ColumnStatistics {
                        null_count: Precision::Exact(0),
                        min_value: Precision::Exact(p_scalar.clone()),
                        max_value: Precision::Exact(p_scalar.clone()),
                        ..Default::default()
                    },
                ],
            };
            let mut eq = EquivalenceProperties::new_with_orderings(
                Arc::clone(&schema),
                vec![t_ordering.clone()],
            );
            eq.add_constants([ConstExpr::new(
                Arc::new(Column::new("p", 1)),
                AcrossPartitions::Uniform(Some(p_scalar)),
            )])
            .unwrap();
            Arc::new(ConstLeafExec {
                batch,
                stats,
                cache: Arc::new(PlanProperties::new(
                    eq,
                    Partitioning::UnknownPartitioning(1),
                    EmissionType::Incremental,
                    Boundedness::Bounded,
                )),
            })
        };

        // Two single-partition leaves with disjoint `t` ranges and distinct
        // constant values of `p`. `UnionExec` marks `p` as a per-partition
        // (heterogeneous) constant across its 2 partitions.
        let union =
            UnionExec::try_new(vec![make_leaf(vec![1, 2], "a"), make_leaf(vec![3, 4], "b")])
                .unwrap();
        let merge = merge_on_t(union);
        let p_desc_t_asc = LexOrdering::new(vec![desc(1, "p"), asc(0, "t")]).unwrap();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(p_desc_t_asc.clone(), merge));

        // 1. ProgressiveEvalRule rewrites SortPreservingMergeExec(t) to ProgressiveEvalExec.
        let after_progressive = optimize_any(plan);
        let progressive = after_progressive.children()[0];
        assert!(
            progressive.downcast_ref::<ProgressiveEvalExec>().is_some(),
            "expected ProgressiveEvalExec under SortExec, got {}",
            progressive.name()
        );
        assert!(
            !progressive
                .equivalence_properties()
                .ordering_satisfy(p_desc_t_asc.iter().cloned())
                .unwrap(),
            "ProgressiveEvalExec must clear per-partition constants so (p DESC, t ASC) is not satisfied"
        );

        // 2. Subsequent physical optimization must keep the outer SortExec([p DESC, t ASC])
        //    so 'b' rows sort before 'a' rows.
        let config = ConfigOptions::default();
        let mut after_optimizer = after_progressive;
        for rule in PhysicalOptimizer::new().rules {
            after_optimizer = rule.optimize(after_optimizer, &config).unwrap();
        }
        assert!(
            after_optimizer
                .exists(|node| Ok(node.downcast_ref::<SortExec>().is_some()))
                .unwrap(),
            "outer SortExec on (p DESC, t ASC) must be retained"
        );

        let batches = collect(after_optimizer, Arc::new(TaskContext::default()))
            .await
            .unwrap();
        let rows: Vec<(i64, String)> = batches
            .iter()
            .flat_map(|b| {
                let ts = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                let ps = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
                (0..b.num_rows()).map(move |i| (ts.value(i), ps.value(i).to_string()))
            })
            .collect();

        assert_eq!(
            rows,
            vec![
                (3, "b".to_string()),
                (4, "b".to_string()),
                (1, "a".to_string()),
                (2, "a".to_string()),
            ]
        );
    }
}
