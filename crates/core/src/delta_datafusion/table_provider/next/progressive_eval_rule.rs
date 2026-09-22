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

use datafusion::common::config::ConfigOptions;
use datafusion::common::stats::Statistics;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{Result, ScalarValue};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, Partitioning, PhysicalSortExpr,
};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties as _};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;

use super::progressive_eval::ProgressiveEvalExec;
use super::scan::DeltaScanExec;
use std::cmp::Ordering;
use std::sync::Arc;

/// Replace [`SortPreservingMergeExec`] with [`ProgressiveEvalExec`] when its
/// input partitions are mutually non-overlapping, range-ordered slices of the
/// merge ordering, so concatenating them already yields globally ordered
/// output. See the [module docs](self) for background.
#[derive(Debug, Default)]
pub struct ProgressiveEvalRule;

impl ProgressiveEvalRule {
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
        plan.transform_down(|plan| {
            let Some(merge) =
                (plan.as_ref() as &dyn ExecutionPlan).downcast_ref::<SortPreservingMergeExec>()
            else {
                return Ok(Transformed::no(plan));
            };
            let input = merge.input();
            if contains_hash_join(input)? {
                return Ok(Transformed::no(plan));
            }
            let ranges = match ordered_partition_ranges(input, merge.expr()) {
                Some(ranges) => Some(ranges),
                // Statistics could not prove the partitions ordered. A Delta
                // scan may know it anyway - from a sort pushdown that built
                // the partitions that way, or from the table's assertion that
                // its files never overlap; the ranges are then only
                // descriptive, so it does not matter that they may be
                // unavailable.
                None if partitions_declared_disjoint(input, merge.expr())? => {
                    leading_partition_ranges(input, merge.expr())
                }
                None => return Ok(Transformed::no(plan)),
            };
            let input = preserve_input_order(Arc::clone(input))?;
            let replacement = ProgressiveEvalExec::new(input, ranges, merge.fetch());
            Ok(Transformed::yes(Arc::new(replacement) as _))
        })
        .data()
    }

    fn name(&self) -> &str {
        "ProgressiveEvalRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
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
    (0..plan.output_partitioning().partition_count())
        .map(|partition_idx| {
            let stats = plan.partition_statistics(Some(partition_idx)).ok()?;
            let (starts, ends, _) = get_ordering_stats(&stats, ordering)?;
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

/// Mark the data sources under `plan` order-sensitive, so that they keep the
/// partition-to-data mapping their per-partition statistics describe.
///
/// A file scan that declares no output ordering is free to let its sibling
/// streams steal unopened files from one another at execution time, which is
/// what `datafusion.execution.enable_file_stream_work_stealing` does by
/// default. The partition boundaries proven above are then meaningless: a
/// partition can emit rows planned for another. This is the case for a scan
/// under a per-partition `SortExec` whose file groups were arranged by
/// statistics without an ordering being declared for them.
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
/// order-sensitive, or partitioned by file group, which disables the sharing
/// as well.
fn scan_keeps_files_local(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<DataSourceExec>()
        .and_then(|scan| scan.data_source().as_ref().downcast_ref::<FileScanConfig>())
        .is_some_and(|config| config.preserve_order || config.partitioned_by_file_group)
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
fn ordered_partition_ranges(
    plan: &Arc<dyn ExecutionPlan>,
    ordering: &LexOrdering,
) -> Option<Vec<(ScalarValue, ScalarValue)>> {
    let partition_count = plan.output_partitioning().partition_count();
    let mut prev_ends: Vec<ScalarValue> = Vec::new();
    let mut prev_null_counts: Vec<usize> = Vec::new();
    let mut first_col_ordering = Vec::with_capacity(partition_count);
    for partition_idx in 0..partition_count {
        let partition_stats = plan.partition_statistics(Some(partition_idx)).ok()?;
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
    use datafusion::common::{JoinType, NullEquality};
    use datafusion::execution::{SendableRecordBatchStream, TaskContext};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion::physical_plan::buffer::BufferExec;
    use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
    use datafusion::physical_plan::filter::FilterExecBuilder;
    use datafusion::physical_plan::joins::PartitionMode;
    use datafusion::physical_plan::limit::LocalLimitExec;
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::physical_plan::{DisplayAs, DisplayFormatType, PlanProperties};

    /// Leaf plan reporting fixed per-partition statistics; never executed.
    #[derive(Debug)]
    struct StatsExec {
        stats: Vec<Statistics>,
        cache: Arc<PlanProperties>,
    }

    impl StatsExec {
        fn new(stats: Vec<Statistics>) -> Arc<dyn ExecutionPlan> {
            Self::declaring(stats, None)
        }

        /// A leaf that also declares `ordering` as the order of each of its
        /// partitions.
        fn declaring(
            stats: Vec<Statistics>,
            ordering: Option<LexOrdering>,
        ) -> Arc<dyn ExecutionPlan> {
            let cache = Arc::new(PlanProperties::new(
                EquivalenceProperties::new_with_orderings(test_schema(), ordering),
                Partitioning::UnknownPartitioning(stats.len()),
                EmissionType::Incremental,
                Boundedness::Bounded,
            ));
            Arc::new(Self { stats, cache })
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
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            unimplemented!("StatsExec is only used for planning")
        }

        fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
            match partition {
                Some(idx) => Ok(Arc::new(self.stats[idx].clone())),
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
            partition(vec![exact_i64(0, 99, 0), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(100, 200, 2), exact_i64(0, 10, 0)]),
        ]);
        assert!(ordered_partition_ranges(&plan, &ordering).is_some());

        // Nulls in the first partition would surface mid-stream: rejected.
        let plan = StatsExec::new(vec![
            partition(vec![exact_i64(0, 99, 2), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(100, 200, 0), exact_i64(0, 10, 0)]),
        ]);
        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    #[test]
    fn nulls_only_allowed_in_first_partition_for_nulls_first() {
        let ordering = LexOrdering::new(vec![desc(0, "t")]).unwrap();

        // Descending partitions with nulls leading in the first: accepted.
        let plan = StatsExec::new(vec![
            partition(vec![exact_i64(100, 200, 2), exact_i64(0, 10, 0)]),
            partition(vec![exact_i64(0, 99, 0), exact_i64(0, 10, 0)]),
        ]);
        assert!(ordered_partition_ranges(&plan, &ordering).is_some());

        // Nulls in the last partition sort before its values: rejected.
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
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
        let plan = StatsExec::new(vec![
            partition(vec![exact_i64(0, 99, 0)]),
            partition(vec![exact_i64(100, 200, 0)]),
        ]);
        let ordering = LexOrdering::new(vec![asc(1, "id")]).unwrap();

        assert!(ordered_partition_ranges(&plan, &ordering).is_none());
    }

    /// Two partitions with disjoint `t` ranges: a merge on `t` over them is
    /// replaceable by a progressive eval.
    fn ordered_partitions() -> Arc<dyn ExecutionPlan> {
        StatsExec::new(vec![
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

    #[test]
    fn merge_over_ordered_partitions_is_replaced() {
        let optimized = optimize(merge_on_t(ordered_partitions()));

        assert!(
            optimized.downcast_ref::<ProgressiveEvalExec>().is_some(),
            "expected ProgressiveEvalExec, got {}",
            optimized.name()
        );
    }

    #[test]
    fn merge_over_hash_join_is_kept() {
        // A semi join keeps the probe side's statistics, so the merge above it
        // would be replaceable on statistics alone.
        let build = StatsExec::new(vec![partition(vec![
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
        let ordering = LexOrdering::new(vec![asc(0, "t")]).unwrap();
        assert!(
            ordered_partition_ranges(&join, &ordering).is_some(),
            "the join's partitions should look replaceable on statistics alone"
        );

        let optimized = optimize(merge_on_t(join));

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
        let source = Arc::new(ParquetSource::new(TableSchema::new(schema, vec![])));
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
        use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;

        let scan = parquet_scan(false);
        assert!(!scan_keeps_files_local(&scan));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CoalesceBatchesExec::new(scan, 8));

        let pinned = preserve_input_order(plan).unwrap();

        assert!(pinned.downcast_ref::<CoalesceBatchesExec>().is_some());
        assert!(scan_keeps_files_local(pinned.children()[0]));
    }
}
