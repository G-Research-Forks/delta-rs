//! Physical optimizer rule that replaces a merge of ordered partitions with
//! a plain concatenation.
//!
//! A [`SortPreservingMergeExec`] combines its input partitions into one globally
//! ordered stream. That merge compares rows across all partitions and executes every
//! partition up front.
//! If the partitions are non-overlapping with respect to the sort order, then
//! this merge is unnecessary, and we can stream the partitions one after another
//! using a [`ProgressiveEvalExec`].

use datafusion::common::config::ConfigOptions;
use datafusion::common::stats::Statistics;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{Result, ScalarValue};
use datafusion::physical_expr::LexOrdering;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties as _};

use super::progressive_eval::ProgressiveEvalExec;
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
            let Some(ranges) = ordered_partition_ranges(input, merge.expr()) else {
                return Ok(Transformed::no(plan));
            };
            if ranges.len() != input.output_partitioning().partition_count() {
                return Ok(Transformed::no(plan));
            }
            let replacement =
                ProgressiveEvalExec::new(Arc::clone(input), Some(ranges), merge.fetch());
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

/// To be able to convert to a ProgressiveEval, we need the partitions to
/// be ordered with respect to each other.
/// If this is the case, return a vector of the (start, end) range for the first
/// sort term from each partition, otherwise return None.
fn ordered_partition_ranges(
    plan: &Arc<dyn ExecutionPlan>,
    ordering: &LexOrdering,
) -> Option<Vec<(ScalarValue, ScalarValue)>> {
    // TODO: Rather than rely on the input partitions being in order already,
    // we should allow reordering the partitions.
    let partition_count = plan.output_partitioning().partition_count();
    let mut prev_ends: Vec<ScalarValue> = Vec::new();
    let mut first_col_ordering = Vec::with_capacity(partition_count);
    for partition_idx in 0..partition_count {
        let partition_stats = plan.partition_statistics(Some(partition_idx)).ok()?;
        let (starts, ends, null_counts) = get_ordering_stats(&partition_stats, ordering)?;
        for (i, sort_expr) in ordering.iter().enumerate() {
            if null_counts[i] != 0 {
                // If there are any nulls, this must either be the first or last partition depending
                // on whether nulls_first is set.
                if (sort_expr.options.nulls_first && partition_idx != 0)
                    || (!sort_expr.options.nulls_first && partition_idx != partition_count - 1)
                {
                    return None;
                }
            }
            if partition_idx != 0 {
                // Check partition is ordered correctly with respect to the previous partition.
                // Incomparable values (partial_cmp is None) bail out rather than being
                // silently treated as equal.
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
        starts.push(start.clone());
        ends.push(end.clone());
        null_counts.push(*col_stats.null_count.get_value()?);
    }

    Some((starts, ends, null_counts))
}
