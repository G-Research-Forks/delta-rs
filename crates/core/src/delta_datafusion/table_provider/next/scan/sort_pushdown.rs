//! Sort pushdown implementation for [`DeltaScanExec`](super::exec::DeltaScanExec).
//!
//! DataFusion's `PushdownSort` physical optimizer rule allows regrouping files to
//! satisfy a query order and remove the need for a sort. This module handles this
//! regrouping for queries where the sort order starts with partition columns, and
//! any remaining sort expressions are a prefix of the declared file sort order.
//!
//! * Every data file belongs to exactly one partition, so within a file every
//!   partition column is constant. A file sorted by the declared file sort order
//!   is therefore also sorted by `[<any partition columns…>, <file sort order…>]`.
//! * Partition-column values are known exactly per file, so files can be
//!   bucketed by partition-column and ordered based on the file sort order, when
//!   the files are non-overlapping with respect to the sort order.
//!
//! The resulting file groups are mutually non-overlapping and range-ordered on
//! the combined ordering, so [`DeltaScanExec`](super::exec::DeltaScanExec)
//! advertises it as an output ordering; the `SortExec` is removed and the
//! `SortPreservingMergeExec` above it is later rewritten to a
//! `ProgressiveEvalExec` by [`ProgressiveEvalRule`](super::super::ProgressiveEvalRule).

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_schema::{SchemaRef, SortOptions};
use datafusion::common::{
    ColumnStatistics, HashMap, Result, stats::Precision, utils::compare_rows,
};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::scalar::ScalarValue;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_groups::FileGroup;

use super::{
    DeltaPartitionValues, KeyTypes, MAX_PARTITION_DICT_CARDINALITY, chunk_ordered_files,
    max_num_groups, non_overlapping_file_order, null_free_ordering_prefix,
};

/// Outcome of planning a sort pushdown: the regrouped, range-ordered file groups
/// plus the per-execution-partition partition-column statistics that
/// [`DeltaScanExec`](super::exec::DeltaScanExec) needs to expose so downstream
/// rules can prove the partitions non-overlapping.
pub(super) struct SortPushdownPlan {
    /// File groups, mutually non-overlapping and range-ordered on the combined
    /// `[partition prefix…, file sort order prefix…]` ordering.
    pub(super) file_groups: Vec<FileGroup>,
    /// Per-group exact statistics for the leading partition columns of the
    /// requested ordering, keyed by logical column name. Aligned with
    /// `file_groups` by index; empty for a group whose endpoints cannot be
    /// read as per-column extrema (see `group_prefix_stats`).
    pub(super) per_partition_stats: Vec<HashMap<String, ColumnStatistics>>,
}

/// A partition-column sort key extracted from the requested ordering.
struct PrefixColumn {
    /// Logical (output-schema) column name.
    name: String,
    /// Index of this column in the table's partition columns, i.e. into the
    /// [`DeltaPartitionValues`] attached to each file.
    position: usize,
    options: SortOptions,
}

/// A requested ordering this scan can satisfy by regrouping its files: a
/// non-empty prefix of partition columns, followed by a (possibly empty) prefix
/// of the declared file sort order.
pub(super) struct OrderShape {
    prefix: Vec<PrefixColumn>,
    /// The suffix, expressed over the parquet read schema, or `None` when the
    /// requested ordering is partition columns only.
    suffix: Option<LexOrdering>,
}

/// Re-express `order`, whose columns are bound to `from`, over `to` by column
/// name. Returns `None` when a sort expression is not a plain column or `to`
/// has no column of that name.
pub(super) fn rebind_ordering(
    order: &[PhysicalSortExpr],
    from: &SchemaRef,
    to: &SchemaRef,
) -> Option<Vec<PhysicalSortExpr>> {
    order
        .iter()
        .map(|sort_expr| {
            let column = sort_expr.expr.downcast_ref::<Column>()?;
            let name = from.fields().get(column.index())?.name();
            let index = to.index_of(name).ok()?;
            Some(PhysicalSortExpr::new(
                Arc::new(Column::new(name, index)),
                sort_expr.options,
            ))
        })
        .collect()
}

/// Classify `order` as a partition-column prefix plus a file-sort-order suffix,
/// or `None` when this scan cannot satisfy it by regrouping.
///
/// * `output_schema` – the scan's logical output schema (`order` is bound to it).
/// * `parquet_read_schema` – physical file columns, which `file_sort_order` is
///   bound to; also the prefix of the file scan's table schema, so its column
///   indices address `PartitionedFile` statistics directly.
/// * `partition_column_names` – logical partition columns, in table order.
/// * `file_sort_order` – resolved per-file ordering over `parquet_read_schema`.
pub(super) fn analyze_order(
    order: &LexOrdering,
    output_schema: &SchemaRef,
    parquet_read_schema: &SchemaRef,
    partition_column_names: &[String],
    file_sort_order: Option<&LexOrdering>,
) -> Option<OrderShape> {
    if partition_column_names.is_empty() {
        return None;
    }
    let column_name = |schema: &SchemaRef, sort_expr: &PhysicalSortExpr| -> Option<String> {
        let col = sort_expr.expr.downcast_ref::<Column>()?;
        Some(schema.fields().get(col.index())?.name().clone())
    };

    // Split the ordering at the first non-partition column.
    let mut prefix = Vec::new();
    for sort_expr in order.iter() {
        let name = column_name(output_schema, sort_expr)?;
        let Some(position) = partition_column_names.iter().position(|p| *p == name) else {
            break;
        };
        prefix.push(PrefixColumn {
            name,
            position,
            options: sort_expr.options,
        });
    }
    if prefix.is_empty() {
        // Sort order does not start with partition columns, this can't
        // be handled by a file regrouping.
        return None;
    }
    let suffix_exprs = &order[prefix.len()..];
    if suffix_exprs.is_empty() {
        return Some(OrderShape {
            prefix,
            suffix: None,
        });
    }

    // A partition column after the prefix cannot be honored by regrouping:
    // the files are not ordered on it.
    for suffix_expr in suffix_exprs {
        if partition_column_names.contains(&column_name(output_schema, suffix_expr)?) {
            return None;
        }
    }
    // The suffix must match a prefix of the declared file sort order, column for
    // column and direction for direction, re-expressed over the read schema
    // (which is what the declared order is bound to).
    let file_sort_order = file_sort_order?;
    let suffix = rebind_ordering(suffix_exprs, output_schema, parquet_read_schema)?;
    if suffix.len() > file_sort_order.len() || suffix[..] != file_sort_order[..suffix.len()] {
        return None;
    }

    Some(OrderShape {
        prefix,
        suffix: LexOrdering::new(suffix),
    })
}

/// Files sharing one partition-prefix key, ordered on the suffix.
struct Bucket {
    key: Vec<ScalarValue>,
    files: Vec<PartitionedFile>,
}

impl Bucket {
    /// Cut the bucket's files into `groups` contiguous pieces, each a bucket
    /// under the same key so that its statistics stay exact. See
    /// [`chunk_ordered_files`] for the exact count.
    fn split(self, groups: usize) -> impl Iterator<Item = Bucket> {
        let key = self.key;
        chunk_ordered_files(self.files, groups)
            .into_iter()
            .map(move |piece| Bucket {
                key: key.clone(),
                files: piece.into_inner(),
            })
    }
}

/// Regroup `files` so that reading each group in order yields the ordering
/// described by `shape`, or return `None` when that cannot be guaranteed (the
/// caller then reports `Unsupported`).
///
/// * `parquet_read_schema` – physical file columns; its indices address
///   `PartitionedFile` statistics directly.
/// * `files` - Files to be regrouped. May have a range set, but this is ignored,
///   and the planned pushdown always uses the full file range.
/// * `target_groups` – desired execution-partition count; the result may hold
///   more groups, up to [`group_budget`].
pub(super) fn plan_sort_pushdown(
    shape: &OrderShape,
    parquet_read_schema: &SchemaRef,
    files: &[&PartitionedFile],
    target_groups: usize,
) -> Result<Option<SortPushdownPlan>> {
    if files.is_empty() {
        return Ok(None);
    }
    let prefix = &shape.prefix;
    let prefix_options: Vec<SortOptions> = prefix.iter().map(|col| col.options).collect();
    let cmp_prefix = |a: &[ScalarValue], b: &[ScalarValue]| compare_rows(a, b, &prefix_options);

    // --- Pair every file with its exact partition-prefix key. ---
    // Sort lightweight (key, reference) pairs rather than the large file objects.
    // Files are borrowed until every check has passed to keep a failed pushdown-sort cheap.
    let mut keyed: Vec<(Vec<ScalarValue>, &PartitionedFile)> = Vec::with_capacity(files.len());
    let mut key_types = KeyTypes::default();
    for &file in files {
        let Some(values) = file.extensions.get::<DeltaPartitionValues>() else {
            return Ok(None);
        };
        let mut key = Vec::with_capacity(prefix.len());
        for col in prefix {
            match values.0.get(col.position) {
                // A null partition value cannot be ordered against non-null ones
                // via min/max statistics.
                Some(value) if !value.is_null() => key.push(value.clone()),
                _ => return Ok(None),
            }
        }
        if !key_types.accept(&key) {
            return Ok(None);
        }
        keyed.push((key, file));
    }
    keyed.sort_by(|a, b| cmp_prefix(&a.0, &b.0).expect("keys share a non-nested type per column"));

    // --- Cut the sorted files into buckets of equal prefix key. ---
    // One comparator decides both the bucket boundary and its validity: equal
    // keys extend the current bucket and a strict step opens a new one; the
    // keys were sorted with the same comparator, so any other outcome is unexpected
    // and prevents the pushdown-sort from applying.
    // Buckets hold references: nothing has been cloned yet.
    let mut key_buckets: Vec<(Vec<ScalarValue>, Vec<&PartitionedFile>)> = Vec::new();
    for (key, file) in keyed {
        match key_buckets.last_mut() {
            None => key_buckets.push((key, vec![file])),
            Some((last_key, bucket)) => match cmp_prefix(last_key, &key) {
                Ok(Ordering::Equal) => bucket.push(file),
                Ok(Ordering::Less) => key_buckets.push((key, vec![file])),
                _ => return Ok(None),
            },
        }
    }

    // --- Order each bucket on the suffix, refusing overlap ---
    // Still keep working with references only.
    let mut ordered_buckets: Vec<(Vec<ScalarValue>, Vec<&PartitionedFile>)> =
        Vec::with_capacity(key_buckets.len());
    for (key, bucket) in key_buckets {
        let bucket = match shape.suffix.as_ref() {
            Some(suffix) if bucket.len() >= 2 => {
                // Concatenating several files into one group interleaves each
                // file's nulls into the middle of the group's stream, which
                // min/max cannot detect, so every suffix column must be
                // provably null-free.
                let null_free =
                    null_free_ordering_prefix(suffix, parquet_read_schema, bucket.iter().copied());
                if null_free.is_none_or(|p| p.len() < suffix.len()) {
                    return Ok(None);
                }
                let Ok(bucket) = non_overlapping_file_order(bucket, |file| *file, suffix) else {
                    return Ok(None);
                };
                bucket
            }
            _ => bucket,
        };
        ordered_buckets.push((key, bucket));
    }

    // --- Every check has passed: materialize the buckets. ---
    // A file that arrived split into byte ranges is represented by one of its
    // pieces (see `coalesce_file_ranges`); the copy reads the whole file.
    let buckets: Vec<Bucket> = ordered_buckets
        .into_iter()
        .map(|(key, bucket)| Bucket {
            key,
            files: bucket
                .into_iter()
                .map(|file| {
                    let mut file = file.clone();
                    file.range = None;
                    file
                })
                .collect(),
        })
        .collect();

    // --- Cut the buckets into groups and read the statistics off their keys. ---
    let Some(groups) = chunk_buckets(buckets, target_groups) else {
        return Ok(None);
    };
    let (file_groups, per_partition_stats) = groups
        .into_iter()
        .map(|group| {
            let stats = group_prefix_stats(&group, prefix);
            let files: Vec<PartitionedFile> =
                group.into_iter().flat_map(|bucket| bucket.files).collect();
            (FileGroup::from(files), stats)
        })
        .unzip();

    Ok(Some(SortPushdownPlan {
        file_groups,
        per_partition_stats,
    }))
}

/// Exact statistics for the prefix columns of one group, keyed by logical
/// column name, read off the keys of its first and last buckets.
///
/// The keys are the group's range endpoints under the requested ordering, so
/// for a descending column the first key holds the larger value. Per-column
/// extrema can only be read off the endpoints while at most the last prefix
/// column changes across the group: a run from `[A, 2]` to `[B, 0]` is
/// ordered, but `sub` really covers `0..2` there, and no `ColumnStatistics`
/// describes both that range and where the run begins. Such a group publishes
/// nothing, and `DeltaScanExec` describes it from the table-wide aggregate
/// instead - exact for a lone execution partition, inexact otherwise, which
/// cannot prove neighbouring partitions disjoint, so the merge above the scan
/// stays while the sort is still gone.
fn group_prefix_stats(
    group: &[Bucket],
    prefix: &[PrefixColumn],
) -> HashMap<String, ColumnStatistics> {
    let (Some(first), Some(last)) = (group.first(), group.last()) else {
        return HashMap::new();
    };
    let leading = prefix.len().saturating_sub(1);
    if first.key[..leading] != last.key[..leading] {
        return HashMap::new();
    }
    prefix
        .iter()
        .enumerate()
        .map(|(i, col)| {
            let (min, max) = if col.options.descending {
                (&last.key[i], &first.key[i])
            } else {
                (&first.key[i], &last.key[i])
            };
            (
                col.name.clone(),
                ColumnStatistics {
                    null_count: Precision::Exact(0),
                    min_value: Precision::Exact(min.clone()),
                    max_value: Precision::Exact(max.clone()),
                    ..Default::default()
                },
            )
        })
        .collect()
}

/// Largest group count a pushdown may produce for `target_groups`.
///
/// The group count may exceed the target so that a group can be cut at every
/// change of a leading partition column. That keeps each group's
/// partition-column statistics exact, which lets `ProgressiveEvalRule` prove
/// neighbouring groups disjoint and replace the `SortPreservingMergeExec`
/// above with a `ProgressiveEvalExec`.
///
/// The replacement is not guaranteed, and a merge that is kept opens every
/// partition at once, so the overshoot is capped at [`max_num_groups`], the
/// same bound a scan with a declared file sort order already gets.
pub(super) fn group_budget(target_groups: usize) -> usize {
    if target_groups <= 1 {
        // A single target group is the one case that cannot absorb any overshoot:
        // `PushdownSort` deletes the `SortExec` on `Exact` without re-checking the
        // partitioning, and a single-partition input means the deleted sort was the
        // global one with no merge operator above it, so extra partitions would be
        // coalesced in arbitrary order. A multi-partition input implies a
        // per-partition sort whose consumers do not care how many sorted partitions
        // they get, so a bounded overshoot is sound there.
        1
    } else {
        max_num_groups(target_groups)
    }
}

/// Cut ordered, mutually non-overlapping buckets into roughly `target` groups
/// of whole buckets (or slices of a single bucket), never more than
/// [`group_budget`] allows. `None` only when even that is impossible: more
/// files than the budget times the file-id dictionary can key.
///
/// Group boundaries fall on bucket boundaries, so consecutive groups take a
/// strict step on the partition prefix. When there are more buckets than
/// groups they are packed together; when there are fewer, the largest are
/// split.
///
/// Packing first cuts a group wherever a prefix column other than the last
/// changes, so that every group's statistics can be read exactly off its
/// endpoints (see [`group_prefix_stats`]) and `ProgressiveEvalRule` can prove
/// the groups disjoint. When those cuts alone overrun the budget - many
/// distinct leading values, or a single target group - the buckets are packed
/// by file count alone instead. That still removes the sort; only the merge
/// above it has to stay.
fn chunk_buckets(buckets: Vec<Bucket>, target: usize) -> Option<Vec<Vec<Bucket>>> {
    let target = target.max(1);
    let max_groups = group_budget(target);

    // Split any bucket the file-id dictionary cannot key.
    let buckets: Vec<Bucket> = buckets
        .into_iter()
        .flat_map(|bucket| bucket.split(1))
        .collect();
    if buckets.is_empty() {
        return Some(Vec::new());
    }

    let groups = if buckets.len() < target {
        split_buckets(buckets, target)
    } else {
        let groups = pack_buckets(buckets, target, true);
        if groups.len() <= max_groups {
            groups
        } else {
            // Failed to pack into fewer than the max allowed, fall back to allowing
            // buckets to be split within a run of a prefix column.
            pack_buckets(groups.into_iter().flatten().collect(), target, false)
        }
    };
    (groups.len() <= max_groups).then_some(groups)
}

/// Fewer buckets than groups: split each bucket, giving it a share of the
/// group budget proportional to its file count and at least one group. The
/// remaining budget is derived from the emitted group count because
/// `chunk_ordered_files` can emit more groups than asked for (the file-id
/// dictionary cap).
fn split_buckets(buckets: Vec<Bucket>, target: usize) -> Vec<Vec<Bucket>> {
    let mut out = Vec::with_capacity(target);
    let mut files_left: usize = buckets.iter().map(|bucket| bucket.files.len()).sum();
    for bucket in buckets {
        let files = bucket.files.len();
        let groups_left = target.saturating_sub(out.len()).max(1);
        let share = (groups_left * files).div_ceil(files_left).clamp(1, files);
        out.extend(bucket.split(share).map(|piece| vec![piece]));
        files_left -= files;
    }
    out
}

/// Pack contiguous runs of whole buckets into `target` groups, balanced by
/// file count. A run is also cut when it would outgrow the file-id dictionary
/// and, with `cut_on_leading_change`, when the next bucket differs in a prefix
/// column other than the last (see [`chunk_buckets`]).
fn pack_buckets(
    buckets: Vec<Bucket>,
    target: usize,
    cut_on_leading_change: bool,
) -> Vec<Vec<Bucket>> {
    let mut out: Vec<Vec<Bucket>> = Vec::with_capacity(target);
    let mut files_left: usize = buckets.iter().map(|bucket| bucket.files.len()).sum();
    let mut buckets_left = buckets.len();
    let mut run: Vec<Bucket> = Vec::new();
    let mut run_files = 0;

    for bucket in buckets {
        buckets_left -= 1;
        let files = bucket.files.len();
        let leading = bucket.key.len() - 1;
        let extends = run.last().is_some_and(|last| {
            run_files + files <= MAX_PARTITION_DICT_CARDINALITY
                && (!cut_on_leading_change || last.key[..leading] == bucket.key[..leading])
        });
        // Emit the open run and charge it against the remaining files.
        if !run.is_empty() && !extends {
            files_left -= std::mem::take(&mut run_files);
            out.push(std::mem::take(&mut run));
        }
        run_files += files;
        run.push(bucket);

        // Close the run once it has its share of the files, or once every
        // remaining bucket is needed to fill a remaining group. The remaining
        // group budget is derived from the emitted group count so that forced
        // cuts are charged for the groups they emit.
        let groups_left = target.saturating_sub(out.len()).max(1);
        let share = files_left.div_ceil(groups_left);
        if run_files >= share || buckets_left < groups_left {
            files_left -= std::mem::take(&mut run_files);
            out.push(std::mem::take(&mut run));
        }
    }
    if !run.is_empty() {
        out.push(run);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::Statistics;

    use super::*;
    use crate::delta_datafusion::file_id::wrap_file_id_value;

    fn asc(index: usize, name: &str) -> PhysicalSortExpr {
        PhysicalSortExpr::new(
            Arc::new(Column::new(name, index)),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )
    }

    fn desc(index: usize, name: &str) -> PhysicalSortExpr {
        PhysicalSortExpr::new(
            Arc::new(Column::new(name, index)),
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        )
    }

    fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("part", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]))
    }

    fn parquet_read_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            // Nullable so multi-file groups exercise the null-freeness check.
            Field::new("timestamp", DataType::Int64, true),
            Field::new("value", DataType::Int64, true),
        ]))
    }

    fn file_sort_order() -> LexOrdering {
        LexOrdering::new(vec![asc(0, "timestamp")]).unwrap()
    }

    /// A file carrying its synthetic file-id partition value, its Delta
    /// partition values, and exact timestamp min/max statistics (index 0 of the
    /// parquet table schema).
    fn file(
        url: &str,
        part: ScalarValue,
        ts_min: i64,
        ts_max: i64,
        ts_nulls: usize,
    ) -> PartitionedFile {
        let mut file = PartitionedFile::new(url.to_string(), 100);
        file.partition_values = vec![wrap_file_id_value(url)];
        file.statistics = Some(Arc::new(Statistics {
            num_rows: Precision::Exact(10),
            total_byte_size: Precision::Exact(100),
            column_statistics: vec![
                ColumnStatistics {
                    null_count: Precision::Exact(ts_nulls),
                    min_value: Precision::Exact(ScalarValue::Int64(Some(ts_min))),
                    max_value: Precision::Exact(ScalarValue::Int64(Some(ts_max))),
                    ..Default::default()
                },
                ColumnStatistics::default(),
                ColumnStatistics::default(),
            ],
        }));
        file.extensions
            .insert(DeltaPartitionValues(vec![part].into()));
        file
    }

    fn utf8(part: &str) -> ScalarValue {
        ScalarValue::Utf8(Some(part.to_string()))
    }

    struct Fixture {
        files: Vec<PartitionedFile>,
    }

    impl Fixture {
        /// `specs` is `(url, part, ts_min, ts_max)`.
        fn new(specs: &[(&str, ScalarValue, i64, i64)]) -> Self {
            Self {
                files: specs
                    .iter()
                    .map(|(url, part, ts_min, ts_max)| file(url, part.clone(), *ts_min, *ts_max, 0))
                    .collect(),
            }
        }

        fn plan(
            &self,
            order: &[PhysicalSortExpr],
            file_sort_order: Option<&LexOrdering>,
            target_groups: usize,
        ) -> Option<SortPushdownPlan> {
            let order = LexOrdering::new(order.to_vec()).unwrap();
            let shape = analyze_order(
                &order,
                &output_schema(),
                &parquet_read_schema(),
                &["part".to_string()],
                file_sort_order,
            )?;
            let files: Vec<&PartitionedFile> = self.files.iter().collect();
            plan_sort_pushdown(&shape, &parquet_read_schema(), &files, target_groups).unwrap()
        }
    }

    fn group_urls(group: &FileGroup) -> Vec<String> {
        group
            .iter()
            .map(|f| f.object_meta.location.to_string())
            .collect()
    }

    #[test]
    fn non_overlapping_within_partition_forms_one_group_per_file() {
        let fixture = Fixture::new(&[
            ("a2", utf8("A"), 200, 299),
            ("a0", utf8("A"), 0, 99),
            ("b0", utf8("B"), 50, 149),
            ("a1", utf8("A"), 100, 199),
        ]);
        let order = vec![asc(0, "part"), asc(1, "timestamp")];

        let plan = fixture
            .plan(&order, Some(&file_sort_order()), 4)
            .expect("expected a pushdown plan");

        assert_eq!(plan.file_groups.len(), 4);
        let flat: Vec<_> = plan.file_groups.iter().flat_map(group_urls).collect();
        assert_eq!(flat, vec!["a0", "a1", "a2", "b0"]);

        // Every group is single-partition and its `part` stats are exact.
        for (idx, expected) in ["A", "A", "A", "B"].iter().enumerate() {
            let stat = &plan.per_partition_stats[idx]["part"];
            assert_eq!(stat.null_count, Precision::Exact(0));
            assert_eq!(
                stat.min_value,
                Precision::Exact(ScalarValue::Utf8(Some((*expected).to_string())))
            );
            assert_eq!(stat.min_value, stat.max_value);
        }
    }

    #[test]
    fn groups_never_span_a_partition_boundary() {
        let fixture = Fixture::new(&[
            ("a0", utf8("A"), 0, 99),
            ("a1", utf8("A"), 100, 199),
            ("a2", utf8("A"), 200, 299),
            ("b0", utf8("B"), 50, 149),
        ]);
        let order = vec![asc(0, "part"), asc(1, "timestamp")];

        // Fewer target groups than files: buckets are split but a group must
        // still not mix partition values.
        let plan = fixture
            .plan(&order, Some(&file_sort_order()), 2)
            .expect("expected a pushdown plan");

        for (group, stats) in plan.file_groups.iter().zip(&plan.per_partition_stats) {
            let parts: std::collections::HashSet<_> = group
                .iter()
                .map(|f| f.extensions.get::<DeltaPartitionValues>().unwrap().0[0].clone())
                .collect();
            assert_eq!(parts.len(), 1, "a group spans two partition values");
            // A single-partition group reports that value as an exact point range.
            let stat = &stats["part"];
            assert_eq!(stat.min_value, stat.max_value);
        }
    }

    #[test]
    fn overlapping_files_within_a_partition_are_not_supported() {
        let fixture = Fixture::new(&[
            ("a0", utf8("A"), 0, 150),
            ("a1", utf8("A"), 100, 199),
            ("b0", utf8("B"), 0, 99),
        ]);
        let order = vec![asc(0, "part"), asc(1, "timestamp")];

        assert!(fixture.plan(&order, Some(&file_sort_order()), 4).is_none());
    }

    #[test]
    fn incomparable_partition_values_are_rejected_without_panicking() {
        // Mismatched `ScalarValue` variants for one partition column are
        // incomparable: no regrouping can order them, so the pushdown must be
        // refused - and refused *before* the sort. Sorting mixed-variant keys
        // with a comparator that collapses incomparable pairs to `Equal` is
        // not a total order, which rustc's sort implementation detects and
        // panics on for inputs large enough to leave the insertion-sort path
        // (this exact sequence did).
        let parts = [
            utf8("p034774"),
            ScalarValue::Int64(Some(44153)),
            ScalarValue::Int64(Some(41196)),
            ScalarValue::Int64(Some(92870)),
            utf8("p011034"),
            utf8("p039795"),
            utf8("p067130"),
            ScalarValue::Int64(Some(86902)),
            utf8("p098089"),
            utf8("p046746"),
            ScalarValue::Int64(Some(20123)),
            ScalarValue::Int64(Some(30802)),
            ScalarValue::Int64(Some(47452)),
            ScalarValue::Int64(Some(58400)),
            utf8("p087034"),
            ScalarValue::Int64(Some(44812)),
            utf8("p084890"),
            utf8("p070495"),
            utf8("p093332"),
            ScalarValue::Int64(Some(29365)),
            ScalarValue::Int64(Some(67627)),
        ];
        let specs: Vec<(String, ScalarValue, i64, i64)> = parts
            .into_iter()
            .enumerate()
            .map(|(i, part)| {
                let i = i as i64;
                (format!("f{i:02}"), part, i * 100, i * 100 + 99)
            })
            .collect();
        let specs: Vec<(&str, ScalarValue, i64, i64)> = specs
            .iter()
            .map(|(url, part, lo, hi)| (url.as_str(), part.clone(), *lo, *hi))
            .collect();
        let fixture = Fixture::new(&specs);
        let order = vec![asc(0, "part"), asc(1, "timestamp")];

        assert!(fixture.plan(&order, Some(&file_sort_order()), 4).is_none());
    }

    #[test]
    fn nulls_in_a_multi_file_bucket_suffix_are_not_supported() {
        let mut fixture = Fixture::new(&[("a0", utf8("A"), 0, 99), ("a1", utf8("A"), 100, 199)]);
        // Give the second file a nullable-looking timestamp column with nulls.
        fixture.files[1] = file("a1", utf8("A"), 100, 199, 3);
        let order = vec![asc(0, "part"), asc(1, "timestamp")];

        assert!(fixture.plan(&order, Some(&file_sort_order()), 4).is_none());
    }

    #[test]
    fn leading_data_column_is_not_supported() {
        let fixture = Fixture::new(&[("a0", utf8("A"), 0, 99), ("b0", utf8("B"), 0, 99)]);
        let order = vec![asc(1, "timestamp"), asc(0, "part")];

        assert!(fixture.plan(&order, Some(&file_sort_order()), 4).is_none());
    }

    #[test]
    fn suffix_must_match_the_declared_file_sort_order() {
        let fixture = Fixture::new(&[("a0", utf8("A"), 0, 99), ("b0", utf8("B"), 0, 99)]);
        // `value` is not the declared file sort column.
        let order = vec![asc(0, "part"), asc(2, "value")];

        assert!(fixture.plan(&order, Some(&file_sort_order()), 4).is_none());
    }

    #[test]
    fn partition_only_ordering_needs_no_file_sort_order() {
        let fixture = Fixture::new(&[
            ("b1", utf8("B"), 0, 99),
            ("a0", utf8("A"), 0, 99),
            ("b0", utf8("B"), 0, 99),
        ]);
        let order = vec![asc(0, "part")];

        let plan = fixture
            .plan(&order, None, 4)
            .expect("partition-only ordering should be satisfiable");

        let flat: Vec<_> = plan.file_groups.iter().flat_map(group_urls).collect();
        assert_eq!(flat[0], "a0");
        assert!(flat[1..].iter().all(|url| url.starts_with('b')));
    }

    /// Natural minimum and maximum of one partition column over the files a
    /// group actually holds.
    fn actual_range(group: &FileGroup, position: usize) -> (ScalarValue, ScalarValue) {
        let mut values: Vec<ScalarValue> = group
            .iter()
            .map(|f| f.extensions.get::<DeltaPartitionValues>().unwrap().0[position].clone())
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).expect("comparable partition values"));
        (
            values.first().unwrap().clone(),
            values.last().unwrap().clone(),
        )
    }

    /// Assert every prefix column's published statistics are the natural
    /// extrema of the partition values the group actually holds.
    fn assert_stats_match_files(plan: &SortPushdownPlan, columns: &[&str]) {
        for (group, stats) in plan.file_groups.iter().zip(&plan.per_partition_stats) {
            for (position, name) in columns.iter().enumerate() {
                let (min, max) = actual_range(group, position);
                let stat = &stats[*name];
                assert_eq!(stat.min_value, Precision::Exact(min), "{name} min_value");
                assert_eq!(stat.max_value, Precision::Exact(max), "{name} max_value");
            }
        }
    }

    /// A group's keys are its range endpoints under the requested ordering, so
    /// for a descending column the first key is the group's *maximum*. Copying
    /// them into `min_value`/`max_value` as-is publishes an inverted range as
    /// `Precision::Exact` — `ProgressiveEvalRule` reads it back as
    /// `(start, end) = (max_value, min_value)` and gets the group's endpoints
    /// the wrong way round, and any other consumer sees `min > max`.
    #[test]
    fn descending_partition_prefix_statistics_are_not_inverted() {
        let fixture = Fixture::new(&[
            ("a0", utf8("A"), 0, 99),
            ("b0", utf8("B"), 0, 99),
            ("c0", utf8("C"), 0, 99),
        ]);
        // A descending prefix with an ascending suffix also pins that buckets
        // are emitted in reverse and that mixed sort directions are accepted.
        let order = vec![desc(0, "part"), asc(1, "timestamp")];

        // More buckets than groups, so the leading two are packed into one run
        // that spans `C` … `B`.
        let plan = fixture
            .plan(&order, Some(&file_sort_order()), 2)
            .expect("expected a pushdown plan");

        assert_eq!(
            plan.file_groups.iter().map(group_urls).collect::<Vec<_>>(),
            vec![vec!["c0", "b0"], vec!["a0"]],
        );
        assert_stats_match_files(&plan, &["part"]);

        // How `ProgressiveEvalRule` decodes a descending column: the endpoints
        // must come back in the order the group's files are read.
        let stat = &plan.per_partition_stats[0]["part"];
        assert_eq!(stat.max_value, Precision::Exact(utf8("C")), "range start");
        assert_eq!(stat.min_value, Precision::Exact(utf8("B")), "range end");
    }

    fn two_column_output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("part", DataType::Utf8, false),
            Field::new("sub", DataType::Int64, false),
            Field::new("timestamp", DataType::Int64, true),
            Field::new("value", DataType::Int64, true),
        ]))
    }

    fn two_column_file(url: &str, part: &str, sub: i64) -> PartitionedFile {
        let mut file = file(url, utf8(part), 0, 99, 0);
        file.extensions.insert(DeltaPartitionValues(
            vec![utf8(part), ScalarValue::Int64(Some(sub))].into(),
        ));
        file
    }

    fn plan_two_column(
        files: Vec<PartitionedFile>,
        order: &[PhysicalSortExpr],
        target_groups: usize,
    ) -> Option<SortPushdownPlan> {
        let order = LexOrdering::new(order.to_vec()).unwrap();
        let shape = analyze_order(
            &order,
            &two_column_output_schema(),
            &parquet_read_schema(),
            &["part".to_string(), "sub".to_string()],
            None,
        )?;
        let files: Vec<&PartitionedFile> = files.iter().collect();
        plan_sort_pushdown(&shape, &parquet_read_schema(), &files, target_groups).unwrap()
    }

    fn ordered_bucket(specs: &[(&str, i64, i64)], sort: PhysicalSortExpr) -> Option<Vec<String>> {
        let files: Vec<PartitionedFile> = specs
            .iter()
            .map(|(url, min, max)| file(url, utf8("A"), *min, *max, 0))
            .collect();
        let ordering = LexOrdering::new(vec![sort]).unwrap();
        Some(
            non_overlapping_file_order(files, |file| file, &ordering)
                .ok()?
                .into_iter()
                .map(|file| file.object_meta.location.to_string())
                .collect(),
        )
    }

    /// Disjoint ranges are accepted and returned in ascending order of their
    /// minimum, whatever order they arrived in.
    #[test]
    fn test_order_bucket_sorts_disjoint_files() {
        assert_eq!(
            ordered_bucket(
                &[("c", 20, 29), ("a", 0, 9), ("b", 10, 19)],
                asc(0, "timestamp")
            ),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );
    }

    /// A descending ordering reverses the arrangement, matching what the row
    /// encoding in `split_groups_by_statistics` would produce.
    #[test]
    fn test_order_bucket_reverses_for_a_descending_ordering() {
        assert_eq!(
            ordered_bucket(
                &[("a", 0, 9), ("c", 20, 29), ("b", 10, 19)],
                desc(0, "timestamp")
            ),
            Some(vec!["c".to_string(), "b".to_string(), "a".to_string()])
        );
    }

    /// Overlapping ranges are refused.
    #[test]
    fn test_order_bucket_refuses_overlapping_files() {
        assert_eq!(
            ordered_bucket(&[("a", 0, 15), ("b", 10, 19)], asc(0, "timestamp")),
            None
        );
    }

    /// Ranges that merely touch are accepted: every row of the earlier file
    /// is at or below the shared value and every row of the later one at or
    /// above it, so the concatenation is still non-decreasing.
    #[test]
    fn test_order_bucket_accepts_touching_files() {
        assert_eq!(
            ordered_bucket(&[("b", 10, 19), ("a", 0, 10)], asc(0, "timestamp")),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            ordered_bucket(&[("a", 0, 10), ("b", 10, 19)], desc(0, "timestamp")),
            Some(vec!["b".to_string(), "a".to_string()])
        );
    }

    /// Under a descending ordering a file starts at its maximum and ends at
    /// its minimum, so overlap is judged between the previous minimum and the
    /// next maximum. Comparing the raw extrema instead would accept
    /// `[50, 100]` followed by `[40, 60]` as ordered.
    #[test]
    fn test_order_bucket_refuses_overlapping_files_for_a_descending_ordering() {
        assert_eq!(
            ordered_bucket(&[("a", 50, 100), ("b", 40, 60)], desc(0, "timestamp")),
            None
        );
        assert_eq!(
            ordered_bucket(&[("a", 50, 100), ("b", 40, 49)], desc(0, "timestamp")),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    /// A file with no statistics at all cannot be placed.
    #[test]
    fn test_order_bucket_refuses_without_statistics() {
        let mut files = vec![
            file("a", utf8("A"), 0, 9, 0),
            file("b", utf8("A"), 10, 19, 0),
        ];
        files[1].statistics = None;
        let ordering = LexOrdering::new(vec![asc(0, "timestamp")]).unwrap();
        assert!(non_overlapping_file_order(files, |file| file, &ordering).is_err());
    }

    /// Statistics whose `ScalarValue` variant differs between files are
    /// incomparable; the bucket is refused before the sort, which would
    /// otherwise see a comparator that is not a total order (see
    /// `incomparable_partition_values_are_rejected_without_panicking`).
    #[test]
    fn test_order_bucket_refuses_incomparable_statistics() {
        let files: Vec<PartitionedFile> = (0..40)
            .map(|i| {
                let mut file = file(&format!("f{i}"), utf8("A"), i * 10, i * 10 + 9, 0);
                if i % 3 == 1 {
                    let stats = Arc::make_mut(file.statistics.as_mut().unwrap());
                    stats.column_statistics[0].min_value =
                        Precision::Exact(utf8(&format!("{:03}", i * 10)));
                    stats.column_statistics[0].max_value =
                        Precision::Exact(utf8(&format!("{:03}", i * 10 + 9)));
                }
                file
            })
            .collect();
        let ordering = LexOrdering::new(vec![asc(0, "timestamp")]).unwrap();
        assert!(non_overlapping_file_order(files, |file| file, &ordering).is_err());
    }

    fn three_two_column_files() -> Vec<PartitionedFile> {
        vec![
            two_column_file("a1", "A", 1),
            two_column_file("a2", "A", 2),
            two_column_file("b0", "B", 0),
        ]
    }

    /// A run that has a neighbour may only span several keys in the *last*
    /// prefix column. A run from `[A, 2]` to `[B, 0]` is properly ordered, but
    /// its endpoints are not the extrema of anything: `sub` really covers
    /// `0..2` there, so no `ColumnStatistics` can describe both the range and
    /// where it starts - and that is what proves the groups disjoint.
    #[test]
    fn packed_runs_do_not_span_a_change_in_a_leading_prefix_column() {
        let order = vec![asc(0, "part"), asc(1, "sub")];
        let plan =
            plan_two_column(three_two_column_files(), &order, 2).expect("expected a pushdown plan");

        assert_eq!(
            plan.file_groups.iter().map(group_urls).collect::<Vec<_>>(),
            vec![vec!["a1", "a2"], vec!["b0"]],
        );
        assert_stats_match_files(&plan, &["part", "sub"]);
    }

    /// A single target group cannot absorb the extra groups that leading-prefix
    /// cuts produce, so the buckets are packed across every prefix column -
    /// which is the only way a single-partition scan can answer a multi-column
    /// partition prefix without a sort. The group publishes no prefix
    /// statistics: `DeltaScanExec` describes a single execution partition from
    /// the exact table-wide aggregate instead.
    #[test]
    fn a_single_target_group_spans_every_prefix_column() {
        let order = vec![asc(0, "part"), asc(1, "sub")];
        let plan =
            plan_two_column(three_two_column_files(), &order, 1).expect("expected a pushdown plan");

        assert_eq!(
            plan.file_groups.iter().map(group_urls).collect::<Vec<_>>(),
            vec![vec!["a1", "a2", "b0"]],
        );
        assert_eq!(plan.per_partition_stats, vec![HashMap::new()]);
    }

    /// One record batch per (partition, day range), each sorted by timestamp.
    /// Timestamp ranges never overlap within a partition but do across
    /// partitions, so `ORDER BY part, timestamp` can only be answered without a
    /// `SortExec` by regrouping the files per partition value.
    fn write_batch(part: &str, start: i64, len: i64) -> RecordBatch {
        use arrow_array::{Int64Array, StringArray, TimestampMicrosecondArray};
        use arrow_schema::TimeUnit;

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("value", DataType::Int64, false),
            Field::new("part", DataType::Utf8, false),
        ]));
        let timestamps: ArrayRef = Arc::new(TimestampMicrosecondArray::from(
            (start..start + len)
                .map(|s| s * 1_000_000)
                .collect::<Vec<_>>(),
        ));
        let values: ArrayRef = Arc::new(Int64Array::from((start..start + len).collect::<Vec<_>>()));
        let parts: ArrayRef = Arc::new(StringArray::from(vec![part; len as usize]));
        RecordBatch::try_new(schema, vec![timestamps, values, parts]).unwrap()
    }

    /// Like [`write_batch`], but with a nullable timestamp column that
    /// optionally ends on a null.
    fn nullable_write_batch(part: &str, start: i64, len: i64, trailing_null: bool) -> RecordBatch {
        use arrow_array::{Int64Array, StringArray, TimestampMicrosecondArray};
        use arrow_schema::TimeUnit;

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
            Field::new("value", DataType::Int64, false),
            Field::new("part", DataType::Utf8, false),
        ]));
        let mut timestamps: Vec<Option<i64>> =
            (start..start + len).map(|s| Some(s * 1_000_000)).collect();
        if trailing_null {
            timestamps.push(None);
        }
        let rows = timestamps.len();
        let values: ArrayRef = Arc::new(Int64Array::from(
            (start..start + rows as i64).collect::<Vec<_>>(),
        ));
        let parts: ArrayRef = Arc::new(StringArray::from(vec![part; rows]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(TimestampMicrosecondArray::from(timestamps)) as ArrayRef,
                values,
                parts,
            ],
        )
        .unwrap()
    }

    /// Offer `columns`, ascending, to `scan` and return the regrouped exec.
    fn exact_pushdown(
        scan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        columns: &[&str],
    ) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
        use datafusion::physical_plan::SortOrderPushdownResult;

        let schema = scan.schema();
        let order: Vec<PhysicalSortExpr> = columns
            .iter()
            .map(|name| asc(schema.index_of(name).unwrap(), name))
            .collect();
        match scan.try_pushdown_sort(&order).unwrap() {
            SortOrderPushdownResult::Exact { inner } => inner,
            other => panic!("expected an exact pushdown, got {other:?}"),
        }
    }

    /// Four files across two partition values: every file sorted by timestamp,
    /// non-overlapping within a partition, overlapping across partitions — so
    /// `ORDER BY part, timestamp` is only answerable without a sort by
    /// regrouping the files per partition value.
    async fn partitioned_sorted_table() -> crate::DeltaTable {
        use crate::protocol::SaveMode;

        let mut table = crate::DeltaTable::new_in_memory()
            .write(vec![write_batch("A", 0, 100)])
            .with_partition_columns(vec!["part"])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap();
        for (part, start) in [("A", 100), ("A", 200), ("B", 50)] {
            table = table
                .write(vec![write_batch(part, start, 100)])
                .with_save_mode(SaveMode::Append)
                .await
                .unwrap();
        }
        table
    }

    /// The regrouped scan is assembled field by field rather than cloned from
    /// the original, so every setting the original carried has to be copied
    /// across by hand. Losing the expression adapter would quietly break the
    /// schema adaptation of filters and projections pushed in afterwards.
    #[tokio::test]
    async fn pushdown_carries_the_scan_settings_across() {
        use datafusion::physical_plan::ExecutionPlan;
        use datafusion_datasource::file_scan_config::FileScanConfig;
        use datafusion_datasource::source::DataSourceExec;

        use crate::delta_datafusion::{FileSortColumn, create_session};

        fn file_scan(plan: &Arc<dyn ExecutionPlan>) -> &FileScanConfig {
            plan.downcast_ref::<DataSourceExec>()
                .expect("a parquet scan")
                .data_source()
                .as_ref()
                .downcast_ref::<FileScanConfig>()
                .expect("a file scan config")
        }

        let table = partitioned_sorted_table().await;
        let ctx = create_session().into_inner();
        let provider = table
            .table_provider()
            .with_file_sort_order([FileSortColumn::asc("timestamp")])
            .await
            .unwrap();
        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let before = file_scan(scan.children()[0]).clone();

        let pushed = exact_pushdown(&scan, &["part", "timestamp"]);
        let after = file_scan(pushed.children()[0]);

        assert_eq!(after.object_store_url, before.object_store_url);
        assert!(Arc::ptr_eq(after.file_source(), before.file_source()));
        assert_eq!(after.limit, before.limit);
        assert_eq!(after.constraints, before.constraints);
        assert_eq!(after.file_compression_type, before.file_compression_type);
        assert_eq!(after.batch_size, before.batch_size);
        assert_eq!(
            after.expr_adapter_factory.is_some(),
            before.expr_adapter_factory.is_some(),
            "the expression adapter must survive the regrouping"
        );
        assert!(
            before.expr_adapter_factory.is_some(),
            "fixture is meaningless without one"
        );
        assert_eq!(
            after.partitioned_by_file_group,
            before.partitioned_by_file_group
        );
    }

    /// An ordering that does not start with partition columns is offered to the
    /// parquet scan instead of being refused outright: this exec keeps rows in
    /// the order they arrive, so the trait asks it to delegate and re-wrap.
    /// DataFusion answers `Inexact` for a reversed request - the `SortExec`
    /// stays, but the scan now reads each file's row groups from the end the
    /// query wants first.
    #[tokio::test]
    async fn ordering_without_a_partition_prefix_is_delegated_to_the_scan() {
        use datafusion::physical_plan::SortOrderPushdownResult;

        use crate::delta_datafusion::{FileSortColumn, create_session};

        let table = partitioned_sorted_table().await;
        let ctx = create_session().into_inner();
        let provider = table
            .table_provider()
            .with_file_sort_order([FileSortColumn::asc("timestamp")])
            .await
            .unwrap();
        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let schema = scan.schema();
        let descending = PhysicalSortExpr::new(
            Arc::new(Column::new(
                "timestamp",
                schema.index_of("timestamp").unwrap(),
            )),
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        );

        assert!(matches!(
            scan.try_pushdown_sort(&[descending]).unwrap(),
            SortOrderPushdownResult::Inexact { .. }
        ));
    }

    /// Deletion vectors are applied by position within each file, so the
    /// reordering the scan does on the `Inexact` path would line the mask up
    /// against the wrong rows - wrong data, which no `SortExec` above can
    /// repair. Nothing is delegated while any file carries one.
    #[tokio::test]
    async fn a_deletion_vector_stops_the_delegation() {
        use datafusion::physical_plan::SortOrderPushdownResult;

        use crate::delta_datafusion::{FileSortColumn, create_session};

        let table_url =
            url::Url::from_directory_path(crate::test_utils::TestTables::WithDvSmall.as_path())
                .unwrap();
        let table = crate::open_table(table_url).await.unwrap();
        let ctx = create_session().into_inner();
        let provider = table
            .table_provider()
            .with_file_sort_order([FileSortColumn::asc("value")])
            .await
            .unwrap();
        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let schema = scan.schema();
        let descending = PhysicalSortExpr::new(
            Arc::new(Column::new("value", schema.index_of("value").unwrap())),
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        );

        assert!(matches!(
            scan.try_pushdown_sort(&[descending]).unwrap(),
            SortOrderPushdownResult::Unsupported
        ));
    }

    /// The regrouped scan must be marked order-sensitive.
    ///
    /// The file-to-partition assignment *is* the pushdown result, but the
    /// combined ordering cannot be declared on the child (partition columns
    /// are not in its schema), and an order-insensitive scan lets DataFusion
    /// pool every file into one queue for the sibling streams to share and
    /// lets a pushed fetch prune earlier row groups.
    #[tokio::test]
    async fn pushdown_marks_the_regrouped_scan_order_sensitive() {
        use datafusion_datasource::file_scan_config::FileScanConfig;
        use datafusion_datasource::source::DataSourceExec;

        use crate::delta_datafusion::{FileSortColumn, create_session};
        use crate::protocol::SaveMode;

        // A null in the sort column stops `get_read_plan` declaring a
        // store-level ordering across the multi-file groups it builds, so the
        // scan the pushdown inherits from is order-insensitive.
        let mut table = crate::DeltaTable::new_in_memory()
            .write(vec![nullable_write_batch("A", 0, 10, false)])
            .with_partition_columns(vec!["part"])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap();
        for (part, start) in [("B", 100), ("C", 200), ("D", 300)] {
            table = table
                .write(vec![nullable_write_batch(part, start, 10, part == "D")])
                .with_save_mode(SaveMode::Append)
                .await
                .unwrap();
        }

        let ctx = create_session().into_inner();
        ctx.sql("SET datafusion.execution.target_partitions = 2")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let provider = table
            .table_provider()
            .with_file_sort_order([FileSortColumn::asc("timestamp")])
            .await
            .unwrap();

        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let pushed = exact_pushdown(&scan, &["part", "timestamp"]);

        let child = Arc::clone(pushed.children()[0]);
        let file_scan = child
            .downcast_ref::<DataSourceExec>()
            .unwrap()
            .data_source()
            .as_ref()
            .downcast_ref::<FileScanConfig>()
            .unwrap();

        assert_eq!(file_scan.file_groups.len(), 2);
        assert!(
            file_scan.output_ordering.is_empty(),
            "the child declares no ordering of its own, so nothing else makes it order-sensitive"
        );
        assert!(
            file_scan.preserve_order,
            "the regrouped scan must be order-sensitive"
        );
    }

    /// Files of one partition share a single interned tuple of partition
    /// values rather than each carrying a copy.
    #[tokio::test]
    async fn files_of_a_partition_share_their_partition_values() {
        use datafusion_datasource::file_scan_config::FileScanConfig;
        use datafusion_datasource::source::DataSourceExec;

        use crate::delta_datafusion::create_session;

        let table = partitioned_sorted_table().await;
        let ctx = create_session().into_inner();
        let provider = table.table_provider().await.unwrap();
        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let file_scan = scan.children()[0]
            .downcast_ref::<DataSourceExec>()
            .unwrap()
            .data_source()
            .as_ref()
            .downcast_ref::<FileScanConfig>()
            .unwrap();

        let mut by_partition: HashMap<String, Vec<&Arc<[ScalarValue]>>> = HashMap::new();
        for file in file_scan.file_groups.iter().flat_map(|group| group.iter()) {
            let values = &file.extensions.get::<DeltaPartitionValues>().unwrap().0;
            by_partition
                .entry(values[0].to_string())
                .or_default()
                .push(values);
        }
        assert!(
            by_partition.values().any(|files| files.len() >= 2),
            "the fixture should hold several files of one partition"
        );
        for files in by_partition.values() {
            assert!(files.iter().all(|values| Arc::ptr_eq(values, files[0])));
        }
    }

    /// Sort-pushdown state is only valid for the file grouping it was computed
    /// for, and `per_partition_stats` is indexed by execution partition. The
    /// `SortExec` is already gone by the time a child is swapped, so a child
    /// with another grouping is refused outright rather than the ordering
    /// being silently dropped; the same child, or one over the same groups,
    /// is accepted and keeps the claim.
    #[tokio::test]
    async fn with_new_children_refuses_a_changed_grouping() {
        use datafusion::physical_expr::Partitioning;
        use datafusion::physical_plan::repartition::RepartitionExec;
        use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};

        use crate::delta_datafusion::{FileSortColumn, create_session};

        let table = partitioned_sorted_table().await;
        let ctx = create_session().into_inner();
        let provider = table
            .table_provider()
            .with_file_sort_order([FileSortColumn::asc("timestamp")])
            .await
            .unwrap();

        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let pushed = exact_pushdown(&scan, &["part", "timestamp"]);

        let leads_with_part = |plan: &Arc<dyn ExecutionPlan>| {
            plan.properties()
                .equivalence_properties()
                .oeq_class()
                .iter()
                .any(|ordering| {
                    ordering
                        .iter()
                        .next()
                        .and_then(|sort_expr| sort_expr.expr.downcast_ref::<Column>())
                        .is_some_and(|col| col.name() == "part")
                })
        };
        assert!(
            leads_with_part(&pushed),
            "a successful pushdown should advertise a `part`-leading ordering"
        );

        let child = Arc::clone(pushed.children()[0]);
        let same = Arc::clone(&pushed)
            .with_new_children(vec![Arc::clone(&child)])
            .unwrap();
        assert!(
            leads_with_part(&same),
            "the same child should keep the pushed ordering"
        );

        let repartitioned = Arc::new(
            RepartitionExec::try_new(
                child,
                Partitioning::RoundRobinBatch(pushed.output_partitioning().partition_count() + 1),
            )
            .unwrap(),
        );
        let error = pushed
            .with_new_children(vec![repartitioned])
            .expect_err("a child with another grouping must be refused")
            .to_string();
        assert!(
            error.contains("pushed-down sort"),
            "unexpected error message: {error}"
        );
    }

    /// `repartitioned` swaps in a re-split input; the cached plan properties
    /// were computed from the old input and must be rebuilt, or the exec
    /// advertises a stale partition count and parents execute too few
    /// partitions.
    #[tokio::test]
    async fn repartitioned_recomputes_partitioning() {
        use datafusion::config::ConfigOptions;
        use datafusion::physical_plan::ExecutionPlanProperties;

        use crate::delta_datafusion::create_session;

        let table = partitioned_sorted_table().await;
        let ctx = create_session().into_inner();
        // No declared file sort order: an ordered scan refuses re-splitting.
        let provider = table.table_provider().await.unwrap();
        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let before = scan.output_partitioning().partition_count();

        let mut config = ConfigOptions::new();
        // The test files are tiny; split on any size.
        config.optimizer.repartition_file_min_size = 1;
        let repartitioned = scan
            .repartitioned(before + 4, &config)
            .unwrap()
            .expect("expected the scan to accept a repartition");

        let child_partitions = repartitioned.children()[0]
            .output_partitioning()
            .partition_count();
        assert_ne!(
            before, child_partitions,
            "the input did not change its partitioning; the test is inert"
        );
        assert_eq!(
            repartitioned.output_partitioning().partition_count(),
            child_partitions,
            "stale cached partitioning survived a repartition"
        );
    }

    /// A deletion vector's keep mask is consumed from the front of one shared
    /// sequence per file, so byte-range pieces of that file read on different
    /// streams would apply each other's positions. A scan with one refuses to
    /// be re-split.
    #[tokio::test]
    async fn a_deletion_vector_stops_the_repartition() {
        use datafusion::config::ConfigOptions;

        use crate::delta_datafusion::create_session;

        let table_url =
            url::Url::from_directory_path(crate::test_utils::TestTables::WithDvSmall.as_path())
                .unwrap();
        let table = crate::open_table(table_url).await.unwrap();
        let ctx = create_session().into_inner();
        let provider = table.table_provider().await.unwrap();
        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();

        let mut config = ConfigOptions::new();
        // The test file is tiny; split on any size.
        config.optimizer.repartition_file_min_size = 1;
        assert!(
            scan.children()[0]
                .repartitioned(4, &config)
                .unwrap()
                .is_some(),
            "the parquet scan would split the file; the test is inert"
        );
        assert!(
            scan.repartitioned(4, &config).unwrap().is_none(),
            "a scan with a deletion vector must not be byte-range split"
        );
    }

    /// Without pushed-down per-partition statistics, a request for one
    /// execution partition's statistics falls back to the table-wide
    /// aggregated partition-column stats. Those are valid bounds for the
    /// partition but must not claim exactness.
    #[tokio::test]
    async fn per_partition_statistics_fall_back_to_inexact_aggregates() {
        use datafusion::config::ConfigOptions;
        use datafusion::physical_plan::ExecutionPlanProperties;

        use crate::delta_datafusion::create_session;

        let table = partitioned_sorted_table().await;
        let ctx = create_session().into_inner();
        let provider = table.table_provider().await.unwrap();
        let scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();

        let mut config = ConfigOptions::new();
        config.optimizer.repartition_file_min_size = 1;
        let scan = scan
            .repartitioned(4, &config)
            .unwrap()
            .expect("expected the scan to accept a repartition");
        assert!(scan.output_partitioning().partition_count() > 1);
        let part_index = scan.schema().index_of("part").unwrap();

        let whole_scan = scan.partition_statistics(None).unwrap();
        assert!(
            matches!(
                whole_scan.column_statistics[part_index].min_value,
                Precision::Exact(_)
            ),
            "table-wide partition-column statistics should stay exact"
        );

        let one_partition = scan.partition_statistics(Some(0)).unwrap();
        for stat in [
            &one_partition.column_statistics[part_index].min_value,
            &one_partition.column_statistics[part_index].max_value,
        ] {
            assert!(
                !matches!(stat, Precision::Exact(_)),
                "table-wide aggregate published as exact for one partition: {stat:?}"
            );
        }
        assert!(
            !matches!(
                one_partition.column_statistics[part_index].null_count,
                Precision::Exact(_)
            ),
            "table-wide null count published as exact for one partition"
        );
    }

    /// Leading-prefix cuts may exceed the target group count, but only up to
    /// `group_budget`. Past that the buckets are packed by file count instead:
    /// the sort is still removed, and the groups that span a change in the
    /// leading column publish no statistics, so the merge above them stays.
    #[test]
    fn leading_prefix_cuts_fall_back_to_packing_past_the_budget() {
        let files = |parts: usize| -> Vec<PartitionedFile> {
            (0..parts)
                .map(|i| two_column_file(&format!("f{i:03}"), &format!("p{i:03}"), 0))
                .collect()
        };
        let order = vec![asc(0, "part"), asc(1, "sub")];

        // 60 distinct leading values overshoot a target of 2 but stay within
        // group_budget(2) = 64, so every one gets its own group.
        let plan = plan_two_column(files(60), &order, 2).expect("expected a pushdown plan");
        assert_eq!(plan.file_groups.len(), 60);
        assert!(
            plan.per_partition_stats
                .iter()
                .all(|stats| !stats.is_empty())
        );

        // 65 distinct leading values exceed the budget: two packed groups, each
        // spanning many leading values, with no statistics to prove them apart.
        let plan = plan_two_column(files(65), &order, 2).expect("expected a pushdown plan");
        assert_eq!(plan.file_groups.len(), 2);
        assert_eq!(
            plan.file_groups.iter().map(FileGroup::len).sum::<usize>(),
            65
        );
        assert!(plan.per_partition_stats.iter().all(HashMap::is_empty));
    }

    /// A bucket larger than the file-id dictionary splits into several
    /// groups; each one must count against the group budget, or the remaining
    /// buckets are packed as if the budget were still free and the scan ends
    /// up with more partitions than requested.
    #[test]
    fn dictionary_split_charges_the_group_budget_per_emitted_group() {
        let light_file = |url: String| PartitionedFile::new(url, 1);
        let mut buckets = vec![Bucket {
            key: vec![utf8("A")],
            files: (0..2 * MAX_PARTITION_DICT_CARDINALITY + 1)
                .map(|i| light_file(format!("a{i}")))
                .collect(),
        }];
        for i in 0..10 {
            buckets.push(Bucket {
                key: vec![utf8(&format!("b{i}"))],
                files: (0..100).map(|j| light_file(format!("b{i}-{j}"))).collect(),
            });
        }

        let groups = chunk_buckets(buckets, 4).expect("within the group budget");

        // The oversized bucket takes three groups on its own, leaving one
        // group for everything else.
        assert_eq!(
            groups.len(),
            4,
            "sizes: {:?}",
            groups.iter().map(|g| group_files(g)).collect::<Vec<_>>()
        );
    }

    /// Number of files in a group of buckets.
    fn group_files(group: &[Bucket]) -> usize {
        group.iter().map(|bucket| bucket.files.len()).sum()
    }

    #[test]
    fn chunk_buckets_splits_when_there_are_fewer_buckets_than_groups() {
        let bucket = |name: char, n: i64| Bucket {
            key: vec![utf8(&name.to_string())],
            files: (0..n)
                .map(|i| file(&format!("{name}{i}"), utf8(&name.to_string()), i, i, 0))
                .collect(),
        };
        let groups = chunk_buckets(vec![bucket('a', 6), bucket('b', 2)], 4)
            .expect("within the group budget");

        // Every group's files come from a single bucket.
        for group in &groups {
            assert_eq!(group.len(), 1);
            let prefixes: std::collections::HashSet<char> = group[0]
                .files
                .iter()
                .map(|f| f.object_meta.location.to_string().chars().next().unwrap())
                .collect();
            assert_eq!(prefixes.len(), 1);
        }
        assert_eq!(groups.iter().map(|g| group_files(g)).sum::<usize>(), 8);
        assert!(groups.len() >= 2, "each bucket gets at least one group");
    }

    /// With more buckets than groups, whole buckets are packed together so a
    /// query selecting many partitions does not explode the scan's partition
    /// count. Groups stay bucket-aligned, so consecutive groups still take a
    /// strict step on the partition prefix.
    #[test]
    fn chunk_buckets_packs_whole_buckets_when_they_outnumber_groups() {
        let buckets: Vec<Bucket> = (0..20)
            .map(|i| {
                let key = ScalarValue::Int64(Some(i));
                Bucket {
                    key: vec![key.clone()],
                    files: vec![file(&format!("f{i}"), key, i, i, 0)],
                }
            })
            .collect();

        let groups = chunk_buckets(buckets, 4).expect("within the group budget");

        assert_eq!(groups.len(), 4);
        assert_eq!(groups.iter().map(|g| group_files(g)).sum::<usize>(), 20);
        // Each group's key range is strictly after the previous group's.
        for pair in groups.windows(2) {
            assert!(
                pair[0].last().unwrap().key[0] < pair[1].first().unwrap().key[0],
                "group key ranges must take a strict step"
            );
        }
    }
}
