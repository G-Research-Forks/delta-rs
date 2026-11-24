//! Kernel-based Delta table scanning with optimized query execution.
//!
//! This module provides efficient table scanning using Delta Kernel, integrating with
//! DataFusion's query engine. It supports:
//!
//! - **Physical scan execution** ([`DeltaScanExec`]) - Reads Parquet data files and applies
//!   Delta protocol transformations (column mapping, deletion vectors, partition values)
//! - **Metadata-only scans** ([`DeltaScanMetaExec`]) - Answers queries like `COUNT(*)`
//!   using file statistics without reading data files
//! - **Predicate pushdown** - Pushes filters to both kernel file skipping and Parquet readers
//!   for efficient data pruning
//! - **Multi-store support** - Handles files across different object stores in a single query
//!
//! The scan planning process in [`plan`] determines which files to read and how to apply
//! predicates, while execution plans handle the actual data reading and transformation.

use std::cmp::Ordering;
use std::{
    collections::{HashSet, VecDeque},
    pin::Pin,
    sync::Arc,
};

use arrow::datatypes::UInt16Type;
use arrow_array::{
    ArrayRef, DictionaryArray, RecordBatch, StringArray, StringViewArray, UInt16Array,
};
use arrow_cast::{CastOptions, cast_with_options};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaBuilder, SchemaRef, SortOptions};
use dashmap::DashMap;
use datafusion::{
    catalog::Session,
    common::{
        ColumnStatistics, HashMap, Result, ScalarValue, Statistics, ToDFSchema,
        internal_datafusion_err, plan_err, stats::Precision, utils::compare_rows,
    },
    datasource::physical_plan::{
        ParquetSource,
        parquet::{CachedParquetFileReaderFactory, metadata::DFParquetMetadata},
    },
    error::DataFusionError,
    execution::{
        cache::{cache_manager::FileMetadataCache, default_cache::DefaultCache},
        object_store::ObjectStoreUrl,
    },
    physical_expr::{
        LexOrdering, Partitioning, PhysicalSortExpr, conjunction, expressions::Column,
    },
    physical_plan::{
        ExecutionPlan, PhysicalExpr,
        coalesce_partitions::CoalescePartitionsExec,
        empty::EmptyExec,
        metrics::{ExecutionPlanMetricsSet, Gauge, MetricBuilder},
        union::UnionExec,
    },
    prelude::Expr,
};
use datafusion_datasource::{
    PartitionedFile, TableSchema, compute_all_files_statistics,
    file::FileSource,
    file_groups::FileGroup,
    file_scan_config::{FileScanConfig, FileScanConfigBuilder},
    source::DataSourceExec,
};
use datafusion_physical_expr_adapter::{
    BatchAdapter, BatchAdapterFactory, PhysicalExprAdapterFactory,
};
use delta_kernel::{
    Engine, Expression, engine::arrow_data::ArrowEngineData, expressions::StructData,
    scan::ScanMetadata, table_features::TableFeature,
};
use deltalake_kernel_executor::io::storage::to_partitioned_file;
use deltalake_kernel_executor::query::scan::expr_adapter::{
    DeltaPhysicalExprAdapterFactory, relax_schema_nested_nullability,
};
use futures::{Stream, StreamExt as _, TryStreamExt as _, future::ready};
use itertools::Itertools as _;
use object_store::ObjectMeta;
use parquet::arrow::RowNumber;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData};
use tracing::debug;
use url::Url;

pub use self::exec::DeltaScanExec;
use self::exec::DvExecutionState;
use self::exec_meta::DeltaScanMetaExec;
pub(crate) use self::plan::{KernelScanPlan, ProjectedScanContract, supports_filters_pushdown};
use self::replay::{ScanFileContext, ScanFileStream};
pub(crate) use self::runtime_filter::RuntimeFileFilter;
use self::runtime_filter::RuntimeScanFilePruner;
use super::{FileSelection, ResolvedFileSelection};
use crate::{
    DeltaTableError,
    delta_datafusion::{
        DeltaScanConfig,
        engine::{AsObjectStoreUrl as _, to_datafusion_scalar},
        file_id::wrap_file_id_value,
        table_provider::next::DeletionVectorSelection,
    },
    kernel::{LogicalFileView, StructDataExt as _},
};

mod exec;
mod exec_meta;
mod plan;
mod replay;
mod runtime_filter;
mod sort_pushdown;

type ScanMetadataStream = Pin<Box<dyn Stream<Item = Result<ScanMetadata, DeltaTableError>> + Send>>;
type PublicFileIdMap = HashMap<String, String>;
type PhysicalFileIdentityMap = HashMap<String, PhysicalFileIdentity>;
const PHYSICAL_POSITION_COLUMN: &str = "__delta_rs_physical_position";

#[derive(Clone, Debug, PartialEq, Eq)]
struct PhysicalFileIdentity {
    object_store_url: ObjectStoreUrl,
    object_meta: ObjectMeta,
}

struct ReplayedScanFiles {
    files: Vec<ScanFileContext>,
    transforms: HashMap<String, Arc<Expression>>,
    dvs: HashMap<String, Vec<bool>>,
    public_file_ids: PublicFileIdMap,
    metrics: ExecutionPlanMetricsSet,
    /// The `count_files_scanned` metric in `metrics`
    files_scanned: Gauge,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn execution_plan(
    config: &DeltaScanConfig,
    session: &dyn Session,
    scan_plan: KernelScanPlan,
    stream: ScanMetadataStream,
    engine: Arc<dyn Engine>,
    limit: Option<usize>,
    file_selection: Option<&ResolvedFileSelection>,
    runtime_filter: Option<&RuntimeFileFilter>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(selection) = file_selection
        && selection.active_file_ids.is_empty()
    {
        return Ok(Arc::new(EmptyExec::new(
            scan_plan.contract.result_schema.clone(),
        )));
    }

    let replayed = replay_files(
        Arc::clone(&engine),
        &scan_plan,
        config.clone(),
        stream,
        file_selection,
    )
    .await?;

    let file_id_field = scan_plan.contract.file_id_field.clone();
    if scan_plan.is_metadata_only() && !scan_plan.contract.retain_row_index {
        let map_file = |(file_index, f): (usize, &ScanFileContext)| {
            Ok((
                compact_internal_file_id(file_index),
                match &f.stats.num_rows {
                    Precision::Exact(n) => *n,
                    _ => {
                        return plan_err!(
                            "Expected exact row counts in file: {}",
                            super::redact_url_for_error(&f.meta.location)
                        );
                    }
                },
            ))
        };

        let maybe_file_rows = replayed
            .files
            .iter()
            .enumerate()
            .map(map_file)
            .try_collect::<_, VecDeque<_>, _>();
        if let Ok(file_rows) = maybe_file_rows {
            let retain_file_id = scan_plan.contract.retain_file_id;
            let ReplayedScanFiles {
                transforms,
                dvs,
                public_file_ids,
                metrics,
                ..
            } = replayed;
            let exec = DeltaScanMetaExec::new(
                Arc::new(scan_plan),
                vec![file_rows],
                Arc::new(transforms),
                Arc::new(dvs.into_iter().collect()),
                Arc::new(public_file_ids),
                retain_file_id.then_some(file_id_field),
                metrics,
            );
            return Ok(Arc::new(exec) as _);
        }
    }

    let file_pruner = runtime_filter.map(|filter| {
        Arc::new(RuntimeScanFilePruner::new(
            Arc::clone(filter),
            scan_plan.snapshot.clone(),
            config.clone(),
            engine,
            &replayed.files,
            replayed.files_scanned.clone(),
            // The file id is the partition column after the Parquet file columns
            Column::new(
                scan_plan.contract.file_id_field.name(),
                scan_plan.parquet_read_schema.fields().len(),
            ),
        ))
    });

    let file_sort_order = resolve_file_sort_order(config, &scan_plan);
    get_data_scan_plan(
        session,
        scan_plan,
        replayed,
        limit,
        file_pruner,
        file_sort_order,
        config.assume_no_overlap_on_sort,
    )
    .await
}

/// Resolve the configured per-file sort order into a physical ordering over
/// the parquet read schema.
///
/// Column-mapped tables translate logical column names to physical parquet
/// names. A sort column that is not part of the parquet scan (e.g. dropped by
/// the query's projection) truncates the ordering there: the remaining prefix
/// is still a valid ordering. Returns `None` when no order is configured or
/// no prefix could be resolved.
fn resolve_file_sort_order(
    config: &DeltaScanConfig,
    scan_plan: &KernelScanPlan,
) -> Option<LexOrdering> {
    if config.file_sort_order.is_empty() {
        return None;
    }
    let table_config = scan_plan.table_configuration();
    let column_mapping = table_config.is_feature_enabled(&TableFeature::ColumnMapping);
    let read_schema = &scan_plan.parquet_read_schema;

    let mut sort_exprs = Vec::with_capacity(config.file_sort_order.len());
    for sort_column in &config.file_sort_order {
        let physical_name = if column_mapping {
            let Some(physical) = scan_plan
                .scan
                .logical_schema()
                .field(&sort_column.column)
                .and_then(|field| field.make_physical(table_config.column_mapping_mode()).ok())
            else {
                break;
            };
            physical.name().to_string()
        } else {
            sort_column.column.clone()
        };
        let Ok(index) = read_schema.index_of(&physical_name) else {
            break;
        };
        sort_exprs.push(PhysicalSortExpr::new(
            Arc::new(Column::new(&physical_name, index)),
            SortOptions {
                descending: sort_column.descending,
                nulls_first: sort_column.nulls_first,
            },
        ));
    }
    LexOrdering::new(sort_exprs)
}

/// The longest prefix of `ordering` whose columns provably contain no nulls
/// in any of the files, and cannot contain NaN: the column is not a float,
/// and either it is non-nullable in the read schema or every file's
/// statistics report an exact null count of zero.
///
/// Nulls sort at one end of *each file*, so concatenating multiple files into
/// one ordered group interleaves each file's nulls into the middle of the
/// group's stream. Min/max statistics cannot detect this, so an ordering on a
/// column that may contain nulls must not be declared for multi-file groups.
///
/// NaN does the same - it sorts after every other value of its file and is
/// left out of the min/max statistics - but no statistic counts it, so a
/// float column can never be shown free of it and always ends the prefix.
fn null_free_ordering_prefix<'a>(
    ordering: &LexOrdering,
    read_schema: &SchemaRef,
    files: impl IntoIterator<Item = &'a PartitionedFile> + Clone,
) -> Option<LexOrdering> {
    let sort_exprs = ordering
        .iter()
        .take_while(|sort_expr| {
            let Some(column) = sort_expr.expr.downcast_ref::<Column>() else {
                return false;
            };
            let field = read_schema.field(column.index());
            if field.data_type().is_floating() {
                return false;
            }
            if !field.is_nullable() {
                return true;
            }
            files.clone().into_iter().all(|file| {
                file.statistics
                    .as_ref()
                    .and_then(|stats| stats.column_statistics.get(column.index()))
                    .is_some_and(|column_stats| column_stats.null_count == Precision::Exact(0))
            })
        })
        .cloned()
        .collect_vec();
    LexOrdering::new(sort_exprs)
}

/// Per-column types that every key handed to one sort must share, learned
/// from the first key seen.
///
/// `compare_rows` fails across `ScalarValue` variants, and `sort_by` panics
/// on a comparator that is not a total order. Requiring every key to share
/// the first key's per-column types is what lets the sorts here compare with
/// `expect` instead of collapsing an incomparable pair to `Equal`.
#[derive(Default)]
pub(super) struct KeyTypes(Option<Vec<DataType>>);

impl KeyTypes {
    /// Whether `key` is comparable with every key accepted before it.
    pub(super) fn accept(&mut self, key: &[ScalarValue]) -> bool {
        match &self.0 {
            None => {
                let types: Vec<DataType> = key.iter().map(ScalarValue::data_type).collect();
                if types.iter().any(DataType::is_nested) {
                    return false;
                }
                self.0 = Some(types);
                true
            }
            Some(types) => key
                .iter()
                .map(ScalarValue::data_type)
                .eq(types.iter().cloned()),
        }
    }
}

/// Where the knowledge that files do not overlap comes from.
///
/// Delta statistics are per column, not lexicographic, so a multi-column sort
/// order can leave files that do not really overlap looking as though they
/// might. Two files sorted by `(date, time)` covering `(0, 17)..(1, 9)` and
/// `(1, 17)..(2, 9)` report `date` in `[0, 1]` and `[1, 2]` and `time` in
/// `[0, 23]` for both: the tie on `date` falls through to `time`, where the
/// second file's minimum sits below the first's maximum. Nothing rules out an
/// overlap, so [`Prove`](Self::Prove) refuses them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OverlapPolicy {
    /// Order files only when their statistics show the ranges to be disjoint.
    Prove,
    /// Take the ranges to be disjoint on the table's say-so
    /// ([`DeltaScanConfig::assume_no_overlap_on_sort`]), ordering the files on
    /// their range endpoints alone.
    Assume,
}

/// Reorder `items` by a permutation of their indices.
fn apply_order<T>(items: Vec<T>, order: Vec<usize>) -> Vec<T> {
    let mut slots: Vec<Option<T>> = items.into_iter().map(Some).collect();
    order
        .into_iter()
        .map(|index| slots[index].take().expect("a permutation of the items"))
        .collect()
}

/// The sort options of `ordering`, column by column.
fn sort_options(ordering: &LexOrdering) -> Vec<SortOptions> {
    ordering.iter().map(|sort_expr| sort_expr.options).collect()
}

/// Arrange `items` in the order in which their files (`file` reads the
/// `PartitionedFile` off an item) are mutually non-overlapping on `ordering`.
/// When they overlap or cannot be ordered, `Err` hands the items back as they
/// came.
///
/// This is a specialised version of `FileScanConfig::split_groups_by_statistics`,
/// which splits files into multiple groups. This version only forms a single
/// group and avoids extra overhead involved in split_groups_by_statistics
/// like cloning every file. It also allows ranges to touch at the end-points,
/// which `split_groups_by_statistics` does not.
///
/// Values that cannot be compared, and null bounds, refuse the files rather
/// than being ordered arbitrarily. Nulls among the sort columns are the
/// caller's concern: the bounds say nothing about them.
///
/// The files are proven non-overlapping first, falling back on the table's
/// assertion only when `overlap` allows it and the statistics could not.
/// Under [`OverlapPolicy::Assume`] the overlap check is then replaced by the
/// assertion and the files are placed on their endpoints alone; see
/// [`non_overlapping_order`] for what that changes.
///
/// `Ok` carries the arrangement and whether the assertion is what produced it;
/// tables the proving path already handled keep their behaviour either way.
pub(super) fn arrange_non_overlapping_files<T>(
    items: Vec<T>,
    file: impl Fn(&T) -> &PartitionedFile,
    ordering: &LexOrdering,
    overlap: OverlapPolicy,
) -> Result<(Vec<T>, bool), Vec<T>> {
    // Reading the ranges walks every file's statistics and clones two bounds
    // per sort column, so both policies run over the one reading. A table that
    // asserts non-overlap is one the proving pass is expected to fail, so that
    // walk would otherwise be paid for twice on every query against it.
    let Some(ranges) = file_ranges(items.iter().map(file), ordering) else {
        return Err(items);
    };
    let options = sort_options(ordering);
    if let Some(order) = non_overlapping_order(&ranges, &options, OverlapPolicy::Prove) {
        return Ok((apply_order(items, order), false));
    }
    if overlap == OverlapPolicy::Prove {
        return Err(items);
    }
    match non_overlapping_order(&ranges, &options, OverlapPolicy::Assume) {
        Some(order) => Ok((apply_order(items, order), true)),
        None => Err(items),
    }
}

/// Each file's range endpoints under `ordering`, in the order the files came
/// in; see [`FileRange`]. `None` when a file carries no usable bounds for the
/// sort columns, which refuses the whole set.
///
/// Values that cannot be compared, and null bounds, refuse the files rather
/// than being ordered arbitrarily.
///
/// The bounds are taken to be exact. Wider ones only make
/// [`OverlapPolicy::Prove`] refuse more often and are safe there, but can make
/// [`OverlapPolicy::Assume`] order files incorrectly. Asserting the
/// non-overlapping property also asserts that the min/max statistics can be
/// used to compute the sorted file order.
fn file_ranges<'a>(
    files: impl IntoIterator<Item = &'a PartitionedFile>,
    ordering: &LexOrdering,
) -> Option<Vec<FileRange>> {
    let mut ranges: Vec<FileRange> = Vec::new();
    let mut key_types = KeyTypes::default();
    for (index, file) in files.into_iter().enumerate() {
        let stats = file.statistics.as_ref()?;
        let mut starts = Vec::with_capacity(ordering.len());
        let mut ends = Vec::with_capacity(ordering.len());
        for sort_expr in ordering.iter() {
            let column = sort_expr.expr.downcast_ref::<Column>()?;
            let column_stats = stats.column_statistics.get(column.index())?;
            // Read the bounds whatever their precision, as `MinMaxStatistics`
            // does. A null bound is an absent one: stats parsed against a
            // schema wider than the columns a writer indexed carry typed
            // nulls for the rest, and `compare_rows` would place such a bound
            // after every value rather than nowhere.
            let min = column_stats.min_value.get_value()?;
            let max = column_stats.max_value.get_value()?;
            if min.is_null() || max.is_null() {
                return None;
            }
            let (min, max) = (min.clone(), max.clone());
            let (start, end) = if sort_expr.options.descending {
                (max, min)
            } else {
                (min, max)
            };
            starts.push(start);
            ends.push(end);
        }
        if !(key_types.accept(&starts) && key_types.accept(&ends)) {
            // This file has different types for the key columns to what has been seen before
            return None;
        }
        ranges.push((starts, ends, index));
    }
    Some(ranges)
}

/// [`arrange_non_overlapping_files`] under one policy, as a permutation of
/// the files behind `ranges`, which [`file_ranges`] read off them.
///
/// [`OverlapPolicy::Prove`] sorts the files on the start of their range and
/// refuses the whole set as soon as one range reaches into the next.
///
/// [`OverlapPolicy::Assume`] takes the ranges to be disjoint and so orders the
/// files on their endpoints alone; see [`assumed_range_cmp`] for how, and
/// [`assumed_ranges_separate`] for the pairs it still refuses.
///
/// The ranges are read rather than consumed, so one reading of the statistics
/// serves both policies; see [`arrange_non_overlapping_files`].
fn non_overlapping_order(
    ranges: &[FileRange],
    options: &[SortOptions],
    overlap: OverlapPolicy,
) -> Option<Vec<usize>> {
    let cmp = |a: &[ScalarValue], b: &[ScalarValue]| {
        compare_rows(a, b, options).expect("endpoints share a non-nested type per column")
    };
    let mut ordered: Vec<&FileRange> = ranges.iter().collect();
    match overlap {
        OverlapPolicy::Prove => {
            ordered.sort_by(|a, b| cmp(&a.0, &b.0));
            for pair in ordered.windows(2) {
                if cmp(&pair[0].1, &pair[1].0) == Ordering::Greater {
                    return None;
                }
            }
        }
        OverlapPolicy::Assume => {
            ordered.sort_by(|a, b| assumed_range_cmp(a, b, options));
            for pair in ordered.windows(2) {
                if !assumed_ranges_separate(pair[0], pair[1], options) {
                    return None;
                }
            }
        }
    }
    Some(ordered.into_iter().map(|(_, _, index)| *index).collect())
}

/// One file's range under the sort ordering: where it begins and where it ends
/// on each sort column, plus the file's index among the files handed in.
///
/// Only the *leading* column's pair is the file's true first and last key. A
/// file is sorted on that column first, so its minimum and maximum there are
/// the values its first and last rows carry (potentially flipped for a
/// descending order). On any later column the minimum
/// and maximum are taken across the whole file and need not sit at its ends:
/// a file running from `(5, 20)` to `(6, 1)` reports the second column as
/// `[1, 20]`, neither end of which is where the file starts or stops.
type FileRange = (Vec<ScalarValue>, Vec<ScalarValue>, usize);

/// Compare two endpoints on sort column `column` alone, under that column's
/// sort options.
fn compare_endpoints(
    a: &ScalarValue,
    b: &ScalarValue,
    column: usize,
    options: &[SortOptions],
) -> Ordering {
    compare_rows(
        std::slice::from_ref(a),
        std::slice::from_ref(b),
        &options[column..=column],
    )
    .expect("endpoints share a non-nested type per column")
}

/// Order two files taken to be disjoint, on their range endpoints alone.
fn assumed_range_cmp(a: &FileRange, b: &FileRange, options: &[SortOptions]) -> Ordering {
    (0..options.len())
        .find_map(|column| {
            // Compare start values
            match compare_endpoints(&a.0[column], &b.0[column], column, options) {
                Ordering::Equal => {}
                order => return Some(order),
            }
            // Compare end values in the case of equal starts. A file with a range
            // inside another's span is ordered first.
            match compare_endpoints(&a.1[column], &b.1[column], column, options) {
                Ordering::Equal => None,
                order => Some(order),
            }
        })
        // All sort columns are equal so the files are considered equal:
        .unwrap_or(Ordering::Equal)
}

/// Whether the statistics place `a` and `b` apart at all, so that
/// [`assumed_range_cmp`] ordering them means something.
///
/// Walks the sort columns until one of them settles the pair:
///
/// * When ranges begin at different values, the file beginning earlier is
///   ordered first.
/// * Both ranges beginning with the same value can be ordered if one file
///   has a constant value. The constant values file is ordered first.
/// * If both are constant, move on to the next column.
/// * If neither file holds a constant value, the files can't be ordered and
///   we return false.
///
/// Running out of columns leaves both files pinned to the same key on every
/// one of them, so either ordering is valid, and we return true.
fn assumed_ranges_separate(a: &FileRange, b: &FileRange, options: &[SortOptions]) -> bool {
    let pinned = |range: &FileRange, column: usize| {
        compare_endpoints(&range.0[column], &range.1[column], column, options) == Ordering::Equal
    };
    for column in 0..options.len() {
        if compare_endpoints(&a.0[column], &b.0[column], column, options) != Ordering::Equal {
            return true;
        }
        match (pinned(a, column), pinned(b, column)) {
            (true, true) => continue,
            (a_pinned, b_pinned) => return a_pinned || b_pinned,
        }
    }
    true
}

/// Cut a globally ordered, mutually non-overlapping file list into contiguous
/// groups. Group boundaries follow the ordering, so the groups themselves are
/// non-overlapping and range-ordered.
///
/// Produces exactly `target_partitions` groups (none when there are no files,
/// fewer only when there are not enough files, more only to stay within the
/// file-id dictionary key space), with group sizes differing by at most one
/// file.
fn chunk_ordered_files(files: Vec<PartitionedFile>, target_partitions: usize) -> Vec<FileGroup> {
    if files.is_empty() {
        return Vec::new();
    }
    let num_files = files.len();
    let num_groups = target_partitions
        .clamp(1, num_files)
        .max(num_files.div_ceil(MAX_PARTITION_DICT_CARDINALITY));
    let base_size = num_files / num_groups;
    let leftover = num_files % num_groups;
    let mut files = files.into_iter();
    (0..num_groups)
        .map(|group| {
            // The first `leftover` groups take one extra file.
            let size = base_size + usize::from(group < leftover);
            files.by_ref().take(size).collect::<FileGroup>()
        })
        .collect()
}

/// Build scan file groups such that the files within each group are mutually
/// non-overlapping and ordered on the declared file sort order, so reading a
/// group sequentially preserves that order.
///
/// When every file is non-overlapping with every other file, the ordered file
/// list is cut into contiguous groups that are also non-overlapping *between*
/// each other.
///
/// Otherwise, the statistics-based grouping creates additional groups beyond
/// the target partition count when file ranges overlap; with heavily
/// overlapping files this degrades to one group per file. To bound the number
/// of scan partitions, the result is only used when it stays within
/// `max(64, 2 * target_partitions)` groups.
///
/// Falls back to the default grouping when that cap is exceeded, when
/// statistics are unavailable, or when a group would exceed the file-id
/// dictionary key space; the declared output ordering is then dropped by
/// DataFusion's statistics-based validation of scan orderings rather than
/// producing wrong results.
///
/// [`OverlapPolicy::Assume`] only comes into play once the statistics have
/// failed to rule out an overlap. The files are then placed on those
/// statistics alone, so there is always exactly one ordered run and the
/// grouping below - along with its group cap - is reached only when a file
/// cannot be placed at all. The caller is then responsible for re-declaring
/// the ordering that DataFusion's validation drops; see
/// [`DeltaScanExec::with_no_overlap_assumption`].
///
/// The returned flag reports whether the assertion was actually used; it is
/// false whenever the ordinary statistics-proving path settled the order.
fn split_file_groups_for_ordering(
    files: Vec<PartitionedFile>,
    ordering: &LexOrdering,
    table_schema: &SchemaRef,
    target_partitions: usize,
    overlap: OverlapPolicy,
) -> (Vec<FileGroup>, bool) {
    let max_groups = max_num_groups(target_partitions);

    // Missing/unusable statistics fall through: the grouping below fails the
    // same way and takes its default-grouping fallback.
    let files = match arrange_non_overlapping_files(files, |file| file, ordering, overlap) {
        Ok((ordered_files, assumed)) => {
            return (
                chunk_ordered_files(ordered_files, target_partitions),
                assumed,
            );
        }
        Err(files) => {
            if overlap == OverlapPolicy::Assume {
                debug!(
                    "file statistics cannot place every file in the declared sort order; ignoring the no-overlap assertion"
                );
            }
            files
        }
    };

    let flat = vec![FileGroup::new(files)];
    let default_grouping = |flat: Vec<FileGroup>| {
        partitioned_files_to_file_groups(
            flat.into_iter().flat_map(FileGroup::into_inner),
            target_partitions,
        )
    };
    let groups = match FileScanConfig::split_groups_by_statistics_with_target_partitions(
        table_schema,
        &flat,
        ordering,
        target_partitions.max(1),
    ) {
        Ok(groups)
            if groups.len() <= max_groups
                && groups
                    .iter()
                    .all(|group| group.len() <= MAX_PARTITION_DICT_CARDINALITY) =>
        {
            groups
        }
        Ok(groups) => {
            if groups.len() > max_groups {
                debug!(
                    groups = groups.len(),
                    max_groups,
                    "file sort order grouping of heavily overlapping files produced too many groups; falling back to default grouping"
                );
            } else {
                debug!(
                    "file sort order grouping exceeded the file-id dictionary key space; falling back to default grouping"
                );
            }
            default_grouping(flat)
        }
        Err(err) => {
            debug!(
                error = %err,
                "failed to group files by statistics for the declared file sort order; falling back to default grouping"
            );
            default_grouping(flat)
        }
    };
    (groups, false)
}

/// Load deletion-vector keep masks for the selected files.
///
/// Drain [`ScanFileStream`] to start the DV loading tasks, then collect their
/// results through [`ReceiverStreamBuilder::build`]. A successful collection
/// waits for all DV tasks to finish.
pub(super) async fn replay_deletion_vectors(
    engine: Arc<dyn Engine>,
    scan_plan: &KernelScanPlan,
    config: &DeltaScanConfig,
    stream: ScanMetadataStream,
    file_selection: Option<&ResolvedFileSelection>,
) -> Result<Vec<DeletionVectorSelection>> {
    let mut stream = ScanFileStream::new(
        engine,
        &scan_plan.scan,
        config.clone(),
        file_selection.map(|selection| &selection.active_file_ids),
        stream,
    );
    while stream.try_next().await?.is_some() {}

    let dv_stream = stream.dv_stream.build();
    // A DV task must return a keep mask.
    let dvs: DashMap<_, _> = dv_stream
        .and_then(|(url, dv, num_records, cardinality)| {
            ready(match dv {
                Some(keep_mask) => validate_dv_mask(&keep_mask, num_records, cardinality, &url)
                    .and_then(|()| normalize_dv_keep_mask_for_api(keep_mask, num_records, &url))
                    .map(|mask| (url.to_string(), mask))
                    .map_err(DeltaTableError::from),
                None => Err(DeltaTableError::generic(
                    "Invariant violation: DV task spawned for file without deletion vector",
                )),
            })
        })
        .try_collect()
        .await?;

    let mut vectors: Vec<_> = dvs
        .into_iter()
        .map(|(filepath, keep_mask)| DeletionVectorSelection {
            filepath,
            keep_mask,
        })
        .collect();
    vectors.sort_unstable_by(|left, right| left.filepath.cmp(&right.filepath));
    Ok(vectors)
}

pub(super) async fn resolve_file_selection(
    selection: &FileSelection,
    scan_plan: &KernelScanPlan,
    stream: ScanMetadataStream,
) -> Result<ResolvedFileSelection> {
    let requested_file_ids =
        resolve_input_file_ids_on_blocking_pool(selection, scan_plan.scan.table_root()).await?;
    if requested_file_ids.is_empty() {
        return Ok(ResolvedFileSelection::new(
            HashSet::new(),
            Vec::new(),
            selection.missing_file_policy,
        ));
    }

    let mut missing_file_ids = requested_file_ids;
    let selected_active_file_ids = collect_selected_active_file_ids(
        scan_plan.scan.table_root(),
        stream,
        &mut missing_file_ids,
    )
    .await?;
    let mut missing_file_ids: Vec<_> = missing_file_ids.into_iter().collect();
    missing_file_ids.sort_unstable();

    let resolved = ResolvedFileSelection::new(
        selected_active_file_ids,
        missing_file_ids,
        selection.missing_file_policy,
    );
    resolved.validate_missing()?;
    Ok(resolved)
}

async fn resolve_input_file_ids_on_blocking_pool(
    selection: &FileSelection,
    table_root: &Url,
) -> Result<HashSet<String>> {
    let selection = selection.clone();
    let table_root = table_root.clone();
    tokio::task::spawn_blocking(move || selection.resolve_input_file_ids(&table_root))
        .await
        .map_err(|err| DataFusionError::External(Box::new(err)))?
        .map_err(DataFusionError::from)
}

async fn collect_selected_active_file_ids(
    table_root: &Url,
    stream: ScanMetadataStream,
    missing_file_ids: &mut HashSet<String>,
) -> Result<HashSet<String>> {
    let mut selected_active_file_ids = HashSet::new();
    for_each_selected_file(table_root, stream, |file_url| {
        let file_id = file_url.to_string();
        if missing_file_ids.remove(&file_id) {
            selected_active_file_ids.insert(file_id);
        }
        !missing_file_ids.is_empty()
    })
    .await?;
    Ok(selected_active_file_ids)
}

/// Call `f` with the URL of each file that a kernel scan selects, until `f` returns `false`.
async fn for_each_selected_file(
    table_root: &Url,
    mut stream: ScanMetadataStream,
    mut f: impl FnMut(Url) -> bool,
) -> Result<()> {
    while let Some(scan_data) = stream.try_next().await? {
        let (data, mut selection_vector) = scan_data.scan_files.into_parts();
        let batch: RecordBatch = ArrowEngineData::try_from_engine_data(data)
            .map_err(DeltaTableError::from)?
            .into();
        // Delta Kernel may return a short selection vector. Missing entries are selected.
        selection_vector.resize(batch.num_rows(), true);

        for (idx, selected) in selection_vector.into_iter().enumerate() {
            if selected {
                let file_url = replay::parse_path(
                    table_root,
                    LogicalFileView::new(batch.clone(), idx).path_raw(),
                )?;
                if !f(file_url) {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

async fn replay_files(
    engine: Arc<dyn Engine>,
    scan_plan: &KernelScanPlan,
    scan_config: DeltaScanConfig,
    stream: ScanMetadataStream,
    file_selection: Option<&ResolvedFileSelection>,
) -> Result<ReplayedScanFiles> {
    let mut stream = ScanFileStream::new(
        engine,
        &scan_plan.scan,
        scan_config,
        file_selection.map(|selection| &selection.active_file_ids),
        stream,
    );
    let mut files = Vec::new();
    while let Some(file) = stream.try_next().await? {
        files.extend(file);
    }

    let mut public_file_ids = PublicFileIdMap::default();
    if scan_plan.contract.retain_file_id {
        for (file_index, file) in files.iter().enumerate() {
            public_file_ids.insert(
                compact_internal_file_id(file_index),
                file.meta.location.to_string(),
            );
        }
    }

    let transforms: HashMap<_, _> = files
        .iter_mut()
        .enumerate()
        .flat_map(|(file_index, file)| {
            file.transform
                .take()
                .map(|t| (compact_internal_file_id(file_index), t))
        })
        .collect();

    let dv_stream = stream.dv_stream.build();
    let dvs_by_url: HashMap<_, _> = dv_stream
        .try_filter_map(|(url, dv, num_records, cardinality)| {
            ready(match dv {
                Some(mask) => validate_dv_mask(&mask, num_records, cardinality, &url)
                    .map(|()| Some((url.to_string(), mask)))
                    .map_err(DeltaTableError::from),
                None => Err(DeltaTableError::generic(
                    "Invariant violation: DV task returned no selection vector",
                )),
            })
        })
        .try_collect()
        .await?;
    let dvs = remap_deletion_vectors_to_internal_file_ids(&files, dvs_by_url)?;

    let metrics = ExecutionPlanMetricsSet::new();
    // A gauge, because a runtime filter can skip planned files.
    let files_scanned = MetricBuilder::new(&metrics).global_gauge("count_files_scanned");
    files_scanned.add(stream.metrics.num_scanned);

    Ok(ReplayedScanFiles {
        files,
        transforms,
        dvs,
        public_file_ids,
        metrics,
        files_scanned,
    })
}

fn validate_dv_mask(
    mask: &[bool],
    num_records: Option<u64>,
    cardinality: i64,
    file_url: &Url,
) -> Result<()> {
    let deleted = u64::try_from(cardinality).map_err(|_| {
        DataFusionError::Plan(format!(
            "Negative deletion vector cardinality for file: {}",
            super::redact_url_for_error(file_url)
        ))
    })?;
    let actual = mask.iter().filter(|keep| !**keep).count() as u64;
    if deleted != actual
        || num_records.is_some_and(|count| deleted > count || mask.len() as u64 > count)
    {
        return plan_err!(
            "Deletion vector cardinality or mask exceeds physical row count for file: {}",
            super::redact_url_for_error(file_url)
        );
    }
    Ok(())
}

/// Pad a DV keep mask to `numRecords` for `deletion_vectors()`.
///
/// Kernel stops the mask at the highest deleted row. Fill trailing entries with `true`.
/// Return an error when `numRecords` is missing or shorter than the mask.
///
/// Scan execution uses physical positions without padding.
fn normalize_dv_keep_mask_for_api(
    mut mask: Vec<bool>,
    num_records: Option<u64>,
    file_url: &Url,
) -> Result<Vec<bool>> {
    let redacted_url = super::redact_url_for_error(file_url);
    let Some(num_records) = num_records else {
        return plan_err!(
            "Missing numRecords for file with deletion vector: {}",
            redacted_url
        );
    };
    let num_records = usize::try_from(num_records).map_err(|_| {
        DataFusionError::Execution(format!(
            "numRecords does not fit usize for file with deletion vector: {redacted_url}"
        ))
    })?;
    if mask.len() > num_records {
        return plan_err!(
            "Deletion vector mask length {} exceeds numRecords {} for file: {}",
            mask.len(),
            num_records,
            redacted_url
        );
    }
    mask.resize(num_records, true);
    Ok(mask)
}

fn validate_dv_parquet_metadata(
    metadata: &ParquetMetaData,
    log_count: Option<u64>,
    mask_len: usize,
    id: &str,
    file_id_name: &str,
) -> Result<u64> {
    let count = u64::try_from(metadata.file_metadata().num_rows()).map_err(|_| {
        DataFusionError::Plan(format!("negative Parquet row count for file id '{id}'"))
    })?;
    if count > i64::MAX as u64 || log_count.is_some_and(|n| n != count) {
        return plan_err!("Parquet and Delta physical row counts disagree for file id '{id}'");
    }
    let mut total = 0_u64;
    for (group_index, group) in metadata.row_groups().iter().enumerate() {
        let Ok(expected) = i16::try_from(group_index) else {
            return plan_err!(
                "Parquet row-group index {group_index} exceeds i16 for file id '{id}'"
            );
        };
        if group.ordinal() != Some(expected) {
            return plan_err!(
                "Parquet row-group ordinal is missing or inconsistent for file id '{id}'"
            );
        }
        let rows = u64::try_from(group.num_rows()).map_err(|_| {
            DataFusionError::Plan(format!(
                "negative Parquet row-group count for file id '{id}'"
            ))
        })?;
        total = total.checked_add(rows).ok_or_else(|| {
            DataFusionError::Plan(format!(
                "Parquet row-group count overflow for file id '{id}'"
            ))
        })?;
    }
    if total != count {
        return plan_err!("Parquet row groups disagree with footer count for file id '{id}'");
    }
    if metadata
        .file_metadata()
        .schema_descr()
        .root_schema()
        .get_fields()
        .iter()
        .any(|field| field.name() == PHYSICAL_POSITION_COLUMN || field.name() == file_id_name)
    {
        return plan_err!(
            "Parquet field collides with an internal DV scan column for file id '{id}'"
        );
    }
    if mask_len as u64 > count {
        return plan_err!("Deletion vector mask exceeds Parquet row count for file id '{id}'");
    }
    Ok(count)
}

async fn get_data_scan_plan(
    session: &dyn Session,
    scan_plan: KernelScanPlan,
    replayed: ReplayedScanFiles,
    limit: Option<usize>,
    file_pruner: Option<Arc<RuntimeScanFilePruner>>,
    // Sort order (over `parquet_read_schema`) that every data file adheres to.
    file_sort_order: Option<LexOrdering>,
    // Whether the table asserts its files never overlap on `file_sort_order`.
    config_assumes_no_overlap: bool,
) -> Result<Arc<dyn ExecutionPlan>> {
    let ReplayedScanFiles {
        files,
        transforms,
        dvs,
        public_file_ids,
        metrics,
        ..
    } = replayed;
    let has_deletion_vectors = !dvs.is_empty();
    let mut partition_stats = HashMap::new();
    let log_counts: Vec<_> = files.iter().map(|file| file.num_records).collect();
    let partition_column_names = scan_plan
        .table_configuration()
        .metadata()
        .partition_columns();
    // The files share only as many distinct partition tuples as the table has
    // partitions: intern them, so that every file's extension is a handle to
    // a shared tuple and each tuple's bounds are folded into the aggregate
    // statistics once. Only null counts accrue per file.
    let mut interned: HashSet<Arc<[ScalarValue]>> = HashSet::new();

    // Convert files into DataFusion `PartitionedFile`s grouped by object store.
    // Create one `DataSourceExec` plan for each store.
    // Add a compact scan file id as a partition value for file correlation.
    // The exec maps that id back to the public file path only when the file column is projected.
    let to_store_file = |(file_index, f): (usize, ScanFileContext)| {
        // Convert each file's kernel partition values once; the aggregated
        // statistics and the per-file extension must agree.
        let partition_values = f
            .partitions
            .as_ref()
            .map(|data| extract_partition_values(data, partition_column_names))
            .transpose()?
            .map(|values| match interned.get(values.as_slice()) {
                Some(shared) => Arc::clone(shared),
                None => {
                    fold_partition_bounds(partition_column_names, &values, &mut partition_stats);
                    let shared: Arc<[ScalarValue]> = values.into();
                    interned.insert(Arc::clone(&shared));
                    shared
                }
            });
        if let Some(values) = &partition_values {
            add_partition_null_counts(
                partition_column_names,
                values,
                &f.stats,
                &mut partition_stats,
            );
        }
        let mut partitioned_file = to_partitioned_file(&f.meta)?;
        let file_value = wrap_file_id_value(compact_internal_file_id(file_index));
        // NOTE: `PartitionedFile::with_statistics` appends exact stats for partition columns based
        // on `partition_values`, so partition values must be set first.
        partitioned_file.partition_values = vec![file_value.clone()];
        if f.stats.column_statistics.is_empty() {
            // No predicate to prune on (#4522): keep only num_rows/total_byte_size, no
            // per-file column vector. Assign directly so `with_statistics` doesn't add a
            // partition-column entry; table-level column stats are rebuilt below.
            partitioned_file.statistics = Some(Arc::new(f.stats));
        } else {
            // Avoid doubling the stats Vec when `with_statistics` appends partition entries.
            let mut stats = f.stats;
            stats
                .column_statistics
                .reserve_exact(partitioned_file.partition_values.len());
            partitioned_file = partitioned_file.with_statistics(Arc::new(stats));
        }
        // `DeltaScanExec::try_pushdown_sort` regroups files by their
        // partition-column values; attach them so they survive the regrouping.
        // Whether that path is ever taken is decided there, not here.
        if let Some(values) = partition_values {
            partitioned_file
                .extensions
                .insert(DeltaPartitionValues(values));
        }
        Ok::<_, DataFusionError>((f.meta.location.as_object_store_url(), partitioned_file))
    };

    let partitioned_files = files
        .into_iter()
        .enumerate()
        .map(to_store_file)
        .try_collect::<_, Vec<_>, _>()?;

    let mut dv_metadata_caches: HashMap<ObjectStoreUrl, Arc<FileMetadataCache>> = HashMap::new();
    let mut uses_dv_cache = vec![false; partitioned_files.len()];

    let dv_state = if has_deletion_vectors {
        let file_id_name = scan_plan.contract.file_id_field.name();
        if file_id_name == PHYSICAL_POSITION_COLUMN
            || scan_plan
                .parquet_read_schema
                .field_with_name(PHYSICAL_POSITION_COLUMN)
                .is_ok()
        {
            return plan_err!("physical position column conflicts with the scan schema");
        }
        let mut masks = dvs;
        let mut entries = HashMap::new();
        let mut identities = PhysicalFileIdentityMap::new();
        let mut footer_tasks = Vec::new();
        for (index, (store_url, file)) in partitioned_files.iter().enumerate() {
            let id = compact_internal_file_id(index);
            let mask = masks.remove(&id);
            if let Some(mask) = mask {
                uses_dv_cache[index] = true;
                dv_metadata_caches
                    .entry(store_url.clone())
                    .or_insert_with(|| Arc::new(DefaultCache::new(usize::MAX)));
                footer_tasks.push((
                    id.clone(),
                    store_url.clone(),
                    file.object_meta.clone(),
                    log_counts[index],
                    mask,
                ));
            } else {
                entries.insert(
                    id.clone(),
                    exec::DeletionVectorEntry {
                        keep_mask: None,
                        row_count: None,
                    },
                );
            }
            identities.insert(
                id,
                PhysicalFileIdentity {
                    object_store_url: store_url.clone(),
                    object_meta: file.object_meta.clone(),
                },
            );
        }
        if !masks.is_empty() {
            return plan_err!("Deletion vector was loaded for an unselected file");
        }
        let loaded = futures::stream::iter(footer_tasks)
            .map(|(id, store_url, object_meta, log_count, mask)| {
                let cache = Arc::clone(
                    dv_metadata_caches
                        .get(&store_url)
                        .expect("selected store must have a scan-local metadata cache"),
                );
                async move {
                    let store = session.runtime_env().object_store(&store_url)?;
                    let metadata = DFParquetMetadata::new(store.as_ref(), &object_meta)
                        .with_file_metadata_cache(Some(cache))
                        .with_page_index_policy(Some(PageIndexPolicy::Skip))
                        .fetch_metadata()
                        .await?;
                    let count = validate_dv_parquet_metadata(
                        &metadata,
                        log_count,
                        mask.len(),
                        &id,
                        file_id_name,
                    )?;
                    Ok::<_, DataFusionError>((
                        id,
                        exec::DeletionVectorEntry {
                            keep_mask: Some(mask),
                            row_count: Some(count),
                        },
                    ))
                }
            })
            .buffer_unordered(8)
            .try_collect::<Vec<_>>()
            .await?;
        entries.extend(loaded);
        DvExecutionState::Physical {
            entries: Arc::new(entries),
            physical_file_identities: Arc::new(identities),
        }
    } else {
        DvExecutionState::NotPresent
    };

    let files_by_store = partitioned_files
        .into_iter()
        .zip(uses_dv_cache)
        .map(|((store, file), uses_dv_cache)| ((store, uses_dv_cache), file))
        .into_group_map()
        .into_iter()
        .map(|((store, uses_dv_cache), files)| {
            let cache = uses_dv_cache.then(|| {
                dv_metadata_caches
                    .get(&store)
                    .expect("DV store must have a scan-local metadata cache")
                    .clone()
            });
            (store, files, has_deletion_vectors, cache)
        });

    // TODO(roeap); not sure exactly how row tracking is implemented in kernel right now
    // so leaving predicate as None for now until we are sure this is safe to do.
    let table_config = scan_plan.table_configuration();
    let predicate = if table_config.is_feature_enabled(&TableFeature::RowTracking) {
        None
    } else {
        scan_plan.parquet_predicate.as_ref()
    };
    let file_id_field = scan_plan.contract.file_id_field.clone();
    // The assertion says files do not overlap on the declared sort order, so
    // it means nothing without one. Scans with deletion vectors read through
    // a single stream: their file groups are coalesced below, which
    // interleaves them, so no arrangement of the groups can be declared.
    let overlap = if config_assumes_no_overlap && file_sort_order.is_some() && !has_deletion_vectors
    {
        OverlapPolicy::Assume
    } else {
        OverlapPolicy::Prove
    };
    let (pq_plan, assumed_ordering) = get_read_plan(
        session,
        files_by_store,
        &scan_plan.parquet_read_schema,
        &scan_plan.parquet_predicate_schema,
        limit,
        &file_id_field,
        predicate,
        file_pruner.as_ref().map(|pruner| pruner.predicate()),
        file_sort_order.as_ref(),
        overlap,
    )
    .await?;
    let pq_plan = if has_deletion_vectors && pq_plan.properties().partitioning.partition_count() > 1
    {
        Arc::new(CoalescePartitionsExec::new(pq_plan)) as Arc<dyn ExecutionPlan>
    } else {
        pq_plan
    };

    let transforms = Arc::new(transforms);
    let public_file_ids = Arc::new(public_file_ids);
    let exec = DeltaScanExec::new(
        Arc::new(scan_plan),
        pq_plan,
        Arc::clone(&transforms),
        dv_state,
        Arc::clone(&public_file_ids),
        partition_stats,
        metrics,
    )
    .with_file_pruner(file_pruner)
    .with_file_sort_order(file_sort_order)
    .with_no_overlap_assumption(overlap, assumed_ordering);

    Ok(Arc::new(exec))
}

/// Partition-column values for one data file, aligned to the table's
/// partition columns. Attached to each [`PartitionedFile`] via an extension;
/// files of the same partition share one tuple.
#[derive(Debug)]
pub(super) struct DeltaPartitionValues(pub(super) Arc<[ScalarValue]>);

/// Extract the partition-column values for one file from its kernel
/// partition struct, aligned to `partition_column_names` (missing columns become
/// [`ScalarValue::Null`]).
fn extract_partition_values(
    data: &StructData,
    partition_column_names: &[String],
) -> Result<Vec<ScalarValue>> {
    partition_column_names
        .iter()
        .map(
            |name| match data.index_of(name).and_then(|i| data.value(i)) {
                Some(scalar) => to_datafusion_scalar(scalar).map_err(|e| {
                    DataFusionError::External(Box::new(e)).context(format!(
                        "failure during planning, when translating the partition value of column '{name}'"
                    ))
                }),
                None => Ok(ScalarValue::Null),
            },
        )
        .try_collect()
}

/// Fold a partition tuple's bounds into the table-wide per-column statistics.
/// `values` is aligned to `partition_column_names`; a null value
/// leaves the column's bounds unknown.
fn fold_partition_bounds(
    partition_column_names: &[String],
    values: &[ScalarValue],
    part_stats: &mut HashMap<String, ColumnStatistics>,
) {
    for (name, converted) in partition_column_names.iter().zip(values.iter()) {
        let value = if converted.is_null() {
            Precision::Absent
        } else {
            Precision::Exact(converted.clone())
        };
        if let Some(part_stat) = part_stats.get_mut(name) {
            part_stat.min_value = part_stat.min_value.min(&value);
            part_stat.max_value = part_stat.max_value.max(&value);
        } else {
            part_stats.insert(
                name.clone(),
                ColumnStatistics {
                    null_count: Precision::Exact(0),
                    min_value: value.clone(),
                    max_value: value,
                    distinct_count: Precision::Absent,
                    sum_value: Precision::Absent,
                    byte_size: Precision::Absent,
                },
            );
        }
    }
}

/// Count one file's rows against every partition column its tuple holds a
/// null for. The tuple's bounds must have been folded already.
fn add_partition_null_counts(
    partition_column_names: &[String],
    values: &[ScalarValue],
    stats: &Statistics,
    part_stats: &mut HashMap<String, ColumnStatistics>,
) {
    for (name, converted) in partition_column_names.iter().zip(values.iter()) {
        if !converted.is_null() {
            continue;
        }
        if let Some(part_stat) = part_stats.get_mut(name) {
            part_stat.null_count = part_stat.null_count.add(&stats.num_rows);
        }
    }
}

type FilesByStore = (
    ObjectStoreUrl,
    Vec<PartitionedFile>,
    bool,
    Option<Arc<FileMetadataCache>>,
);

fn compact_internal_file_id(file_index: usize) -> String {
    file_index.to_string()
}

/// The file index of an id from [`compact_internal_file_id`].
fn internal_file_index(file_id: &str) -> Option<usize> {
    file_id.parse().ok()
}

fn remap_deletion_vectors_to_internal_file_ids(
    files: &[ScanFileContext],
    mut dvs_by_url: HashMap<String, Vec<bool>>,
) -> Result<HashMap<String, Vec<bool>>> {
    let mut dvs = HashMap::new();
    for (file_index, file) in files.iter().enumerate() {
        if dvs_by_url.is_empty() {
            break;
        }
        if let Some(dv) = dvs_by_url.remove(file.meta.location.as_str()) {
            dvs.insert(compact_internal_file_id(file_index), dv);
        }
    }
    if let Some(file_url) = dvs_by_url.keys().next() {
        let redacted_url = Url::parse(file_url)
            .map(|url| super::redact_url_for_error(&url))
            .unwrap_or_else(|_| file_url.clone());
        return Err(internal_datafusion_err!(
            "missing internal file id mapping for file with deletion vector: {redacted_url}"
        ));
    }
    Ok(dvs)
}

fn public_file_id<'a>(
    public_file_ids: &'a PublicFileIdMap,
    internal_file_id: &str,
) -> Result<&'a str> {
    public_file_ids
        .get(internal_file_id)
        .map(String::as_str)
        .ok_or_else(|| {
            internal_datafusion_err!(
                "missing public file id mapping for internal file id '{internal_file_id}'"
            )
        })
}

fn file_id_array_for_value(
    file_id_field: &FieldRef,
    file_id: &str,
    row_count: usize,
) -> Result<ArrayRef> {
    let keys = UInt16Array::from(vec![0u16; row_count]);
    let values: ArrayRef = match file_id_field.data_type() {
        DataType::Dictionary(_, value_type) if value_type.as_ref() == &DataType::Utf8View => {
            if row_count == 0 {
                Arc::new(StringViewArray::from_iter_values(std::iter::empty::<&str>()))
            } else {
                Arc::new(StringViewArray::from_iter_values([file_id]))
            }
        }
        _ => {
            if row_count == 0 {
                Arc::new(StringArray::from(Vec::<Option<&str>>::new()))
            } else {
                Arc::new(StringArray::from(vec![Some(file_id)]))
            }
        }
    };

    let file_id_array: DictionaryArray<UInt16Type> = DictionaryArray::try_new(keys, values)?;
    Ok(Arc::new(file_id_array))
}

/// Maximum number of distinct values representable by DataFusion's default partition dictionary
/// encoding (`Dictionary<UInt16, _>`).
const MAX_PARTITION_DICT_CARDINALITY: usize = (u16::MAX as usize) + 1;

/// Maximum number of file groups to create for a given target partition count.
/// This allows the number of partitions to exceed the target, but prevents a
/// very large number of partitions that could be detrimental to performance.
pub(crate) fn max_num_groups(target_partitions: usize) -> usize {
    usize::max(64, target_partitions.saturating_mul(2))
}

fn partitioned_files_to_file_groups(
    files: impl IntoIterator<Item = PartitionedFile>,
    target_partitions: usize,
) -> Vec<FileGroup> {
    partitioned_files_to_file_groups_with_limit(
        files,
        target_partitions,
        MAX_PARTITION_DICT_CARDINALITY,
    )
}

fn partitioned_files_to_file_groups_with_limit(
    files: impl IntoIterator<Item = PartitionedFile>,
    target_partitions: usize,
    max_files_per_group: usize,
) -> Vec<FileGroup> {
    // Each `PartitionedFile` is assigned to exactly one file group. DeltaScanStream stores
    // row ordinal counters per execution partition. Whole file ownership is required for
    // scan row ordinals, so files are distributed round robin rather than range split.
    //
    // The group count drives scan parallelism: one group is one execution partition, and
    // files within a group are read sequentially. A single group serializes every file
    // fetch, which is ruinous on remote stores with many small files. DataFusion's own
    // repartitioning does not help here: it declines below repartition_file_min_size and
    // would split files across streams, which row ordinals forbid.
    //
    // Partition values are dictionary encoded using a UInt16 key (DataFusion's default
    // `wrap_partition_type_in_dict`). Keep file groups small enough that the file-id partition
    // dictionary doesn't exceed the key space (one distinct value per file).
    let files = files.into_iter().collect_vec();
    if files.is_empty() {
        return Vec::new();
    }
    let group_count = target_partitions
        .max(1)
        .max(files.len().div_ceil(max_files_per_group))
        .min(files.len());
    let mut file_groups = vec![Vec::new(); group_count];
    for (index, file) in files.into_iter().enumerate() {
        file_groups[index % group_count].push(file);
    }
    let file_groups = file_groups.into_iter().map(FileGroup::new).collect_vec();

    #[cfg(debug_assertions)]
    {
        let mut owner_by_path = HashMap::new();
        for (partition, group) in file_groups.iter().enumerate() {
            for file in group.iter() {
                let path = file.object_meta.location.to_string();
                if let Some(previous_partition) = owner_by_path.insert(path.clone(), partition) {
                    debug_assert_eq!(
                        previous_partition, partition,
                        "file {path} was assigned to multiple scan partitions; row indexes require whole file ownership"
                    );
                }
            }
        }
    }

    file_groups
}

#[allow(clippy::too_many_arguments)]
async fn get_read_plan(
    state: &dyn Session,
    files_by_store: impl IntoIterator<Item = FilesByStore>,
    // Schema of physical file columns to read from Parquet (no Delta partitions, no file-id).
    //
    // This is also the schema used for Parquet pruning/pushdown. It may include view types
    // (e.g. Utf8View/BinaryView) depending on `DeltaScanConfig`.
    parquet_read_schema: &SchemaRef,
    // Predicate binding schema used to bind Parquet predicates, including the synthetic file id
    // column when the provider exposes it.
    parquet_predicate_schema: &SchemaRef,
    limit: Option<usize>,
    file_id_field: &FieldRef,
    predicate: Option<&Expr>,
    // Skips the files that the runtime file filter does not keep, see `RuntimeScanFilePruner`.
    // `predicate` is not set when the table has deletion vectors, because it can remove single
    // rows, and the deletion vector of a file must see all rows of that file. This predicate is
    // always set, because it keeps or removes a file with all its rows.
    file_predicate: Option<Arc<dyn PhysicalExpr>>,
    // Sort order (over `parquet_read_schema`) that every data file adheres to.
    file_sort_order: Option<&LexOrdering>,
    // Whether the files may be taken to be non-overlapping on `file_sort_order`
    // without proving it from their statistics. Narrowed to
    // [`OverlapPolicy::Prove`] below when the files span several stores.
    overlap: OverlapPolicy,
    // Returns the read plan, plus the ordering (over `parquet_read_schema`) its
    // partitions follow only by virtue of that assumption, if any.
) -> Result<(Arc<dyn ExecutionPlan>, Option<LexOrdering>)> {
    let mut plans = Vec::new();
    // Set when the grouping used relies on a non-overlapping assertion
    // being set on the table.
    let mut assumed_ordering = None;

    // The per-store plans are unioned, which interleaves their files, so an
    // arrangement resting on the non-overlapping assertion could not be
    // declared for the result whatever each store's grouping looks like.
    // Arranging on the assertion here would only displace the grouping DataFusion
    // can certify by itself and leave the scan advertising no ordering at all,
    // so this grouping keeps to what the statistics prove. The table's
    // declaration is a separate matter, held by the exec: see
    // [`DeltaScanExec::with_no_overlap_assumption`].
    let files_by_store = files_by_store.into_iter().collect_vec();
    let overlap = if files_by_store.len() > 1 {
        OverlapPolicy::Prove
    } else {
        overlap
    };

    // Relax nested-field nullability on the schema handed to the parquet source:
    // Delta's `nullable = false` is a write-time invariant, and Spark-written files
    // may mark nested fields optional. The logical schema is restored above the
    // parquet scan by DeltaScanExec's transforms.
    let parquet_read_schema = Arc::new(relax_schema_nested_nullability(parquet_read_schema));
    let parquet_read_schema = &parquet_read_schema;

    let pq_options = crate::datafile::ReaderProperties::default().to_table_parquet_options(state);
    let metadata_size_hint = pq_options.global.metadata_size_hint;

    let mut full_read_schema = SchemaBuilder::from(parquet_read_schema.as_ref().clone());
    full_read_schema.push(file_id_field.as_ref().clone().with_nullable(true));
    let full_read_schema = Arc::new(full_read_schema.finish());
    let parquet_predicate_df_schema = parquet_predicate_schema.clone().to_dfschema()?;
    let adapter_factory = Arc::new(DeltaPhysicalExprAdapterFactory);

    for (store_url, files, has_deletion_vectors, scan_cache) in files_by_store.into_iter() {
        let store = state.runtime_env().object_store(&store_url)?;
        let metadata_cache = scan_cache
            .unwrap_or_else(|| state.runtime_env().cache_manager.get_file_metadata_cache());
        let reader_factory = Arc::new(CachedParquetFileReaderFactory::new(store, metadata_cache));

        // NOTE: In the "next" provider, DataFusion's Parquet scan partition fields are file-id
        // only. Delta partition columns/values are injected via kernel transforms and handled
        // above Parquet, so they are not part of the Parquet partition schema here.
        let builder = TableSchema::builder(parquet_read_schema.clone())
            .with_table_partition_cols(vec![file_id_field.clone()]);
        let table_schema = if has_deletion_vectors {
            builder
                .with_virtual_columns(vec![Arc::new(
                    Field::new(PHYSICAL_POSITION_COLUMN, DataType::Int64, true)
                        .with_extension_type(RowNumber),
                )])
                .build()
        } else {
            builder.build()
        };
        let full_table_schema = table_schema.table_schema().clone();
        let mut file_source = ParquetSource::new(table_schema)
            .with_table_parquet_options(pq_options.clone())
            .with_parquet_file_reader_factory(reader_factory);
        if let Some(hint) = metadata_size_hint {
            file_source = file_source.with_metadata_size_hint(hint);
        }

        // TODO(roeap); we might be able to also push selection vectors into the read plan
        // by creating parquet access plans. However we need to make sure this does not
        // interfere with other delta features like row ids.
        if !has_deletion_vectors && let Some(pred) = predicate {
            match state.create_physical_expr(pred.clone(), &parquet_predicate_df_schema) {
                Ok(physical) => match adapter_factory
                    .create(parquet_predicate_schema.clone(), full_read_schema.clone())
                {
                    Ok(adapter) => match adapter.rewrite(physical) {
                        Ok(rewritten) => {
                            file_source = file_source
                                .with_predicate(rewritten)
                                .with_pushdown_filters(true);
                        }
                        Err(err) => {
                            debug!(
                                predicate = ?pred,
                                schema = ?parquet_predicate_schema,
                                error = %err,
                                "Skipping parquet predicate pushdown because predicate adaptation to the read schema failed"
                            );
                        }
                    },
                    Err(err) => {
                        debug!(
                            predicate = ?pred,
                            schema = ?parquet_predicate_schema,
                            error = %err,
                            "Skipping parquet predicate pushdown because predicate adapter creation failed"
                        );
                    }
                },
                Err(err) => {
                    debug!(
                        predicate = ?pred,
                        schema = ?parquet_predicate_schema,
                        error = %err,
                        "Skipping parquet predicate pushdown because predicate binding failed"
                    );
                }
            }
        }

        if let Some(file_predicate) = &file_predicate {
            let predicate = match file_source.filter() {
                Some(predicate) => conjunction([predicate, Arc::clone(file_predicate)]),
                None => Arc::clone(file_predicate),
            };
            file_source = file_source.with_predicate(predicate);
        }

        let target_partitions = state.config().target_partitions();
        let (file_groups, store_sort_order) = match file_sort_order {
            Some(ordering) => {
                let null_free_prefix =
                    null_free_ordering_prefix(ordering, parquet_read_schema, files.iter());
                let null_free_len = null_free_prefix.as_ref().map_or(0, |prefix| prefix.len());
                let (file_groups, assumed) = split_file_groups_for_ordering(
                    files,
                    ordering,
                    &full_table_schema,
                    target_partitions,
                    overlap,
                );
                // A single sorted file per group upholds the full ordering even
                // when sort columns contain nulls or NaN; multi-file groups
                // only uphold the prefix of the ordering that is provably free
                // of both (see `null_free_ordering_prefix`).
                let store_sort_order = if file_groups.iter().all(|group| group.len() <= 1) {
                    Some(ordering.clone())
                } else {
                    if null_free_len < ordering.len() {
                        debug!(
                            "file sort order columns may contain nulls or NaN; declaring only the prefix of the ordering that is free of both"
                        );
                    }
                    null_free_prefix.clone()
                };
                // Nulls or NaN within files prevent ordering even when the table asserts
                // an ordering, so we can only declare the null-free prefix as an assumed order.
                if assumed {
                    assumed_ordering = null_free_prefix;
                }
                (file_groups, store_sort_order)
            }
            None => (
                partitioned_files_to_file_groups(files, target_partitions),
                None,
            ),
        };
        let full_table_width = full_table_schema.fields().len();
        let (file_groups, mut statistics) =
            compute_all_files_statistics(file_groups, full_table_schema, true, false)?;

        // Per-file column stats are omitted when there's no predicate (#4522); the merged
        // stat then has an empty column vector. Rebuild it as all-unknown full width so the
        // file source and `map_statistics` see the expected column count.
        if statistics.column_statistics.is_empty() && full_table_width > 0 {
            statistics.column_statistics = vec![ColumnStatistics::new_unknown(); full_table_width];
        }

        let file_group_count = file_groups.len();
        let builder = FileScanConfigBuilder::new(store_url, Arc::new(file_source))
            .with_file_groups(file_groups)
            .with_statistics(statistics)
            .with_limit(if has_deletion_vectors { None } else { limit })
            .with_expr_adapter(Some(adapter_factory.clone() as _));
        let builder = match store_sort_order {
            Some(ordering) => builder.with_output_ordering(vec![ordering]),
            None => builder,
        };
        // If there was a file order set, the groups were arranged by statistics
        // and exact per-group statistics are published. This applies even when the
        // null-free ordering is empty and no output ordering is set.
        // For the exact partition statistics to remain correct,
        // order must be preserved to prevent work stealing from rearranging
        // the data that ends up in each partition.
        let builder = if file_sort_order.is_some() {
            builder.with_preserve_order(true)
        } else {
            builder
        };
        let builder = if has_deletion_vectors {
            builder
                .with_output_partitioning(Some(Partitioning::UnknownPartitioning(file_group_count)))
        } else {
            builder
        };
        let config = builder.build();

        plans.push(DataSourceExec::from_data_source(config) as Arc<dyn ExecutionPlan>);
    }

    Ok((
        match plans.len() {
            0 => Arc::new(EmptyExec::new(full_read_schema.clone())),
            1 => plans.remove(0),
            _ => UnionExec::try_new(plans)?,
        },
        assumed_ordering,
    ))
}

// Small helper to reuse some code between exec and exec_meta
fn finalize_transformed_batch(
    batch: RecordBatch,
    scan_plan: &KernelScanPlan,
    file_id_col: Option<(ArrayRef, FieldRef)>,
    schema_adapter: &mut SchemaAdapter,
) -> Result<RecordBatch> {
    let result = if let Some(projection) = scan_plan.contract.result_projection.as_ref() {
        batch.project(projection)?
    } else {
        batch
    };
    // NOTE: most data is read properly typed already, however columns added via
    // literals in the transformations may need to be cast to the physical expected type.
    let result = if result.schema_ref().eq(&scan_plan.contract.result_schema) {
        result
    } else {
        let adapted = schema_adapter.adapt(result)?;
        if adapted.schema_ref().eq(&scan_plan.contract.result_schema) {
            adapted
        } else {
            // The batch adapter targets a nullability-relaxed schema (see
            // `expr_adapter`). Restamp to the strict logical schema; this
            // validates actual data nulls rather than declared nullability.
            crate::kernel::cast_record_batch(
                &adapted,
                scan_plan.contract.result_schema.clone(),
                false,
                false,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?
        }
    };
    if let Some((arr, field)) = file_id_col {
        let arr = if arr.data_type() != field.data_type() {
            let options = CastOptions {
                safe: true,
                ..Default::default()
            };
            cast_with_options(arr.as_ref(), field.data_type(), &options)?
        } else {
            arr
        };
        let mut columns = result.columns().to_vec();
        columns.push(arr);
        let mut fields = result.schema().fields().to_vec();
        fields.push(field);
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    } else {
        Ok(result)
    }
}

/// Caches a [`BatchAdapter`] for the most recently seen source schema, avoiding
/// repeated expression-tree construction when consecutive batches share the same
/// physical schema (the common case within a single file).
struct SchemaAdapter {
    factory: BatchAdapterFactory,
    /// Single-entry cache: the source schema for the currently cached adapter.
    cached_source: Option<SchemaRef>,
    cached_adapter: Option<BatchAdapter>,
}

impl SchemaAdapter {
    fn new(target_schema: SchemaRef) -> Self {
        Self {
            factory: BatchAdapterFactory::new(target_schema)
                .with_adapter_factory(Arc::new(DeltaPhysicalExprAdapterFactory)),
            cached_source: None,
            cached_adapter: None,
        }
    }

    /// Adapt the batch to the target schema, using a cached adapter when the
    /// source schema matches the previous call.
    fn adapt(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        let source_schema = batch.schema();
        let can_reuse = matches!(
            (&self.cached_source, &self.cached_adapter),
            (Some(cached_source), Some(_)) if cached_source.eq(&source_schema)
        );
        let needs_rebuild = !can_reuse;
        if needs_rebuild {
            let adapter = self.factory.make_adapter(&source_schema)?;
            self.cached_source = Some(source_schema);
            self.cached_adapter = Some(adapter);
        }
        match self.cached_adapter.as_ref() {
            Some(adapter) => adapter.adapt_batch(&batch),
            None => plan_err!(
                "schema adapter cache entry missing for source schema: {:?}",
                batch.schema()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::AsArray;
    use arrow_array::Array;
    use arrow_array::{
        BinaryArray, BinaryViewArray, Int32Array, Int64Array, RecordBatch, RecordBatchOptions,
        StringArray, StringViewArray, StructArray,
    };
    use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
    use datafusion::{
        common::ScalarValue,
        error::DataFusionError,
        physical_plan::collect,
        prelude::{col, lit},
    };
    use delta_kernel::FileMeta;
    use object_store::{ObjectStoreExt as _, memory::InMemory, path::Path};
    use parquet::arrow::ArrowWriter;
    use url::Url;

    use crate::{
        assert_batches_sorted_eq,
        delta_datafusion::{
            DeltaScanConfig, MissingSelectedFilePolicy,
            engine::DataFusionEngine,
            session::create_session,
            table_provider::next::{FILE_ID_COLUMN_DEFAULT, FileSelection},
        },
        kernel::Snapshot,
        test_utils::{TestResult, TestTables},
    };

    use super::{plan::build_parquet_predicate_schema, *};

    #[test]
    fn test_partitioned_files_to_file_groups_respects_dictionary_cardinality_limit() {
        let files = (0..=MAX_PARTITION_DICT_CARDINALITY)
            .map(|i| PartitionedFile::new(format!("memory:///f{i}.parquet"), 0))
            .collect_vec();

        let groups = partitioned_files_to_file_groups(files, 1);
        assert_eq!(groups.len(), 2);
        assert!(
            groups
                .iter()
                .all(|g| g.len() <= MAX_PARTITION_DICT_CARDINALITY)
        );
        let total: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total, MAX_PARTITION_DICT_CARDINALITY + 1);
    }

    #[test]
    fn test_partitioned_files_to_file_groups_spreads_across_target_partitions() {
        let files = (0..10)
            .map(|i| PartitionedFile::new(format!("memory:///f{i}.parquet"), 0))
            .collect_vec();

        let groups = partitioned_files_to_file_groups(files, 4);
        assert_eq!(groups.len(), 4);
        let sizes = groups.iter().map(|g| g.len()).collect_vec();
        assert_eq!(sizes, vec![3, 3, 2, 2]);

        // never more groups than files
        let groups = partitioned_files_to_file_groups(
            (0..2)
                .map(|i| PartitionedFile::new(format!("memory:///g{i}.parquet"), 0))
                .collect_vec(),
            8,
        );
        assert_eq!(groups.len(), 2);

        assert!(partitioned_files_to_file_groups(Vec::new(), 4).is_empty());
    }

    #[test]
    fn test_dv_rejects_row_group_index_beyond_i16() -> TestResult {
        use parquet::file::metadata::{FileMetaData, RowGroupMetaData};
        use parquet::schema::types::{SchemaDescriptor, Type};

        let schema = Arc::new(SchemaDescriptor::new(Arc::new(
            Type::group_type_builder("schema").build()?,
        )));
        let mut groups = (0..=i16::MAX)
            .map(|ordinal| {
                RowGroupMetaData::builder(Arc::clone(&schema))
                    .set_ordinal(ordinal)
                    .build()
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        groups.push(RowGroupMetaData::builder(Arc::clone(&schema)).build()?);
        let metadata =
            ParquetMetaData::new(FileMetaData::new(1, 0, None, None, schema, None), groups);

        let err = validate_dv_parquet_metadata(&metadata, None, 0, "f0", "file_id")
            .expect_err("row-group index must fit i16");
        assert!(err.to_string().contains("row-group index"));
        Ok(())
    }

    #[tokio::test]
    async fn test_resolve_empty_file_selection_does_not_poll_metadata_stream() -> TestResult {
        let log_store = TestTables::Simple.table_builder()?.build_storage()?;
        let snapshot = Snapshot::try_new(&log_store, None).await?;
        let scan_plan =
            KernelScanPlan::try_new(&snapshot, None, &[], &DeltaScanConfig::default(), None)?;
        let stream: ScanMetadataStream = Box::pin(futures::stream::poll_fn(|_| {
            panic!("unexpected metadata stream poll for empty file selection")
        }));

        let resolved = resolve_file_selection(
            &FileSelection::from_file_paths(Vec::<String>::new()),
            &scan_plan,
            stream,
        )
        .await?;

        assert!(resolved.active_file_ids.is_empty());
        assert!(resolved.missing_file_ids.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_resolved_file_selection_plan_does_not_poll_metadata_stream() -> TestResult {
        let log_store = TestTables::Simple.table_builder()?.build_storage()?;
        let snapshot = Snapshot::try_new(&log_store, None).await?;
        let scan_plan =
            KernelScanPlan::try_new(&snapshot, None, &[], &DeltaScanConfig::default(), None)?;
        let stream: ScanMetadataStream = Box::pin(futures::stream::poll_fn(|_| {
            panic!("unexpected metadata stream poll for empty file selection execution")
        }));
        let session = create_session().into_inner();
        let state = session.state();
        let engine = DataFusionEngine::new_from_session(&state);
        let selection = ResolvedFileSelection::new(
            std::collections::HashSet::new(),
            Vec::new(),
            MissingSelectedFilePolicy::Error,
        );

        let plan = execution_plan(
            &DeltaScanConfig::default(),
            &state,
            scan_plan,
            stream,
            engine,
            None,
            Some(&selection),
            None,
        )
        .await?;

        assert!(plan.is::<EmptyExec>());

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_uses_compact_internal_file_id_partition_values() -> TestResult {
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = create_session().into_inner();

        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");
        let data_source = exec.children()[0]
            .downcast_ref::<DataSourceExec>()
            .expect("expected DataSourceExec child");
        let (file_scan_config, _) = data_source
            .downcast_to_file_source::<ParquetSource>()
            .expect("expected parquet file source");

        let internal_file_ids = file_scan_config
            .file_groups
            .iter()
            .flat_map(|group| group.iter())
            .map(|file| {
                file.partition_values
                    .first()
                    .and_then(|value| value.try_as_str().flatten())
                    .expect("file-id partition value")
            })
            .collect_vec();

        assert!(
            !internal_file_ids.is_empty(),
            "test fixture should plan at least one file"
        );
        for internal_file_id in internal_file_ids {
            assert!(
                internal_file_id.len() <= 20,
                "internal file id should be compact, got {internal_file_id:?}"
            );
            assert!(
                !internal_file_id.contains('/'),
                "internal file id should not carry a full file path, got {internal_file_id:?}"
            );
        }

        Ok(())
    }

    /// Regression test for <https://github.com/delta-io/delta-rs/issues/4864>:
    /// the DataFusion session's `datafusion.execution.parquet.metadata_size_hint`
    /// must be applied to the scan's `ParquetSource` so the parquet reader prefetches
    /// the footer with a single hint-sized range read instead of first probing the
    /// footer length with a small (8-byte) request.
    #[tokio::test]
    async fn test_scan_applies_parquet_metadata_size_hint() -> TestResult {
        use crate::delta_datafusion::DeltaSessionContext;
        use crate::delta_datafusion::engine::AsObjectStoreUrl as _;
        use crate::test_utils::object_store::{
            RecordedObjectStoreOperation, RecordedPathKind, drain_recorded_object_store_operations,
            recording_log_store,
        };
        use futures::TryStreamExt as _;

        // The hint must be small enough to actually prefetch the footer of the single
        // data file written below, yet large enough that delta-rs's parquet footer cache
        // would already have to read past the tail of the file if it were consulted with
        // the file's object size (`tail_bytes = max(footer+8, 16KB)`), so observing an
        // end-of-file range read of exactly `METADATA_SIZE_HINT` bytes can only come
        // from the parquet metadata prefetch.
        const METADATA_SIZE_HINT: u64 = 4096;
        const NUM_ROWS: i64 = 5000;

        let arrow_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![Arc::new(Int64Array::from_iter_values(0..NUM_ROWS))],
        )?;
        // Disable compression so the data file is guaranteed to be larger than the
        // footer cache's minimum tail read (`max(footer+8, 16KB)`).
        let writer_properties = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::UNCOMPRESSED)
            .build();
        let table = crate::DeltaTable::new_in_memory()
            .write(vec![batch])
            .with_writer_properties(writer_properties)
            .await?;

        let (log_store, mut operations) = recording_log_store(table.log_store());

        let session =
            DeltaSessionContext::new_with_session_overrides(&std::collections::HashMap::from([(
                "datafusion.execution.parquet.metadata_size_hint".to_string(),
                METADATA_SIZE_HINT.to_string(),
            )]))?
            .into_inner();
        // Register the recording store for the table's root URL so data-file reads flow
        // through it (the scan looks up stores in the runtime env by the files' fully
        // qualified store URLs).
        session.state().runtime_env().register_object_store(
            log_store.root_url().as_object_store_url().as_ref(),
            log_store.object_store(),
        );

        let provider = table.table_provider().await?;
        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let batches = collect(scan, session.task_ctx()).await?;
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            NUM_ROWS as usize
        );

        // Sanity: every written parquet file must be larger than the hint, so a hinted
        // reader issues a bounded prefetch read of exactly `metadata_size_hint` bytes
        // ending at the end of the file.
        let parquet_sizes: Vec<u64> = log_store
            .object_store()
            .list(None)
            .try_filter_map(|meta| async move {
                Ok(meta
                    .location
                    .as_ref()
                    .ends_with(".parquet")
                    .then_some(meta.size))
            })
            .try_collect()
            .await?;
        assert!(!parquet_sizes.is_empty(), "expected written parquet files");
        assert!(
            parquet_sizes.iter().all(|&size| size > METADATA_SIZE_HINT),
            "test precondition: parquet files must be larger than the metadata size hint, got {parquet_sizes:?}"
        );
        // Sanity: delta-rs's parquet footer cache would need to read the last
        // `max(footer+8, 16KB)` bytes, so the files must be larger than that to keep the
        // prefetch-read assertion below unambiguous.
        const FOOTER_CACHE_MIN_TAIL_BYTES: u64 = 16 * 1024;
        assert!(
            parquet_sizes
                .iter()
                .all(|&size| size > FOOTER_CACHE_MIN_TAIL_BYTES),
            "test precondition: parquet files must be larger than the footer cache's \
             minimum tail read, got {parquet_sizes:?}"
        );

        let operations = drain_recorded_object_store_operations(&mut operations).await;
        let has_prefetch_read = operations.iter().any(|op| {
            let is_prefetch = |range: &std::ops::Range<u64>| {
                range.end - range.start == METADATA_SIZE_HINT && parquet_sizes.contains(&range.end)
            };
            match op {
                RecordedObjectStoreOperation::GetRange(
                    RecordedPathKind::Data | RecordedPathKind::Other,
                    range,
                ) => is_prefetch(range),
                RecordedObjectStoreOperation::GetRanges(
                    RecordedPathKind::Data | RecordedPathKind::Other,
                    ranges,
                ) => ranges.iter().any(is_prefetch),
                _ => false,
            }
        });
        assert!(
            has_prefetch_read,
            "scans must propagate datafusion.execution.parquet.metadata_size_hint to the \
             ParquetSource (https://github.com/delta-io/delta-rs/issues/4864); recorded operations: {operations:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_public_file_id_uses_file_path_with_compact_internal_ids() -> TestResult {
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = create_session().into_inner();

        session.register_table("delta_table", provider)?;

        let file_id_batches = session
            .sql("SELECT CAST(file_id AS STRING) AS file_id FROM delta_table LIMIT 1")
            .await?
            .collect()
            .await?;
        let file_id = file_id_batches[0].column(0).as_string_view().value(0);
        assert!(
            file_id.starts_with("file://") && file_id.ends_with(".parquet"),
            "public file id should remain a file path, got {file_id:?}"
        );

        let escaped_file_id = file_id.replace('\'', "''");
        let df = session
            .sql(&format!(
                "SELECT id FROM delta_table WHERE file_id = '{escaped_file_id}'"
            ))
            .await?;
        let filtered = df.collect().await?;

        assert!(filtered.iter().map(|batch| batch.num_rows()).sum::<usize>() > 0);
        assert!(filtered[0].schema().column_with_name("file_id").is_none());

        Ok(())
    }

    fn scan_file_context(file_url: &str) -> ScanFileContext {
        ScanFileContext {
            meta: FileMeta {
                location: Url::parse(file_url).expect("valid test URL"),
                last_modified: 0,
                size: 0,
            },
            transform: None,
            stats: Statistics::new_unknown(&Schema::empty()),
            num_records: None,
            partitions: None,
        }
    }

    #[test]
    fn test_remap_deletion_vectors_to_internal_file_ids_uses_compact_keys() -> TestResult {
        let files = vec![
            scan_file_context("s3://bucket/very/long/path/first.parquet"),
            scan_file_context("s3://bucket/very/long/path/second.parquet"),
        ];
        let mut dvs_by_url = HashMap::new();
        dvs_by_url.insert(files[1].meta.location.to_string(), vec![true, false, true]);

        let dvs = remap_deletion_vectors_to_internal_file_ids(&files, dvs_by_url)?;

        assert!(dvs.contains_key("1"));
        assert!(!dvs.contains_key(files[1].meta.location.as_str()));
        assert_eq!(
            dvs.get("1").expect("compact dv key").as_slice(),
            &[true, false, true]
        );
        Ok(())
    }

    #[test]
    fn test_remap_deletion_vectors_to_internal_file_ids_errors_for_unknown_url() {
        let files = vec![scan_file_context(
            "s3://bucket/very/long/path/first.parquet",
        )];
        let mut dvs_by_url = HashMap::new();
        dvs_by_url.insert(
            "s3://bucket/very/long/path/missing.parquet?X-Amz-Signature=secret-token".to_string(),
            vec![true],
        );

        let err = remap_deletion_vectors_to_internal_file_ids(&files, dvs_by_url).unwrap_err();
        let err = err.to_string();
        assert!(
            err.contains("missing internal file id mapping for file with deletion vector"),
            "unexpected error: {err}"
        );
        assert!(
            !err.contains("secret-token"),
            "error should redact URL query secrets: {err}"
        );
    }

    /// A `PartitionedFile` with exact min/max/null-count statistics for one
    /// Int64 sort column.
    fn stats_file(path: &str, min: i64, max: i64) -> PartitionedFile {
        let mut file = PartitionedFile::new(format!("{path}.parquet"), 100);
        file.statistics = Some(Arc::new(Statistics {
            num_rows: Precision::Exact(10),
            total_byte_size: Precision::Exact(100),
            column_statistics: vec![ColumnStatistics {
                null_count: Precision::Exact(0),
                min_value: Precision::Exact(ScalarValue::Int64(Some(min))),
                max_value: Precision::Exact(ScalarValue::Int64(Some(max))),
                ..Default::default()
            }],
        }));
        file
    }

    fn int64_sort_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("t", DataType::Int64, false)]))
    }

    fn int64_asc_ordering() -> LexOrdering {
        LexOrdering::new(vec![PhysicalSortExpr::new(
            Arc::new(Column::new("t", 0)),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )])
        .unwrap()
    }

    fn group_paths(group: &FileGroup) -> Vec<String> {
        file_names(group.iter())
    }

    fn file_names<'a>(files: impl IntoIterator<Item = &'a PartitionedFile>) -> Vec<String> {
        files
            .into_iter()
            .map(|file| file.object_meta.location.to_string())
            .collect()
    }

    #[test]
    fn test_split_file_groups_non_overlapping_files_form_contiguous_ranges() {
        // Non-overlapping ranges, deliberately out of order.
        let files = vec![
            stats_file("c", 200, 299),
            stats_file("a", 0, 99),
            stats_file("d", 300, 399),
            stats_file("b", 100, 199),
        ];

        let (groups, assumed) = split_file_groups_for_ordering(
            files,
            &int64_asc_ordering(),
            &int64_sort_schema(),
            2,
            OverlapPolicy::Prove,
        );
        assert!(!assumed);

        assert_eq!(groups.len(), 2);
        assert_eq!(group_paths(&groups[0]), vec!["a.parquet", "b.parquet"]);
        assert_eq!(group_paths(&groups[1]), vec!["c.parquet", "d.parquet"]);
    }

    #[test]
    fn test_split_file_groups_more_targets_than_files() {
        let files = vec![stats_file("a", 0, 99), stats_file("b", 100, 199)];

        let (groups, assumed) = split_file_groups_for_ordering(
            files,
            &int64_asc_ordering(),
            &int64_sort_schema(),
            8,
            OverlapPolicy::Prove,
        );
        assert!(!assumed);

        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_split_file_groups_overlapping_files() {
        // b overlaps a; the balanced grouping applies.
        let files = vec![
            stats_file("a", 0, 150),
            stats_file("b", 100, 199),
            stats_file("c", 200, 299),
        ];

        let (groups, assumed) = split_file_groups_for_ordering(
            files,
            &int64_asc_ordering(),
            &int64_sort_schema(),
            2,
            OverlapPolicy::Prove,
        );
        assert!(!assumed);

        // Files within each group must still be non-overlapping.
        assert!(groups.len() >= 2);
    }

    /// Bounds are folded once per distinct partition tuple; null counts accrue
    /// per file, and a null partition value leaves the column's bounds unknown.
    #[test]
    fn test_partition_stats_fold_bounds_once_and_null_counts_per_file() {
        let names = vec!["part".to_string(), "sub".to_string()];
        let rows = |n: usize| Statistics {
            num_rows: Precision::Exact(n),
            ..Statistics::new_unknown(&Schema::empty())
        };
        let a1 = vec![ScalarValue::from("A"), ScalarValue::Int64(Some(1))];
        let b_null = vec![ScalarValue::from("B"), ScalarValue::Null];
        let mut part_stats = HashMap::new();

        fold_partition_bounds(&names, &a1, &mut part_stats);
        add_partition_null_counts(&names, &a1, &rows(10), &mut part_stats);
        add_partition_null_counts(&names, &a1, &rows(5), &mut part_stats);
        fold_partition_bounds(&names, &b_null, &mut part_stats);
        add_partition_null_counts(&names, &b_null, &rows(3), &mut part_stats);

        let part = &part_stats["part"];
        assert_eq!(part.null_count, Precision::Exact(0));
        assert_eq!(part.min_value, Precision::Exact(ScalarValue::from("A")));
        assert_eq!(part.max_value, Precision::Exact(ScalarValue::from("B")));
        let sub = &part_stats["sub"];
        assert_eq!(sub.null_count, Precision::Exact(3));
        assert_eq!(sub.min_value, Precision::Absent);
        assert_eq!(sub.max_value, Precision::Absent);
    }

    #[test]
    fn test_arrange_non_overlapping_files_rejects_null_bounds() {
        // `b` carries typed-null bounds, as a file whose statistics omit the
        // sort column does once parsed against the table schema. Its real
        // rows could fall anywhere, so no order is non-overlapping.
        let mut unbounded = stats_file("b", 0, 0);
        unbounded.statistics = Some(Arc::new(Statistics {
            num_rows: Precision::Exact(10),
            total_byte_size: Precision::Exact(100),
            column_statistics: vec![ColumnStatistics {
                null_count: Precision::Exact(0),
                min_value: Precision::Exact(ScalarValue::Int64(None)),
                max_value: Precision::Exact(ScalarValue::Int64(None)),
                ..Default::default()
            }],
        }));
        let files = vec![stats_file("a", 0, 99), unbounded, stats_file("c", 100, 199)];

        let result = arrange_non_overlapping_files(
            files,
            |file| file,
            &int64_asc_ordering(),
            OverlapPolicy::Prove,
        );

        assert!(result.is_err());
    }

    /// A `PartitionedFile` with exact statistics for two Int64 sort columns,
    /// given as per-column `(min, max)` pairs.
    fn two_column_stats_file(path: &str, first: (i64, i64), second: (i64, i64)) -> PartitionedFile {
        let column = |(min, max): (i64, i64)| ColumnStatistics {
            null_count: Precision::Exact(0),
            min_value: Precision::Exact(ScalarValue::Int64(Some(min))),
            max_value: Precision::Exact(ScalarValue::Int64(Some(max))),
            ..Default::default()
        };
        let mut file = PartitionedFile::new(format!("{path}.parquet"), 100);
        file.statistics = Some(Arc::new(Statistics {
            num_rows: Precision::Exact(10),
            total_byte_size: Precision::Exact(100),
            column_statistics: vec![column(first), column(second)],
        }));
        file
    }

    fn two_column_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("date", DataType::Int64, false),
            Field::new("time", DataType::Int64, false),
        ]))
    }

    fn two_column_asc_ordering() -> LexOrdering {
        LexOrdering::new(vec![
            PhysicalSortExpr::new(
                Arc::new(Column::new("date", 0)),
                SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            ),
            PhysicalSortExpr::new(
                Arc::new(Column::new("time", 1)),
                SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            ),
        ])
        .expect("non-empty ordering")
    }

    fn assumed_order(files: Vec<PartitionedFile>, ordering: &LexOrdering) -> Option<Vec<String>> {
        arrange_non_overlapping_files(files, |file| file, ordering, OverlapPolicy::Assume)
            .ok()
            .map(|(ordered, _)| file_names(&ordered))
    }

    #[test]
    fn test_assumed_order_sorts_on_lexicographic_endpoints() {
        // Day-boundary files: per-column statistics overlap on `time`, so only
        // the assertion can place them. Deliberately out of order.
        let files = vec![
            two_column_stats_file("b", (1, 2), (0, 23)),
            two_column_stats_file("a", (0, 1), (0, 23)),
        ];

        assert_eq!(
            assumed_order(files.clone(), &two_column_asc_ordering()),
            Some(vec!["a.parquet".to_string(), "b.parquet".to_string()])
        );
        // The same files are refused when the order has to be proven.
        assert!(
            arrange_non_overlapping_files(
                files,
                |file| file,
                &two_column_asc_ordering(),
                OverlapPolicy::Prove
            )
            .is_err()
        );
    }

    #[test]
    fn test_assumed_order_breaks_equal_starts_on_the_end() {
        // Both files start at the same key, so only the end separates them.
        let files = vec![
            two_column_stats_file("wide", (5, 6), (0, 23)),
            two_column_stats_file("narrow", (5, 5), (0, 23)),
        ];

        assert_eq!(
            assumed_order(files, &two_column_asc_ordering()),
            Some(vec![
                "narrow.parquet".to_string(),
                "wide.parquet".to_string()
            ])
        );
    }

    /// Files meeting at a millisecond boundary, as the log reports them: `a`
    /// truly ends at 100ms, but a timestamp maximum landing exactly on a
    /// millisecond is read back rounded up to the next one. The widened bound
    /// makes the files overlap, which the proof refuses, but under an
    /// ascending order the files are placed on where they start, so the
    /// widened maximum cannot put them the wrong way round.
    #[test]
    fn test_assumed_order_places_files_meeting_at_a_widened_maximum() {
        let files = vec![stats_file("b", 100, 200), stats_file("a", 0, 101)];

        assert!(
            arrange_non_overlapping_files(
                files.clone(),
                |file| file,
                &int64_asc_ordering(),
                OverlapPolicy::Prove
            )
            .is_err()
        );
        assert_eq!(
            assumed_order(files, &int64_asc_ordering()),
            Some(vec!["a.parquet".to_string(), "b.parquet".to_string()])
        );
    }

    #[test]
    fn test_assumed_order_reverses_for_a_descending_ordering() {
        let files = vec![
            two_column_stats_file("a", (0, 1), (0, 23)),
            two_column_stats_file("b", (1, 2), (0, 23)),
        ];
        let descending = LexOrdering::new(vec![PhysicalSortExpr::new(
            Arc::new(Column::new("date", 0)),
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        )])
        .expect("non-empty ordering");

        assert_eq!(
            assumed_order(files, &descending),
            Some(vec!["b.parquet".to_string(), "a.parquet".to_string()])
        );
    }

    /// Identical bounds on a column the files span describe an overlap however
    /// the table is declared, so the files are refused rather than ordered
    /// arbitrarily.
    #[test]
    fn test_assumed_order_refuses_files_sharing_both_endpoints() {
        let files = vec![
            two_column_stats_file("a", (5, 5), (0, 23)),
            two_column_stats_file("b", (5, 5), (0, 23)),
        ];

        assert_eq!(assumed_order(files, &two_column_asc_ordering()), None);
    }

    /// A file sitting inside one value of the leading column, followed by one
    /// that crosses into the next value. Only the leading column's bounds are
    /// the file's first and last key: the crossing file's second-column
    /// minimum belongs to its *later* rows, so reading it would place that
    /// file first. The leading column separates them on its own.
    #[test]
    fn test_assumed_order_ignores_later_columns_while_the_leading_one_spans() {
        // a covers (5, 5)..(5, 10); b covers (5, 20)..(6, 1).
        let files = vec![
            two_column_stats_file("b", (5, 6), (1, 20)),
            two_column_stats_file("a", (5, 5), (5, 10)),
        ];

        assert_eq!(
            assumed_order(files, &two_column_asc_ordering()),
            Some(vec!["a.parquet".to_string(), "b.parquet".to_string()])
        );
    }

    /// Two files covering the same span of the leading column meet somewhere
    /// inside it, so nothing the later columns say can place them.
    #[test]
    fn test_assumed_order_refuses_files_spanning_the_same_leading_range() {
        let files = vec![
            two_column_stats_file("a", (5, 6), (0, 9)),
            two_column_stats_file("b", (5, 6), (10, 20)),
        ];

        assert_eq!(assumed_order(files, &two_column_asc_ordering()), None);
    }

    /// With the leading column pinned to one value in both files, the second
    /// column's bounds are their first and last keys and do place them.
    #[test]
    fn test_assumed_order_reads_the_second_column_under_a_constant_first() {
        let files = vec![
            two_column_stats_file("late", (5, 5), (10, 20)),
            two_column_stats_file("early", (5, 5), (0, 9)),
        ];

        assert_eq!(
            assumed_order(files, &two_column_asc_ordering()),
            Some(vec![
                "early.parquet".to_string(),
                "late.parquet".to_string()
            ])
        );
    }

    /// Two files pinned to the same single key hold nothing but that key, so
    /// either order is sorted and both are kept.
    #[test]
    fn test_assumed_order_keeps_files_sharing_one_constant_key() {
        let files = vec![
            two_column_stats_file("b", (5, 5), (7, 7)),
            two_column_stats_file("a", (5, 5), (7, 7)),
        ];

        assert_eq!(
            assumed_order(files, &two_column_asc_ordering()),
            Some(vec!["b.parquet".to_string(), "a.parquet".to_string()])
        );
    }

    /// A file with no statistics at all cannot be placed. One whose bounds are
    /// merely marked inexact can be: Delta marks every bound exact whether or
    /// not it was widened, so screening on the mark would catch nothing that
    /// matters while refusing files that are fine.
    #[test]
    fn test_assumed_order_places_inexact_bounds_but_not_missing_ones() {
        let mut inexact = stats_file("b", 100, 199);
        let stats = Arc::make_mut(inexact.statistics.as_mut().expect("statistics"));
        stats.column_statistics[0].min_value = Precision::Inexact(ScalarValue::Int64(Some(100)));

        assert_eq!(
            assumed_order(vec![stats_file("a", 0, 99), inexact], &int64_asc_ordering()),
            Some(vec!["a.parquet".to_string(), "b.parquet".to_string()])
        );
        assert_eq!(
            assumed_order(
                vec![
                    stats_file("a", 0, 99),
                    PartitionedFile::new("memory:///b.parquet", 100),
                ],
                &int64_asc_ordering()
            ),
            None
        );
    }

    #[test]
    fn test_split_file_groups_assuming_no_overlap_groups_unprovable_files() {
        let files = vec![
            two_column_stats_file("b", (1, 2), (0, 23)),
            two_column_stats_file("a", (0, 1), (0, 23)),
        ];

        // Without the assertion the files look overlapping and land in
        // separate groups; with it they form one ordered run.
        let (proven, assumed) = split_file_groups_for_ordering(
            files.clone(),
            &two_column_asc_ordering(),
            &two_column_schema(),
            1,
            OverlapPolicy::Prove,
        );
        assert!(!assumed);
        assert_eq!(proven.len(), 2);

        let (groups, assumed) = split_file_groups_for_ordering(
            files,
            &two_column_asc_ordering(),
            &two_column_schema(),
            1,
            OverlapPolicy::Assume,
        );
        assert!(assumed);
        assert_eq!(groups.len(), 1);
        assert_eq!(group_paths(&groups[0]), vec!["a.parquet", "b.parquet"]);
    }

    #[test]
    fn test_split_file_groups_assuming_no_overlap_falls_back_without_statistics() {
        let files = vec![
            PartitionedFile::new("memory:///a.parquet", 100),
            PartitionedFile::new("memory:///b.parquet", 100),
        ];

        let (groups, assumed) = split_file_groups_for_ordering(
            files.clone(),
            &int64_asc_ordering(),
            &int64_sort_schema(),
            2,
            OverlapPolicy::Assume,
        );

        assert!(!assumed, "the assertion must not be reported as applied");
        // Same grouping as an unordered scan: round robin across target partitions.
        let default_groups = partitioned_files_to_file_groups(files, 2);
        assert_eq!(
            groups.iter().map(group_paths).collect_vec(),
            default_groups.iter().map(group_paths).collect_vec()
        );
    }

    #[test]
    fn test_split_file_groups_missing_statistics_fall_back_to_default_grouping() {
        let files = vec![
            PartitionedFile::new("memory:///a.parquet", 100),
            PartitionedFile::new("memory:///b.parquet", 100),
        ];

        let (groups, assumed) = split_file_groups_for_ordering(
            files.clone(),
            &int64_asc_ordering(),
            &int64_sort_schema(),
            2,
            OverlapPolicy::Prove,
        );
        assert!(!assumed);

        // Same grouping as an unordered scan: round robin across target partitions.
        let default_groups = partitioned_files_to_file_groups(files, 2);
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups.iter().map(group_paths).collect_vec(),
            default_groups.iter().map(group_paths).collect_vec()
        );
    }

    #[test]
    fn test_chunk_ordered_files_hits_target_partition_count_exactly() {
        // A shortfall (e.g. ceil(100 / 24) = 5 → only 20 groups) makes
        // DataFusion repartition the scan, destroying the ordering claims.
        let files = (0..100)
            .map(|i| PartitionedFile::new(format!("f{i}.parquet"), 0))
            .collect_vec();

        let groups = chunk_ordered_files(files, 24);
        assert_eq!(groups.len(), 24);
        assert!(groups.iter().all(|group| (4..=5).contains(&group.len())));
        assert_eq!(groups.iter().map(FileGroup::len).sum::<usize>(), 100);
        // Contiguity: files stay in their original order across the groups.
        let flattened = groups
            .iter()
            .flat_map(|group| group.iter())
            .map(|file| file.object_meta.location.to_string())
            .collect_vec();
        assert_eq!(
            flattened,
            (0..100).map(|i| format!("f{i}.parquet")).collect_vec()
        );
    }

    #[test]
    fn test_chunk_ordered_files_empty_input_produces_no_groups() {
        let groups = chunk_ordered_files(Vec::new(), 4);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_chunk_ordered_files_respects_dictionary_cardinality_limit() {
        let files = (0..=MAX_PARTITION_DICT_CARDINALITY)
            .map(|i| PartitionedFile::new(format!("memory:///f{i}.parquet"), 0))
            .collect_vec();

        let groups = chunk_ordered_files(files, 1);
        assert_eq!(groups.len(), 2);
        assert!(
            groups
                .iter()
                .all(|group| group.len() <= MAX_PARTITION_DICT_CARDINALITY)
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "row indexes require whole file ownership")]
    fn test_partitioned_files_to_file_groups_rejects_split_file_across_groups_in_debug() {
        let files = vec![
            PartitionedFile::new("memory:///same.parquet", 0),
            PartitionedFile::new("memory:///other.parquet", 0),
            PartitionedFile::new("memory:///same.parquet", 0),
        ];

        let _ = partitioned_files_to_file_groups_with_limit(files, 1, 1);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_pads_short_mask_with_true() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let actual = normalize_dv_keep_mask_for_api(vec![true, false], Some(4), &url).unwrap();
        assert_eq!(actual, vec![true, false, true, true]);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_keeps_equal_length_mask() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let mask = vec![true, false, true];
        let actual = normalize_dv_keep_mask_for_api(mask.clone(), Some(3), &url).unwrap();
        assert_eq!(actual, mask);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_pads_empty_mask_to_all_true() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let actual = normalize_dv_keep_mask_for_api(Vec::new(), Some(3), &url).unwrap();
        assert_eq!(actual, vec![true, true, true]);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_mask_longer_than_num_records() {
        let url =
            Url::parse("s3://user:secret@example.com/table/file.parquet?sig=token#frag").unwrap();
        let expected_url = super::super::redact_url_for_error(&url);
        let err = normalize_dv_keep_mask_for_api(vec![true, false, true], Some(2), &url)
            .expect_err("longer mask should error");
        let message = err.to_string();
        assert!(message.contains("exceeds numRecords"));
        assert!(message.contains(&expected_url));
        assert!(!message.contains("sig=token"));
        assert!(!message.contains("secret"));
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_num_records_missing() {
        let url =
            Url::parse("s3://user:secret@example.com/table/file.parquet?sig=token#frag").unwrap();
        let expected_url = super::super::redact_url_for_error(&url);
        let err = normalize_dv_keep_mask_for_api(vec![true], None, &url)
            .expect_err("missing numRecords should error");
        let message = err.to_string();
        assert!(message.contains("Missing numRecords"));
        assert!(message.contains(&expected_url));
        assert!(!message.contains("sig=token"));
        assert!(!message.contains("secret"));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_num_records_overflow_usize() {
        // This branch is only reachable on 32-bit targets where u64 may exceed usize.
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let overflow_num_records = (usize::MAX as u64) + 1;
        let err = normalize_dv_keep_mask_for_api(vec![true], Some(overflow_num_records), &url)
            .expect_err("numRecords that does not fit usize should error");
        assert!(err.to_string().contains("does not fit usize"));
    }

    #[test]
    fn test_schema_adapter_synthesizes_nullable_columns() {
        let source_schema = Arc::new(Schema::new(Fields::empty()));
        let source = RecordBatch::try_new_with_options(
            source_schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(2)),
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema.clone());
        let adapted = adapter.adapt(source).unwrap();

        assert_eq!(adapted.schema().as_ref(), target_schema.as_ref());
        assert_eq!(adapted.num_rows(), 2);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id.null_count(), 2);
    }

    #[test]
    fn test_schema_adapter_missing_non_nullable_column_errors() {
        let source_schema = Arc::new(Schema::new(Fields::empty()));
        let source = RecordBatch::try_new_with_options(
            source_schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("missing non-nullable columns should error");
        match err {
            DataFusionError::Execution(msg) => {
                assert!(
                    msg.contains("Non-nullable column 'id'"),
                    "expected non-nullable missing-column error, got: {msg}"
                );
                assert!(
                    msg.contains("missing from the physical schema"),
                    "expected missing physical schema detail, got: {msg}"
                );
            }
            other => {
                panic!("expected execution error for missing non-nullable column, got: {other}")
            }
        }
    }

    #[test]
    fn test_schema_adapter_invalid_scalar_cast_errors() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, true)]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![Arc::new(StringArray::from(vec![Some("not-an-int")]))],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("invalid value cast should fail under DataFusion default cast semantics");
        match err {
            DataFusionError::ArrowError(inner, _) => {
                assert!(
                    matches!(inner.as_ref(), ArrowError::CastError(_)),
                    "expected arrow cast error, got: {inner}"
                );
            }
            other => panic!("expected arrow cast error for invalid scalar cast, got: {other}"),
        }
    }

    #[test]
    fn test_schema_adapter_type_widening() {
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let mut adapter = SchemaAdapter::new(target_schema.clone());
        let adapted = adapter.adapt(source).unwrap();

        assert_eq!(adapted.schema().as_ref(), target_schema.as_ref());
        assert_eq!(adapted.num_rows(), 3);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id.values(), &[1i64, 2, 3]);
    }

    #[test]
    fn test_schema_adapter_overflow_cast_errors() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![Arc::new(Int64Array::from(vec![i64::from(i32::MAX) + 1]))],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("overflow cast should fail under DataFusion default cast semantics");
        match err {
            DataFusionError::ArrowError(inner, _) => {
                assert!(
                    matches!(inner.as_ref(), ArrowError::CastError(_)),
                    "expected arrow cast error, got: {inner}"
                );
            }
            other => panic!("expected arrow cast error for overflow cast, got: {other}"),
        }
    }

    #[test]
    fn test_schema_adapter_caches_across_calls() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let mut adapter = SchemaAdapter::new(target_schema);

        let batch1 = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(Int32Array::from(vec![2]))],
        )
        .unwrap();

        let _ = adapter.adapt(batch1).unwrap();
        assert!(adapter.cached_source.is_some());

        // Second call with the same schema should hit the cache (no rebuild).
        let adapted = adapter.adapt(batch2).unwrap();
        assert_eq!(adapted.num_rows(), 1);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id.values(), &[2i64]);
    }

    #[tokio::test]
    async fn test_parquet_plan() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(store_url.as_object_store_url(), vec![file], false, None)];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | memory:///test_data.parquet |",
            "| 2  | b     | memory:///test_data.parquet |",
            "| 3  | c     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        // respect limits
        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            Some(1),
            &file_id_field,
            None,
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        // extended schema with missing column
        let arrow_schema_extended = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
            Field::new("value2", DataType::Utf8, true),
        ]));
        let parquet_predicate_schema_extended =
            build_parquet_predicate_schema(&arrow_schema_extended, &file_id_field);
        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema_extended,
            &parquet_predicate_schema_extended,
            Some(1),
            &file_id_field,
            None,
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+--------+-----------------------------+",
            "| id | value | value2 | __delta_rs_file_id__        |",
            "+----+-------+--------+-----------------------------+",
            "| 1  | a     |        | memory:///test_data.parquet |",
            "+----+-------+--------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_nested() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let nested_fields: Fields = vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
        ]
        .into();
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("nested", DataType::Struct(nested_fields.clone()), true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StructArray::try_new(
                    nested_fields,
                    vec![
                        Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
                        Arc::new(StringArray::from(vec![Some("aa"), Some("bb"), Some("cc")])),
                    ],
                    None,
                )?),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(store_url.as_object_store_url(), vec![file], false, None)];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+---------------+-----------------------------+",
            "| id | nested        | __delta_rs_file_id__        |",
            "+----+---------------+-----------------------------+",
            "| 1  | {a: a, b: aa} | memory:///test_data.parquet |",
            "| 2  | {a: b, b: bb} | memory:///test_data.parquet |",
            "| 3  | {a: c, b: cc} | memory:///test_data.parquet |",
            "+----+---------------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        let nested_fields_extended: Fields = vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Utf8, true),
        ]
        .into();
        let arrow_schema_extended = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "nested",
                DataType::Struct(nested_fields_extended.clone()),
                true,
            ),
        ]));
        let parquet_predicate_schema_extended =
            build_parquet_predicate_schema(&arrow_schema_extended, &file_id_field);
        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema_extended,
            &parquet_predicate_schema_extended,
            None,
            &file_id_field,
            None,
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+--------------------+-----------------------------+",
            "| id | nested             | __delta_rs_file_id__        |",
            "+----+--------------------+-----------------------------+",
            "| 1  | {a: a, b: aa, c: } | memory:///test_data.parquet |",
            "| 2  | {a: b, b: bb, c: } | memory:///test_data.parquet |",
            "| 3  | {a: c, b: cc, c: } | memory:///test_data.parquet |",
            "+----+--------------------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    /// Reading a file that contains actual null values in a nested field declared
    /// non-nullable in the table schema must fail: the nullability relaxation in
    /// the expression adapter only accommodates Spark-style relaxed on-disk
    /// schemas, while the final `cast_record_batch` in
    /// `finalize_transformed_batch` validates the actual data.
    #[tokio::test]
    async fn test_scan_rejects_actual_nested_nulls() -> TestResult {
        let store_url = Url::parse("memory:///")?;
        let log_store =
            crate::logstore::logstore_for(&store_url, crate::logstore::StorageConfig::default())?;

        // Delta log schema: metadata_col.int_id is NOT nullable.
        let meta_type = crate::kernel::StructType::try_new(vec![crate::kernel::StructField::new(
            "int_id",
            crate::kernel::DataType::Primitive(crate::kernel::PrimitiveType::String),
            false,
        )])?;
        let table = crate::operations::create::CreateBuilder::new()
            .with_log_store(log_store.clone())
            .with_columns(vec![crate::kernel::StructField::new(
                "metadata_col",
                crate::kernel::DataType::Struct(Box::new(meta_type)),
                true,
            )])
            .await?;

        // Parquet file whose data contains an actual null in int_id, registered
        // via a raw Add action to bypass the delta-rs writer.
        let meta_fields = Fields::from(vec![Field::new("int_id", DataType::Utf8, true)]);
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "metadata_col",
            DataType::Struct(meta_fields.clone()),
            true,
        )]));
        let meta = StructArray::try_new(
            meta_fields,
            vec![Arc::new(StringArray::from(vec![Some("t1"), None])) as ArrayRef],
            None,
        )?;
        let batch = RecordBatch::try_new(file_schema.clone(), vec![Arc::new(meta)])?;

        let mut buffer = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buffer, file_schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        let size = buffer.len() as i64;

        let store = log_store.object_store();
        store
            .put(&Path::from("part-00000.parquet"), buffer.into())
            .await?;

        crate::kernel::transaction::CommitBuilder::default()
            .with_actions(vec![crate::kernel::Action::Add(crate::kernel::Add {
                path: "part-00000.parquet".to_string(),
                size,
                data_change: true,
                ..Default::default()
            })])
            .build(
                Some(table.snapshot()?),
                log_store.clone(),
                crate::protocol::DeltaOperation::Write {
                    mode: crate::protocol::SaveMode::Append,
                    partition_by: None,
                    predicate: None,
                },
            )
            .await?;

        let snapshot = Snapshot::try_new(&log_store, None).await?;
        let provider = crate::delta_datafusion::table_provider::next::DeltaScan::builder()
            .with_snapshot(snapshot)
            .with_log_store(log_store)
            .await?;

        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store);
        session.register_table("delta_table", provider)?;

        let result = session
            .sql("SELECT * FROM delta_table")
            .await?
            .collect()
            .await;
        assert!(
            result.is_err(),
            "expected error when reading actual nulls in a non-nullable nested field, got: {result:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_multiple_stores() -> TestResult {
        let store_1 = Arc::new(InMemory::new());
        let store_url_1 = Url::parse("first:///")?;
        let store_2 = Arc::new(InMemory::new());
        let store_url_2 = Url::parse("second:///")?;

        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url_1, store_1.clone());
        session
            .runtime_env()
            .register_object_store(&store_url_2, store_2.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));

        let data_1 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(StringArray::from(vec![Some("a")])),
            ],
        )?;
        let data_2 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2])),
                Arc::new(StringArray::from(vec![Some("b")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data_1)?;
        arrow_writer.close()?;
        let path = Path::from("test_data.parquet");
        store_1.put(&path, buffer.into()).await?;
        let mut file_1: PartitionedFile = store_1.head(&path).await?.into();
        file_1
            .partition_values
            .push(wrap_file_id_value("first:///test_data.parquet"));

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data_2)?;
        arrow_writer.close()?;
        let path = Path::from("test_data.parquet");
        store_2.put(&path, buffer.into()).await?;
        let mut file_2: PartitionedFile = store_2.head(&path).await?.into();
        file_2
            .partition_values
            .push(wrap_file_id_value("second:///test_data.parquet"));

        let files_by_store = vec![
            (store_url_1.as_object_store_url(), vec![file_1], false, None),
            (store_url_2.as_object_store_url(), vec![file_2], false, None),
        ];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | first:///test_data.parquet  |",
            "| 2  | b     | second:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    /// The file groups of every parquet scan in a plan, in plan order.
    fn plan_file_groups(plan: &Arc<dyn ExecutionPlan>) -> Vec<Vec<String>> {
        let mut groups = Vec::new();
        if let Some(source) = plan.downcast_ref::<DataSourceExec>()
            && let Some(config) = source
                .data_source()
                .as_ref()
                .downcast_ref::<FileScanConfig>()
        {
            groups.extend(config.file_groups.iter().map(group_paths));
        }
        for child in plan.children() {
            groups.extend(plan_file_groups(child));
        }
        groups
    }

    async fn read_plan_file_groups(
        state: &dyn Session,
        files_by_store: Vec<FilesByStore>,
        parquet_read_schema: &SchemaRef,
        file_id_field: &FieldRef,
        overlap: OverlapPolicy,
    ) -> TestResult<(Vec<Vec<String>>, Option<LexOrdering>)> {
        let parquet_predicate_schema =
            build_parquet_predicate_schema(parquet_read_schema, file_id_field);
        let (plan, assumed_ordering) = get_read_plan(
            state,
            files_by_store,
            parquet_read_schema,
            &parquet_predicate_schema,
            None,
            file_id_field,
            None,
            None,
            Some(&int64_asc_ordering()),
            overlap,
        )
        .await?;
        Ok((plan_file_groups(&plan), assumed_ordering))
    }

    /// Files spanning several stores are grouped on their statistics alone,
    /// whatever the table asserts: the per-store plans are unioned, so an
    /// assumed arrangement could not be declared for the result and taking one
    /// would only displace the grouping DataFusion can certify by itself.
    #[tokio::test]
    async fn test_multi_store_read_plan_ignores_no_overlap_assertion() -> TestResult {
        let store_url_1 = Url::parse("first:///")?;
        let store_url_2 = Url::parse("second:///")?;
        let mut state = create_session().into_inner().state();
        // Two groups per store, so an assumed arrangement and a proven one cut
        // the same files differently.
        state.config_mut().options_mut().execution.target_partitions = 2;
        state
            .runtime_env()
            .register_object_store(&store_url_1, Arc::new(InMemory::new()));
        state
            .runtime_env()
            .register_object_store(&store_url_2, Arc::new(InMemory::new()));

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_read_schema = int64_sort_schema();

        // Ranges that overlap their neighbours, so no order can be proven and
        // the assertion would otherwise place the files itself.
        let store_files = |store: &str| {
            [(0, 99), (50, 149), (100, 199), (150, 249)]
                .into_iter()
                .enumerate()
                .map(|(index, (min, max))| {
                    let name = format!("{store}-f{index}");
                    let mut file = stats_file(&name, min, max);
                    file.partition_values
                        .push(wrap_file_id_value(format!("{store}:///{name}.parquet")));
                    file
                })
                .collect_vec()
        };
        let one_store = vec![(
            store_url_1.as_object_store_url(),
            store_files("first"),
            false,
            None,
        )];
        let two_stores = vec![
            (
                store_url_1.as_object_store_url(),
                store_files("first"),
                false,
                None,
            ),
            (
                store_url_2.as_object_store_url(),
                store_files("second"),
                false,
                None,
            ),
        ];

        // These files are ones the assertion really does regroup, so the
        // multi-store check below cannot pass for want of a difference.
        let (proven, _) = read_plan_file_groups(
            &state,
            one_store.clone(),
            &parquet_read_schema,
            &file_id_field,
            OverlapPolicy::Prove,
        )
        .await?;
        let (asserted, assumed_ordering) = read_plan_file_groups(
            &state,
            one_store,
            &parquet_read_schema,
            &file_id_field,
            OverlapPolicy::Assume,
        )
        .await?;
        assert_ne!(proven, asserted);
        assert!(assumed_ordering.is_some());

        let (proven, _) = read_plan_file_groups(
            &state,
            two_stores.clone(),
            &parquet_read_schema,
            &file_id_field,
            OverlapPolicy::Prove,
        )
        .await?;
        let (asserted, assumed_ordering) = read_plan_file_groups(
            &state,
            two_stores,
            &parquet_read_schema,
            &file_id_field,
            OverlapPolicy::Assume,
        )
        .await?;
        assert_eq!(proven, asserted);
        // And with nothing arranged on the assertion, nothing is left for the
        // exec to re-declare.
        assert!(assumed_ordering.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_predicate() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(store_url.as_object_store_url(), vec![file], false, None)];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let predicate = col("id").eq(lit(2i32));
        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 2  | b     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_skips_pushdown_when_logical_rewrite_fails() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let parquet_read_schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let logical_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("missing", DataType::Int32, false),
        ]));
        let data = RecordBatch::try_new(
            parquet_read_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer =
            ArrowWriter::try_new(&mut buffer, parquet_read_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_rewrite_failure.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_rewrite_failure.parquet"));

        let files_by_store = vec![(store_url.as_object_store_url(), vec![file], false, None)];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&logical_schema, &file_id_field);
        let predicate = col("missing").eq(lit(1i32));

        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+----------------------------------------+",
            "| id | __delta_rs_file_id__                   |",
            "+----+----------------------------------------+",
            "| 1  | memory:///test_rewrite_failure.parquet |",
            "| 2  | memory:///test_rewrite_failure.parquet |",
            "| 3  | memory:///test_rewrite_failure.parquet |",
            "+----+----------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_view_literal_against_base_parquet_file() -> TestResult {
        use datafusion::scalar::ScalarValue;

        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        // Write a Parquet file with base types, but read it with a view-typed schema.
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_view_literal.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_view_literal.parquet"));

        let files_by_store = vec![(store_url.as_object_store_url(), vec![file], false, None)];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("name").eq(lit(ScalarValue::Utf8View(Some("bob".to_string()))));
        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        let expected = vec![
            "+----+------+-------------------------------------+",
            "| id | name | __delta_rs_file_id__                |",
            "+----+------+-------------------------------------+",
            "| 2  | bob  | memory:///test_view_literal.parquet |",
            "+----+------+-------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_sql_literal_against_view_schema() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        // Write a Parquet file with base types, but read it with a view-typed schema.
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_sql_literal.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_sql_literal.parquet"));

        let files_by_store = vec![(store_url.as_object_store_url(), vec![file], false, None)];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("name").eq(lit("bob"));
        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        let expected = vec![
            "+----+------+------------------------------------+",
            "| id | name | __delta_rs_file_id__               |",
            "+----+------+------------------------------------+",
            "| 2  | bob  | memory:///test_sql_literal.parquet |",
            "+----+------+------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_physical_column_mapping_names() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let physical_name = "col-3877fd94-0973-4941-ac6b-646849a1ff65";
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(physical_name, DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(physical_name, DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_column_mapping_pushdown.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values.push(wrap_file_id_value(
            "memory:///test_column_mapping_pushdown.parquet",
        ));

        let files_by_store = vec![(store_url.as_object_store_url(), vec![file], false, None)];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col(physical_name).eq(lit("bob"));
        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[0].num_columns(), 3);

        let id_col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 2);

        let name_col = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        assert_eq!(name_col.value(0), "bob");

        assert_eq!(batches[0].schema().field(1).name(), physical_name);
        assert_eq!(batches[0].schema().field(2).name(), FILE_ID_COLUMN_DEFAULT);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_binaryview_literal_against_base_parquet_file()
    -> TestResult {
        use datafusion::scalar::ScalarValue;

        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Binary, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::BinaryView, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(BinaryArray::from_opt_vec(vec![
                    Some(b"aaa".as_slice()),
                    Some(b"bbb".as_slice()),
                    Some(b"ccc".as_slice()),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_binary_view.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_binary_view.parquet"));

        let files_by_store = vec![(store_url.as_object_store_url(), vec![file], false, None)];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("data").eq(lit(ScalarValue::BinaryView(Some(b"bbb".to_vec()))));
        let (plan, _) = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
            None,
            OverlapPolicy::Prove,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let id_col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 2);

        let data_col = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        assert_eq!(data_col.value(0), b"bbb");

        assert_eq!(batches[0].num_columns(), 3);
        assert_eq!(batches[0].schema().field(2).name(), FILE_ID_COLUMN_DEFAULT);

        Ok(())
    }
}
