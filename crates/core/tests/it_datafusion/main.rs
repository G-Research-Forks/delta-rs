#![cfg(feature = "datafusion")]

mod fs_common;

mod checkpoint_writer;
mod command_filesystem_check;
mod command_merge;
mod command_optimize;
mod command_restore;
mod command_vacuum;
mod commit_info_format;
mod datafusion_dat;
mod datafusion_table_provider;
mod dst_commits;
mod file_selection_bench_bridge;
#[cfg(feature = "float16")]
mod float16;
mod integration;
mod integration_checkpoint;
mod integration_datafusion;
#[cfg(feature = "nanosecond-time")]
mod nanosecond_time;
#[cfg(feature = "nanosecond-timestamps")]
mod nanosecond_timestamps;
mod operation_scope;
mod read_delta_log_test;
mod read_delta_partitions_test;
mod scan_file_statistics;
mod sort_order;
mod timestamp_statistics;
#[cfg(feature = "unsigned-integers")]
mod unsigned_integers;
