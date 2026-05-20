//! KMS-based Parquet encryption utilities intended for integration tests and examples.
//!
//! This module is **not part of the stable public API**. It lives in `test_utils` and is
//! compiled in non-test builds only to allow downstream integration-test crates to depend on it
//! without pulling in a separate crate. Do not rely on it for production use.
//!
//! # Key types
//!
//! - [`TableEncryption`]: bundles an [`EncryptionFactory`] with its
//!   [`EncryptionFactoryOptions`], providing helpers to produce per-file
//!   [`WriterPropertiesBuilder`] values.
//! - [`KmsFileFormatOptions`]: a [`FileFormatOptions`](crate::table::file_format_options::FileFormatOptions)
//!   implementation that wires KMS-based encryption into [`DeltaTable`](crate::DeltaTable)
//!   operations. Pass it to [`DeltaTableBuilder::with_file_format_options`](crate::table::builder::DeltaTableBuilder::with_file_format_options)
//!   or [`DeltaOps::with_file_format_options`](crate::operations::DeltaOps::with_file_format_options).
//!   See `crates/deltalake/examples/basic_operations_encryption.rs` for a complete example.
//! - [`MockKmsClient`]: a test-only [`EncryptionFactory`] that generates a unique key per
//!   file and remembers it for decryption. Useful in unit and integration tests where a
//!   real KMS is not available.

use crate::table::file_format_options::{
    FileFormatOptions, TableOptions, WriterPropertiesFactory, WriterPropertiesFactoryRef,
};
use crate::{DeltaResult, crate_version};
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::config::{ConfigField, EncryptionFactoryOptions, ExtensionOptions};
use datafusion::execution::parquet_encryption::EncryptionFactory;
use object_store::path::Path;
use parquet::basic::Compression;
use parquet::encryption::decrypt::FileDecryptionProperties;
use parquet::encryption::encrypt::FileEncryptionProperties;
use parquet::file::properties::{WriterProperties, WriterPropertiesBuilder};
use parquet::schema::types::ColumnPath;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub type SchemaRef = Arc<ArrowSchema>;

#[derive(Clone, Debug)]
pub struct TableEncryption {
    encryption_factory: Arc<dyn EncryptionFactory>,
    configuration: EncryptionFactoryOptions,
}

impl TableEncryption {
    pub fn new(
        encryption_factory: Arc<dyn EncryptionFactory>,
        configuration: EncryptionFactoryOptions,
    ) -> Self {
        Self {
            encryption_factory,
            configuration,
        }
    }

    pub fn new_with_extension_options<T: ExtensionOptions>(
        encryption_factory: Arc<dyn EncryptionFactory>,
        options: &T,
    ) -> DeltaResult<Self> {
        let mut configuration = EncryptionFactoryOptions::default();
        for entry in options.entries() {
            if let Some(value) = &entry.value {
                configuration.set(&entry.key, value)?;
            }
        }
        Ok(Self {
            encryption_factory,
            configuration,
        })
    }

    pub fn encryption_factory(&self) -> &Arc<dyn EncryptionFactory> {
        &self.encryption_factory
    }

    pub fn configuration(&self) -> &EncryptionFactoryOptions {
        &self.configuration
    }

    pub async fn update_writer_properties(
        &self,
        mut builder: WriterPropertiesBuilder,
        file_path: &Path,
        file_schema: &SchemaRef,
    ) -> DeltaResult<WriterPropertiesBuilder> {
        let encryption_properties = self
            .encryption_factory
            .get_file_encryption_properties(&self.configuration, file_schema, file_path)
            .await?;
        if let Some(encryption_properties) = encryption_properties {
            builder = builder.with_file_encryption_properties(encryption_properties);
        }
        Ok(builder)
    }
}

// More advanced factory with KMS support
#[derive(Clone, Debug)]
pub struct KMSWriterPropertiesFactory {
    writer_properties: WriterProperties,
    encryption: Option<TableEncryption>,
}

impl KMSWriterPropertiesFactory {
    pub fn with_encryption(table_encryption: TableEncryption) -> Self {
        let writer_properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY) // Code assumes Snappy by default
            .set_created_by(format!("delta-rs version {}", crate_version()))
            .build();
        Self {
            writer_properties,
            encryption: Some(table_encryption),
        }
    }
}

#[async_trait]
impl WriterPropertiesFactory for KMSWriterPropertiesFactory {
    fn compression(&self, column_path: &ColumnPath) -> Compression {
        self.writer_properties.compression(column_path)
    }

    async fn create_writer_properties(
        &self,
        file_path: &Path,
        file_schema: &Arc<ArrowSchema>,
    ) -> DeltaResult<WriterProperties> {
        let mut builder: WriterPropertiesBuilder = self.writer_properties.clone().into();
        if let Some(encryption) = self.encryption.as_ref() {
            builder = encryption
                .update_writer_properties(builder, file_path, file_schema)
                .await?;
        }
        Ok(builder.build())
    }
}

// -------------------------------------------------------------------------------------------------
// FileFormatOptions for KMS encryption based on settings in TableEncryption
// -------------------------------------------------------------------------------------------------
pub struct KmsFileFormatOptions {
    table_encryption: TableEncryption,
    writer_properties_factory: WriterPropertiesFactoryRef,
    encryption_factory_id: String,
}

impl KmsFileFormatOptions {
    pub fn new(table_encryption: TableEncryption) -> Self {
        let encryption_factory_id = format!("delta-{}", Uuid::new_v4());
        let writer_properties_factory = Arc::new(KMSWriterPropertiesFactory::with_encryption(
            table_encryption.clone(),
        ));
        Self {
            table_encryption,
            writer_properties_factory,
            encryption_factory_id,
        }
    }
}

impl Debug for KmsFileFormatOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KmsFileFormatOptions")
            .finish_non_exhaustive()
    }
}

impl FileFormatOptions for KmsFileFormatOptions {
    fn table_options(&self) -> TableOptions {
        let mut table_options = TableOptions::default();
        table_options.parquet.crypto.factory_id = Some(self.encryption_factory_id.clone());
        table_options.parquet.crypto.factory_options =
            self.table_encryption.configuration().clone();
        table_options
    }

    fn writer_properties_factory(&self) -> WriterPropertiesFactoryRef {
        Arc::clone(&self.writer_properties_factory)
    }

    fn update_session(&self, session: &dyn Session) -> DeltaResult<()> {
        // Ensure DataFusion has the encryption factory registered
        session.runtime_env().register_parquet_encryption_factory(
            &self.encryption_factory_id,
            Arc::clone(self.table_encryption.encryption_factory()),
        );
        Ok(())
    }
}

// -------------------------------------------------------------------------------------------------
// Mock KMS client for testing purposes
// -------------------------------------------------------------------------------------------------

/// Mock encryption factory implementation for use in tests.
/// Generates unique encryption keys for each file and stores them for later decryption.
#[derive(Debug, Default)]
pub struct MockKmsClient {
    encryption_keys: Mutex<HashMap<Path, Vec<u8>>>,
    counter: AtomicU64,
}

impl MockKmsClient {
    pub fn new() -> Self {
        Self {
            encryption_keys: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl EncryptionFactory for MockKmsClient {
    async fn get_file_encryption_properties(
        &self,
        _config: &EncryptionFactoryOptions,
        _schema: &SchemaRef,
        file_path: &Path,
    ) -> datafusion::error::Result<Option<Arc<FileEncryptionProperties>>> {
        let file_idx = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&file_idx.to_le_bytes());
        let key = key.to_vec();
        let mut keys = self.encryption_keys.lock().unwrap();
        // Use just the filename as key to handle path prefix differences between write and read
        let filename = Path::from(file_path.filename().unwrap_or(file_path.as_ref()));
        keys.insert(filename, key.clone());
        let encryption_properties = FileEncryptionProperties::builder(key).build()?;
        Ok(Some(encryption_properties))
    }

    async fn get_file_decryption_properties(
        &self,
        _config: &EncryptionFactoryOptions,
        file_path: &Path,
    ) -> datafusion::error::Result<Option<Arc<FileDecryptionProperties>>> {
        let keys = self.encryption_keys.lock().unwrap();
        // Use just the filename as key to handle path prefix differences between write and read
        let filename = Path::from(file_path.filename().unwrap_or(file_path.as_ref()));
        let key = keys.get(&filename).ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(format!("No key for file {file_path:?}"))
        })?;
        let decryption_properties = FileDecryptionProperties::builder(key.clone()).build()?;
        Ok(Some(decryption_properties))
    }
}
