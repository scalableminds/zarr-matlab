// Adapter exposing `zarrs_http::HTTPStore`, which implements the
// `zarrs_storage` 0.4 traits, through the `zarrs_storage` 0.5 traits used by
// zarrs 0.24. The read-only trait surface is identical between the two
// versions; only the key, byte-range and error types need converting. Drop
// this once `zarrs_http` is released against `zarrs_storage` 0.5.

use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{MaybeBytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};
use zarrs_storage_04::byte_range::ByteRange as ByteRange04;
use zarrs_storage_04::{
    ReadableStorageTraits as ReadableStorageTraits04, StorageError as StorageError04,
    StoreKey as StoreKey04,
};

pub struct HttpStore(pub zarrs_http::HTTPStore);

fn key_04(key: &StoreKey) -> StoreKey04 {
    // Both versions validate keys identically, so a valid key always converts.
    StoreKey04::new(key.as_str()).expect("a valid store key")
}

fn byte_range_04(byte_range: ByteRange) -> ByteRange04 {
    match byte_range {
        ByteRange::FromStart(offset, length) => ByteRange04::FromStart(offset, length),
        ByteRange::Suffix(length) => ByteRange04::Suffix(length),
    }
}

fn error_05(err: StorageError04) -> StorageError {
    StorageError::Other(err.to_string())
}

impl ReadableStorageTraits for HttpStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(&key_04(key)).map_err(error_05)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let byte_ranges = Box::new(byte_ranges.map(byte_range_04));
        let bytes = self
            .0
            .get_partial_many(&key_04(key), byte_ranges)
            .map_err(error_05)?;
        Ok(bytes.map(|bytes| {
            Box::new(bytes.map(|b| b.map_err(error_05))) as Box<dyn Iterator<Item = _> + 'a>
        }))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(&key_04(key)).map_err(error_05)
    }

    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}
