use zarrs::array::{Array, ArrayMetadataOptions, ArrayMetadataV3};

use crate::ffi::*;
use crate::util::*;

pub(crate) fn create(rhs: &[MxArray]) -> Result<()> {
    if rhs.len() != 2 {
        return Err(format!(
            "Invalid number of input arguments. Expected 2 (path, json), got {}",
            rhs.len()
        ));
    }

    let store_str = rhs[0];
    let json_str = rhs[1];

    let path = mx_array_to_str(store_str)?;
    let json = mx_array_to_str(json_str)?;

    // Parse JSON as ArrayMetadataV3
    let metadata = if let Ok(metadata) = serde_json::from_str::<ArrayMetadataV3>(&json) {
        metadata
    } else {
        return Err("Failed to parse JSON metadata".to_string());
    };

    let store = create_writable_store(path)?;

    // Create the array with the provided metadata
    let array = zarrs_result_to_str_error(
        Array::new_with_metadata(store, "/", metadata.into()),
        "Error while creating array",
    )?;

    // Store the metadata to disk
    zarrs_result_to_str_error(
        array.store_metadata_opt(
            &ArrayMetadataOptions::default().with_include_zarrs_metadata(false),
        ),
        "Error while writing array metadata",
    )?;

    Ok(())
}
