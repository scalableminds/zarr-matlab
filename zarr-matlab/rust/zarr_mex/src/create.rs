use zarrs::array::{Array, ArrayMetadata, ArrayMetadataOptions, ArrayMetadataV2, ArrayMetadataV3};

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

    // Parse the JSON as either Zarr V3 or Zarr V2 array metadata, dispatching on
    // `zarr_format`. `store_metadata_opt` below then writes `zarr.json` or
    // `.zarray`/`.zattrs` accordingly, since the default `ArrayMetadataOptions`
    // leave the metadata version unconverted.
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|err| format!("Failed to parse JSON metadata: {}", err))?;
    let metadata: ArrayMetadata = match value.get("zarr_format").and_then(serde_json::Value::as_u64)
    {
        Some(3) | None => ArrayMetadata::V3(
            serde_json::from_value::<ArrayMetadataV3>(value)
                .map_err(|err| format!("Failed to parse Zarr V3 array metadata: {}", err))?,
        ),
        Some(2) => ArrayMetadata::V2(
            serde_json::from_value::<ArrayMetadataV2>(value)
                .map_err(|err| format!("Failed to parse Zarr V2 array metadata: {}", err))?,
        ),
        Some(other) => {
            return Err(format!(
                "Unsupported zarr_format {}. Expected 2 or 3.",
                other
            ))
        }
    };

    let store = create_writable_store(path)?;

    // Create the array with the provided metadata
    let array = zarrs_result_to_str_error(
        Array::new_with_metadata(store, "/", metadata),
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
