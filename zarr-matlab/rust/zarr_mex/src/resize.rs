use zarrs::array::{Array, ArrayMetadataOptions};

use crate::ffi::*;
use crate::util::*;

pub(crate) fn resize(rhs: &[MxArray]) -> Result<()> {
    if rhs.len() != 2 {
        return Err(format!(
            "Invalid number of input arguments. Expected 2 (path, shape), got {}",
            rhs.len()
        ));
    }

    let store_str = rhs[0];
    let shape_arr = rhs[1];

    let path = mx_array_to_str(store_str)?;

    // Parse shape from MATLAB array
    let shape_f64 = mx_array_to_f64_slice(shape_arr)?;
    let new_shape: Vec<u64> = shape_f64
        .iter()
        .map(|&x| as_nat(x))
        .collect::<Result<Vec<u64>>>()?;

    // Open the array
    let store = create_writable_store(path)?;
    let mut array = zarrs_result_to_str_error(Array::open(store, "/"))?;

    // Resize the array
    zarrs_result_to_str_error(array.set_shape(new_shape))?;

    // Store the updated metadata
    zarrs_result_to_str_error(
        array.store_metadata_opt(
            &ArrayMetadataOptions::default().with_include_zarrs_metadata(false),
        ),
    )?;

    Ok(())
}
