use zarrs::array::{Array, ArrayBytes};

use crate::ffi::*;
use crate::util::*;

pub(crate) fn write(rhs: &[MxArray]) -> Result<()> {
    // rhs is either [store, data] (write at origin) or [store, bbox, data].
    if rhs.len() != 2 && rhs.len() != 3 {
        return Err(format!(
            "Invalid number of input arguments. Expected 2 or 3, got {}",
            rhs.len()
        ));
    }

    let store_str = rhs[0];
    let data_arr = if rhs.len() == 3 { rhs[2] } else { rhs[1] };

    let path = mx_array_to_str(store_str)?;
    let store = create_writable_store(path)?;
    let array = zarrs_result_to_str_error(Array::open(store, "/"), "Error while opening array")?;

    let array_shape = array.shape();
    let ndim = array_shape.len();
    let data_type = array.data_type();
    let type_size = if let Some(type_size) = data_type.fixed_size() {
        type_size
    } else {
        return Err(format!(
            "Unsupported data type {:?} in Zarr array.",
            data_type
        ));
    };

    // build shape: no bbox arg means "write at origin", with the region taken
    // from the data array's own dimensions (trailing singleton dims trimmed to
    // the array's dimensionality, matching MATLAB's size() semantics).
    let bbox = if rhs.len() == 3 {
        mx_array_to_bbox(rhs[1], ndim)?
    } else {
        let data_dims = mx_array_size_to_usize_slice(data_arr);
        let k = data_dims.len().min(ndim);
        let shape: Vec<u64> = data_dims[..k].iter().map(|&d| d as u64).collect();
        BBox::new(vec![0u64; k], shape)
    };
    bbox.check_bounds(array_shape)?;
    let subset = bbox.to_subset()?;

    // prepare data for write
    let array_class = zarrs_data_type_to_mx(&array.data_type())?;
    let data_class = unsafe { mxGetClassID(data_arr) };

    if array_class != data_class {
        return Err(format!(
            "Data type mismatch. Expected {:?} from Zarr array, got {:?} from input.",
            array_class, data_class
        ));
    }

    let data_bytes = copy_as_c_order(data_arr, &bbox.shape, type_size)?;

    // write data (wrap raw bytes so they are stored as-is, not reinterpreted as
    // typed `u8` elements)
    zarrs_result_to_str_error(
        array.store_array_subset(&subset, ArrayBytes::new_flen(data_bytes)),
        "Error while writing data",
    )?;

    Ok(())
}
