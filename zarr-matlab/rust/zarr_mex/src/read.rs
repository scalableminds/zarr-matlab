use zarrs::array::Array;

use crate::ffi::*;
use crate::util::*;

pub(crate) fn read(rhs: &[MxArray]) -> Result<MxArrayMut> {
    if rhs.len() != 2 {
        return Err(format!(
            "Invalid number of input arguments. Expected 2, got {}",
            rhs.len()
        ));
    }

    let store_str = rhs[0];
    let bbox_arr = rhs[1];

    let path = mx_array_to_str(store_str)?;
    let store = create_readable_store(path)?;
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

    // build shape
    let bbox = mx_array_to_bbox(bbox_arr, ndim)?;
    bbox.check_bounds(array_shape)?;
    let subset = bbox.to_subset()?;

    // prepare allocation
    let mat_class = zarrs_data_type_to_mx(&array.data_type())?;

    // read data
    let data_all = zarrs_result_to_str_error(
        array.retrieve_array_subset(&subset),
        "Error while reading data from array",
    )?;
    let zarr_buf = zarrs_result_to_str_error(
        data_all.into_fixed(),
        "Error while reading read data into buffer",
    )?
    .into_owned(); // in c-order

    let mat_arr = create_numeric_array(&bbox.shape, mat_class, MxComplexity::Real)?;

    copy_as_fortran_order(&zarr_buf, mat_arr, &bbox.shape, type_size)?;

    Ok(mat_arr)
}
