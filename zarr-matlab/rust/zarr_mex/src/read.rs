use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;

use std::path::PathBuf;
use std::sync::Arc;

use crate::ffi::*;
use crate::util::*;

pub(crate) fn read(rhs: &[MxArray]) -> Result<MxArrayMut> {
    let (store_str, bbox_arr) = match rhs {
        [store_str, bbox_arr] => (store_str, bbox_arr),
        _ => {
            return Err(format!(
                "Invalid number of input arguments. Expected 2, got {}",
                rhs.len()
            ))
        }
    };

    let store_path: PathBuf = mx_array_to_str(rhs[0])?.into();

    let store: zarrs::storage::ReadableWritableListableStorage = Arc::new(
        zarrs_result_to_str_error(zarrs::filesystem::FilesystemStore::new(&store_path))?,
    );
    let array = zarrs_result_to_str_error(Array::open(store.clone(), "/"))?;

    let array_shape = array.shape();
    let ndim = array_shape.len();
    let data_type = array.data_type();
    let type_size = if let Some(type_size) = data_type.fixed_size() {
        type_size
    } else {
        return Err("Unsupported data type".to_string());
    };

    // build shape
    let bbox = mx_array_to_bbox(rhs[1], ndim)?;

    if bbox
        .start
        .iter()
        .zip(bbox.shape.iter())
        .zip(array_shape.iter())
        .any(|((bbox_min_x, bbox_shape_x), shape_x)| (*bbox_min_x + *bbox_shape_x) > *shape_x)
    {
        return Err(format!(
            "Bounding box {:?} is out of bounds for array of shape={:?}.",
            &bbox, array_shape
        ));
    }
    let subset = zarrs_result_to_str_error(ArraySubset::new_with_start_shape(
        bbox.start.clone(),
        bbox.shape.clone(),
    ))?;

    // prepare allocation
    let mat_class = zarrs_data_type_to_mx(&array.data_type())?;

    // read data
    let data_all = zarrs_result_to_str_error(array.retrieve_array_subset(&subset))?;
    let zarr_buf = zarrs_result_to_str_error(data_all.into_fixed())?.into_owned(); // in c-order

    let mat_arr = create_numeric_array(&bbox.shape, mat_class, MxComplexity::Real)?;

    copy_as_fortran_order(&zarr_buf, mat_arr, &bbox.shape, type_size)?;

    Ok(mat_arr)
}
