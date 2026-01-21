use zarrs::array::Array;

use std::path::PathBuf;
use std::sync::Arc;

use crate::ffi::*;
use crate::util::*;

pub(crate) fn write(rhs: &[MxArray]) -> Result<()> {
    if rhs.len() != 3 {
        return Err(format!(
            "Invalid number of input arguments. Expected 3, got {}",
            rhs.len()
        ));
    }

    let store_str = rhs[0];
    let bbox_arr = rhs[1];
    let data_arr = rhs[2];

    let store_path: PathBuf = mx_array_to_str(store_str)?.into();

    println!("store_path: {:?}", store_path);

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
        return Err(format!(
            "Unsupported data type {:?} in Zarr array.",
            data_type
        ));
    };

    // build shape
    let bbox = mx_array_to_bbox(bbox_arr, ndim)?;
    bbox.check_bounds(array_shape)?;
    let subset = bbox.to_subset()?;

    println!("subset: {:?}", subset);

    // prepare data for write
    let array_class = zarrs_data_type_to_mx(&array.data_type())?;
    let data_class = unsafe { mxGetClassID(data_arr) };
    // if array_class != data_class {
    return Err(format!(
        "Data type mismatch. Expected {:?} from Zarr array, got {:?} from input.",
        array_class, data_class
    ));
    // }

    // println!("data_class: {:?}", data_class);

    // let data_bytes = copy_as_c_order(data_arr, &bbox.shape, type_size)?;

    // println!("c-order");

    // // write data
    // zarrs_result_to_str_error(array.store_array_subset(&subset, data_bytes))?;

    Ok(())
}
