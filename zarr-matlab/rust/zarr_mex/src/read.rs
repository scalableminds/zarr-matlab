use zarrs::array::ArrayBytes;

use crate::ffi::*;
use crate::util::*;

pub(crate) fn read(rhs: &[MxArray]) -> Result<MxArrayMut> {
    // rhs is either [array] (whole array) or [array, bbox] (explicit region),
    // where `array` is a path string or a cached handle.
    if rhs.len() != 1 && rhs.len() != 2 {
        return Err(format!(
            "Invalid number of input arguments. Expected 1 or 2, got {}",
            rhs.len()
        ));
    }

    let arg = resolve_array_arg(rhs[0])?;

    arg.with(|array| {
        let array_shape = array.shape();
        let ndim = array_shape.len();
        let data_type = array.data_type();
        let type_size = data_type
            .fixed_size()
            .ok_or_else(|| format!("Unsupported data type {:?} in Zarr array.", data_type))?;

        // build shape: no bbox arg means "read the whole array"
        let bbox = if rhs.len() == 2 {
            mx_array_to_bbox(rhs[1], ndim)?
        } else {
            BBox::new(vec![0u64; ndim], array_shape.to_vec())
        };
        bbox.check_bounds(array_shape)?;
        let subset = bbox.to_subset()?;

        // prepare allocation
        let mat_class = zarrs_data_type_to_mx(array.data_type())?;

        // read data
        let data_all: ArrayBytes = array.retrieve_array_subset(&subset)?;
        // Borrow the decoded bytes (in c-order) directly instead of cloning them;
        // the transpose reads straight from this buffer into the MATLAB array.
        let zarr_buf = zarrs_result_to_str_error(
            data_all.into_fixed(),
            "Error while reading read data into buffer",
        )?; // Cow<[u8]>, in c-order

        let mat_arr = create_numeric_array(&bbox.shape, mat_class, MxComplexity::Real)?;

        copy_as_fortran_order(&zarr_buf, mat_arr, &bbox.shape, type_size)?;

        Ok(mat_arr)
    })
}
