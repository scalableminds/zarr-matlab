use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView};

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
            .as_fixed()
            .map(|fixed| fixed.size())
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

        let mat_arr = create_numeric_array(&bbox.shape, mat_class, MxComplexity::Real)?;

        // Decode straight into the MATLAB array through a column-major view of
        // its buffer, so zarrs places each element at its F-order position and
        // no intermediate C-order buffer or transpose is needed.
        let mat_buf = mx_array_mut_to_u8_slice_mut(mat_arr)?;
        let f_strides = f_strides_of(&bbox.shape);
        // SAFETY: this is the only view of `mat_buf`, and distinct indices of a
        // dense column-major layout address distinct elements.
        let view = unsafe {
            ArrayBytesFixedDisjointView::new_strided(
                UnsafeCellSlice::new(mat_buf),
                type_size,
                0,
                &bbox.shape,
                &f_strides,
            )
        };
        let view = zarrs_result_to_str_error(view, "Error while creating output view")?;
        array.retrieve_array_subset_into(&subset, &mut ArrayBytesDecodeIntoTarget::Fixed(view))?;

        Ok(mat_arr)
    })
}
