use crate::ffi::*;
use crate::util::*;

pub(crate) fn resize(rhs: &[MxArray]) -> Result<()> {
    if rhs.len() != 2 {
        return Err(format!(
            "Invalid number of input arguments. Expected 2 (array, shape), got {}",
            rhs.len()
        ));
    }

    let shape_arr = rhs[1];

    // Parse shape from MATLAB array
    let shape_f64 = mx_array_to_f64_slice(shape_arr)?;
    let new_shape: Vec<u64> = shape_f64
        .iter()
        .map(|&x| as_nat(x))
        .collect::<Result<Vec<u64>>>()?;

    // Resize the array. For a cached handle this mutates the cached array in
    // place, so the owning object immediately sees the new shape.
    let mut arg = resolve_array_arg(rhs[0])?;
    arg.with_mut(|array| array.resize_and_store(new_shape))
}
