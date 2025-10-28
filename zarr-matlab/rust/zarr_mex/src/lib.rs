extern crate libc;
extern crate zarrs;

mod ffi;
mod macros;
mod read;
mod util;

use ffi::*;
use util::*;

use std::slice;

unsafe fn dispatch(
    nlhs: c_int,
    plhs: *mut MxArrayMut,
    nrhs: c_int,
    prhs: *const MxArray,
) -> Result<()> {
    let rhs = if nrhs > 0 {
        slice::from_raw_parts(prhs, nrhs as usize)
    } else {
        return Err("Invalid number of input arguments".to_string());
    };

    let lhs = if nlhs >= 0 {
        slice::from_raw_parts_mut(plhs, nlhs as usize)
    } else {
        return Err("Invalid number of output arguments".to_string());
    };

    let command = mx_array_to_str(rhs[0])?;
    let rhs = &rhs[1..];

    match command {
        "read" => {
            if lhs.len() != 1 {
                return Err(format!(
                    "Invalid number of output arguments. Expected 1, got {}.",
                    lhs.len()
                ));
            }
            let mat_arr = crate::read::read(rhs)?;
            lhs[0] = mat_arr;
        }
        _ => return Err(format!("Unknown command {:?}", command)),
    }

    Ok(())
}

mex_function!(nlhs, lhs, nrhs, rhs, { dispatch(nlhs, lhs, nrhs, rhs) });
