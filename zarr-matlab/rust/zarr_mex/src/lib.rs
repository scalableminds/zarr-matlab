extern crate libc;
extern crate zarrs;

mod ffi;
mod macros;
mod read;
mod util;

use ffi::*;
use util::*;

use std::slice;

unsafe fn read(
    nlhs: c_int,
    plhs: *mut MxArrayMut,
    nrhs: c_int,
    prhs: *const MxArray,
) -> Result<()> {
    let rhs = match nrhs == 2 {
        true => slice::from_raw_parts(prhs, nrhs as usize),
        false => return Err("Invalid number of input arguments".to_string()),
    };

    let lhs = match nlhs == 1 {
        true => slice::from_raw_parts_mut(plhs, nlhs as usize),
        false => return Err("Invalid number of output arguments".to_string()),
    };

    let mat_arr = crate::read::read(rhs)?;

    // set output
    lhs[0] = mat_arr;

    Ok(())
}

mex_function!(nlhs, lhs, nrhs, rhs, { read(nlhs, lhs, nrhs, rhs) });
