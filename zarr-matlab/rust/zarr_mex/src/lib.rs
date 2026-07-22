extern crate libc;
extern crate rayon;
extern crate zarrs;

mod create;
mod ffi;
mod info;
mod macros;
mod read;
mod resize;
mod util;
mod write;

use ffi::*;
use util::*;

use std::slice;

fn dispatch(nlhs: c_int, plhs: *mut MxArrayMut, nrhs: c_int, prhs: *const MxArray) -> Result<()> {
    let rhs = if nrhs > 0 {
        unsafe { slice::from_raw_parts(prhs, nrhs as usize) }
    } else {
        return Err("Invalid number of input arguments".to_string());
    };

    let lhs = if nlhs >= 0 {
        unsafe { slice::from_raw_parts_mut(plhs, nlhs as usize) }
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
        "write" => {
            if lhs.len() != 0 {
                return Err(format!(
                    "Invalid number of output arguments. Expected 0, got {}.",
                    lhs.len()
                ));
            }
            crate::write::write(rhs)?;
        }
        "create" => {
            if lhs.len() != 0 {
                return Err(format!(
                    "Invalid number of output arguments. Expected 0, got {}.",
                    lhs.len()
                ));
            }
            crate::create::create(rhs)?;
        }
        "resize" => {
            if lhs.len() != 0 {
                return Err(format!(
                    "Invalid number of output arguments. Expected 0, got {}.",
                    lhs.len()
                ));
            }
            crate::resize::resize(rhs)?;
        }
        "info" => {
            if lhs.len() != 1 {
                return Err(format!(
                    "Invalid number of output arguments. Expected 1, got {}.",
                    lhs.len()
                ));
            }
            lhs[0] = crate::info::info(rhs)?;
        }
        "open" => {
            if lhs.len() != 1 {
                return Err(format!(
                    "Invalid number of output arguments. Expected 1, got {}.",
                    lhs.len()
                ));
            }
            lhs[0] = open_handle(rhs)?;
        }
        "close" => {
            if lhs.len() != 0 {
                return Err(format!(
                    "Invalid number of output arguments. Expected 0, got {}.",
                    lhs.len()
                ));
            }
            close_handle(rhs)?;
        }
        _ => return Err(format!("Unknown command {:?}", command)),
    }

    Ok(())
}

mex_function!(nlhs, lhs, nrhs, rhs, { dispatch(nlhs, lhs, nrhs, rhs) });
