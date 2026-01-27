use ffi::*;

use std;
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::slice;
use std::sync::Arc;

use zarrs::array::data_type::DataType;
use zarrs::array_subset::ArraySubset;
use zarrs::storage::{ReadableStorage, ReadableWritableListableStorage, StoreKey};

pub type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone)]
pub struct BBox {
    pub start: Vec<u64>,
    pub shape: Vec<u64>,
}

impl BBox {
    pub fn new(start: Vec<u64>, shape: Vec<u64>) -> BBox {
        BBox { start, shape }
    }

    pub fn check_bounds(&self, array_shape: &[u64]) -> Result<()> {
        if self
            .start
            .iter()
            .zip(self.shape.iter())
            .zip(array_shape.iter())
            .any(|((bbox_min_x, bbox_shape_x), shape_x)| (*bbox_min_x + *bbox_shape_x) > *shape_x)
        {
            return Err(format!(
                "Bounding box {:?} is out of bounds for array of shape={:?}.",
                self, array_shape
            ));
        }
        Ok(())
    }

    pub fn to_subset(&self) -> Result<ArraySubset> {
        let subset = zarrs_result_to_str_error(
            ArraySubset::new_with_start_shape(self.start.clone(), self.shape.clone()),
            "Error while creating array subset",
        )?;
        Ok(subset)
    }
}

pub fn as_nat(f: f64) -> Result<u64> {
    if f <= 0.0 {
        return Err("Input must be positive".to_string());
    }

    match f % 1.0 == 0.0 {
        true => Ok(f as u64),
        false => Err("Input must be an integer".to_string()),
    }
}

pub fn mx_array_to_str<'a>(pm: MxArray) -> Result<&'a str> {
    // Check if it's a MATLAB string scalar (double-quoted string)
    let string_class = CString::new("string").unwrap();
    let is_string = unsafe { mxIsClass(pm, string_class.as_ptr()) };

    let pm_to_convert = if is_string {
        // Convert string scalar to char array using MATLAB's char() function
        let mut char_array: MxArrayMut = std::ptr::null_mut();
        let char_fn = CString::new("char").unwrap();
        let result = unsafe {
            mexCallMATLAB(
                1,
                &mut char_array as *mut MxArrayMut,
                1,
                &pm as *const MxArray,
                char_fn.as_ptr(),
            )
        };
        if result != 0 || char_array.is_null() {
            return Err("Failed to convert MATLAB string to char array".to_string());
        }
        char_array as MxArray
    } else {
        pm
    };

    let pm_ptr = unsafe { mxArrayToUTF8String(pm_to_convert) };

    if pm_ptr.is_null() {
        return Err("mxArrayToUTF8String returned null".to_string());
    }

    let pm_cstr = unsafe { CStr::from_ptr(pm_ptr) };

    match pm_cstr.to_str() {
        Ok(pm_str) => Ok(pm_str),
        Err(_) => Err("mxArray contains invalid UTF-8 data".to_string()),
    }
}

pub fn mx_array_to_f64_slice<'a>(pm: MxArray) -> Result<&'a [f64]> {
    unsafe {
        if !mxIsDouble(pm) {
            return Err("MxArray is not of class \"double\"".to_string());
        };
        if mxIsComplex(pm) {
            return Err("MxArray is complex".to_string());
        };
    }

    let pm_numel = unsafe { mxGetNumberOfElements(pm) };
    let pm_ptr = unsafe { mxGetPr(pm) };

    match pm_ptr.is_null() {
        true => Err("MxArray does not contain real values".to_string()),
        false => Ok(unsafe { slice::from_raw_parts(pm_ptr, pm_numel) }),
    }
}

pub fn mx_array_to_u8_slice<'a>(pm: MxArray) -> Result<&'a [u8]> {
    let numel = unsafe { mxGetNumberOfElements(pm) };
    let elem_size = unsafe { mxGetElementSize(pm) };
    let data = unsafe { mxGetData(pm) } as *const u8;

    if elem_size == 0 {
        Err("Failed to determine element size".to_string())
    } else if data.is_null() {
        Err("Data pointer is null".to_string())
    } else {
        Ok(unsafe { slice::from_raw_parts(data, numel * elem_size) })
    }
}

pub fn mx_array_mut_to_u8_slice_mut<'a>(pm: MxArrayMut) -> Result<&'a mut [u8]> {
    let numel = unsafe { mxGetNumberOfElements(pm) };
    let elem_size = unsafe { mxGetElementSize(pm) };
    let data = unsafe { mxGetData(pm) } as *mut u8;

    if elem_size == 0 {
        Err("Failed to determine element size".to_string())
    } else if data.is_null() {
        Err("Data pointer is null".to_string())
    } else {
        Ok(unsafe { slice::from_raw_parts_mut(data, numel * elem_size) })
    }
}

pub fn mx_array_size_to_usize_slice<'a>(pm: MxArray) -> &'a [usize] {
    let ndims = unsafe { mxGetNumberOfDimensions(pm) };
    let dims = unsafe { mxGetDimensions(pm) };

    unsafe { slice::from_raw_parts(dims, ndims as usize) }
}

pub fn create_numeric_array(
    dims: &[u64],
    class: MxClassId,
    complexity: MxComplexity,
) -> Result<MxArrayMut> {
    let arr = unsafe {
        mxCreateNumericArray(
            dims.len() as size_t,
            dims.as_ptr() as *const usize,
            class as c_int,
            complexity as c_int,
        )
    };

    match arr.is_null() {
        true => Err("Failed to create uninitialized numeric array".to_string()),
        false => Ok(arr),
    }
}

pub fn die(msg: &str) {
    let c_id = CString::new("zarr:error").unwrap();
    let c_msg = CString::new(msg)
        .unwrap_or_else(|_| CString::new("Error message contained null byte").unwrap());
    unsafe { mexErrMsgIdAndTxt(c_id.as_ptr(), c_msg.as_ptr()) }
}

pub fn copy_as_fortran_order(
    in_buf: &[u8],
    out_arr: MxArrayMut,
    shape: &[u64],
    type_size: usize,
) -> Result<()> {
    let total_elems: usize = shape.iter().product::<u64>() as usize;
    if in_buf.len() != total_elems * type_size {
        return Err(format!(
            "Length of input buffer does not match expected size {} != {}",
            in_buf.len(),
            total_elems,
        ));
    }

    let result = mx_array_mut_to_u8_slice_mut(out_arr)?;
    if result.len() != total_elems * type_size {
        return Err(format!(
            "Length of output array does not match expected size {} != {}",
            result.len(),
            total_elems,
        ));
    }

    // Compute F-order (column-major) strides
    let mut f_strides = vec![1u64; shape.len()];
    for i in 1..shape.len() {
        f_strides[i] = f_strides[i - 1] * shape[i - 1];
    }

    // Multi-dimensional index for C-order iteration
    let mut idx = vec![0u64; shape.len()];

    // Iterate over all elements in C-order (sequential read)
    for elem_idx in 0..total_elems {
        // Compute Fortran-order (column-major) offset
        let f_offset_elems: u64 = idx.iter().zip(&f_strides).map(|(&i, &s)| i * s).sum();
        let f_offset_bytes = f_offset_elems as usize * type_size;

        // Sequential read from in_buf
        let src_offset_bytes = elem_idx * type_size;
        let src_slice = &in_buf[src_offset_bytes..src_offset_bytes + type_size];

        // Scattered write to result
        result[f_offset_bytes..f_offset_bytes + type_size].copy_from_slice(src_slice);

        // Increment multi-dimensional index (C-order)
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            } else if d > 0 {
                idx[d] = 0;
            }
        }
    }

    Ok(())
}

pub fn copy_as_c_order(in_arr: MxArray, shape: &[u64], type_size: usize) -> Result<Vec<u8>> {
    let total_elems: usize = shape.iter().product::<u64>() as usize;
    let total_bytes = total_elems * type_size;

    let in_buf = mx_array_to_u8_slice(in_arr)?;
    if in_buf.len() != total_bytes {
        return Err(format!(
            "Length of input array does not match expected size {} != {}",
            in_buf.len(),
            total_bytes,
        ));
    }

    let mut result = vec![0u8; total_bytes];

    // Compute C-order (row-major) strides
    let ndim = shape.len();
    let mut c_strides = vec![1u64; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        c_strides[i] = c_strides[i + 1] * shape[i + 1];
    }

    // Multi-dimensional index for F-order iteration
    let mut idx = vec![0u64; ndim];

    // Iterate over all elements in F-order (sequential read from Fortran-ordered input)
    for elem_idx in 0..total_elems {
        // Compute C-order (row-major) offset for scattered write
        let c_offset_elems: u64 = idx.iter().zip(&c_strides).map(|(&i, &s)| i * s).sum();
        let c_offset_bytes = c_offset_elems as usize * type_size;

        // Sequential read from in_buf (F-order)
        let src_offset_bytes = elem_idx * type_size;
        let src_slice = &in_buf[src_offset_bytes..src_offset_bytes + type_size];

        // Scattered write to result (C-order)
        result[c_offset_bytes..c_offset_bytes + type_size].copy_from_slice(src_slice);

        // Increment multi-dimensional index (F-order: first dimension varies fastest)
        for d in 0..ndim {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            } else if d < ndim - 1 {
                idx[d] = 0;
            }
        }
    }

    Ok(result)
}

fn f64_slice_to_vec(buf: &[f64]) -> Result<Vec<u64>> {
    buf.iter()
        .map(|x| as_nat(*x).or(Err("Invalid value".to_string())))
        .collect()
}

pub fn mx_array_to_bbox(pm: MxArray, ndim: usize) -> Result<BBox> {
    let buf = mx_array_to_f64_slice(pm)?;

    // verify shape of array
    let input_arg_shape = mx_array_size_to_usize_slice(pm);
    if input_arg_shape != &[ndim, 2] {
        return Err(format!(
            "Bounding box has invalid shape. Needs to be [{}, 2]. Got {:?}.",
            ndim, input_arg_shape
        ));
    }

    let bbox_min_f64 = &buf[0..ndim];
    let bbox_max_f64 = &buf[ndim..(ndim * 2)];
    let bbox_min = f64_slice_to_vec(bbox_min_f64)
        .or(Err(format!("Invalid lower bound. Got {:?}.", bbox_min_f64)))?;
    let bbox_max = f64_slice_to_vec(bbox_max_f64)
        .or(Err(format!("Invalid upper bound. Got {:?}.", bbox_max_f64)))?;

    if bbox_min
        .iter()
        .zip(bbox_max.iter())
        .any(|(min_x, max_x)| min_x >= max_x)
    {
        return Err(format!(
            "Bounding box has invalid shape. Got min={:?}, max={:?}.",
            bbox_min, bbox_max
        ));
    }

    let bbox_shape: Vec<u64> = bbox_min
        .iter()
        .zip(bbox_max.iter())
        .map(|(min_x, max_x)| max_x - min_x)
        .collect();

    let bbox_min = bbox_min.iter().map(|x| x - 1).collect();

    if bbox_shape.iter().any(|x| *x < 1) {
        return Err(format!(
            "Bounding box has invalid shape. Got {:?}.",
            bbox_shape
        ));
    }

    Ok(BBox::new(bbox_min, bbox_shape))
}

pub fn zarrs_result_to_str_error<T, E: std::error::Error>(
    result: std::result::Result<T, E>,
    error_msg: &str,
) -> Result<T> {
    match result {
        Ok(ok) => Ok(ok),
        Err(err) => Err(format!("{}: {}", error_msg, err)),
    }
}

pub fn zarrs_data_type_to_mx(data_type: &DataType) -> Result<MxClassId> {
    Ok(match data_type {
        DataType::UInt8 => MxClassId::Uint8,
        DataType::UInt16 => MxClassId::Uint16,
        DataType::UInt32 => MxClassId::Uint32,
        DataType::UInt64 => MxClassId::Uint64,
        DataType::Float32 => MxClassId::Single,
        DataType::Float64 => MxClassId::Double,
        DataType::Int8 => MxClassId::Int8,
        DataType::Int16 => MxClassId::Int16,
        DataType::Int32 => MxClassId::Int32,
        DataType::Int64 => MxClassId::Int64,
        DataType::Bool => MxClassId::Logical,
        _ => {
            return Err("Unsupported data type".to_string());
        }
    })
}

pub fn is_http_url(path: &str) -> bool {
    path.starts_with("http://") || path.starts_with("https://")
}

pub fn create_readable_store(path: &str) -> Result<ReadableStorage> {
    let store: ReadableStorage = if is_http_url(path) {
        let store = zarrs_result_to_str_error(
            zarrs_http::HTTPStore::new(path),
            &format!("Error while opening HTTP store at '{}'", path),
        )?;
        Arc::new(store)
    } else {
        let store_path: PathBuf = path.into();
        let store = zarrs_result_to_str_error(
            zarrs::filesystem::FilesystemStore::new(&store_path),
            &format!("Error while opening local store at '{}'", path),
        )?;
        Arc::new(store)
    };

    // Check if zarr.json exists
    let store_key =
        zarrs_result_to_str_error(StoreKey::new("zarr.json"), "Error while creating store key")?;
    match store.get(&store_key) {
        Ok(Some(_)) => {}
        _ => {
            return Err(format!(
                "No zarr.json found at '{}'. Not a valid Zarr v3 array.",
                path
            ));
        }
    };

    Ok(store)
}

pub fn create_writable_store(path: &str) -> Result<ReadableWritableListableStorage> {
    if is_http_url(path) {
        return Err("HTTP URLs are not supported for write operations".to_string());
    }
    let store_path: PathBuf = path.into();
    let store = zarrs_result_to_str_error(
        zarrs::filesystem::FilesystemStore::new(&store_path),
        &format!("Error while opening (writable) local store at '{}'", path),
    )?;
    Ok(Arc::new(store))
}
