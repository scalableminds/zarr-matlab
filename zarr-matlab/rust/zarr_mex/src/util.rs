use ffi::*;

use rayon::prelude::*;

use std;
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::slice;
use std::sync::Arc;

use zarrs::array::data_type::{
    BoolDataType, Float32DataType, Float64DataType, Int16DataType, Int32DataType, Int64DataType,
    Int8DataType, UInt16DataType, UInt32DataType, UInt64DataType, UInt8DataType,
};
use zarrs::array::{ArraySubset, DataType};
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

// ---------------------------------------------------------------------------
// Strided transpose between C-order (row-major, as produced/consumed by zarrs)
// and F-order (column-major, as used by MATLAB).
//
// The two public entry points below (`copy_as_fortran_order`,
// `copy_as_c_order`) both reduce to a single generic strided copy: read every
// element once following the *destination* layout's fastest axis, and scatter
// it to the position given by the source/destination strides. Compared to the
// previous element-by-element version this adds:
//   * an incremental "odometer" offset (O(1) amortized per element instead of
//     an O(ndim) dot-product),
//   * monomorphization over the element byte-size so the copy is a single
//     aligned load/store of `[u8; N]`,
//   * a cache-blocked 2D fast path, and
//   * rayon parallelism across disjoint contiguous destination chunks.
// ---------------------------------------------------------------------------

// Only parallelize once there is enough work to amortize the thread hand-off,
// and only when each per-thread chunk is itself substantial.
const PARALLEL_THRESHOLD: usize = 1 << 16;
const MIN_PARALLEL_CHUNK: usize = 1 << 12;
// Tile size (in elements) for the cache-blocked 2D transpose.
const TILE: usize = 64;

// Reinterpret a byte slice as a slice of fixed-size element blocks. Sound
// because `[u8; N]` has alignment 1, so any `&[u8]` is validly aligned for it
// and the length is an exact multiple of `N` at every call site.
#[inline]
fn as_chunks<const N: usize>(buf: &[u8]) -> &[[u8; N]] {
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const [u8; N], buf.len() / N) }
}
#[inline]
fn as_chunks_mut<const N: usize>(buf: &mut [u8]) -> &mut [[u8; N]] {
    unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut [u8; N], buf.len() / N) }
}

// Dense F-order (column-major) strides: first axis varies fastest.
fn f_strides_of(shape: &[u64]) -> Vec<u64> {
    let ndim = shape.len();
    let mut s = vec![1u64; ndim];
    for i in 1..ndim {
        s[i] = s[i - 1] * shape[i - 1];
    }
    s
}
// Dense C-order (row-major) strides: last axis varies fastest.
fn c_strides_of(shape: &[u64]) -> Vec<u64> {
    let ndim = shape.len();
    let mut s = vec![1u64; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

// Drop element `p` from a stride/shape vector (used to peel off the axis that
// is parallelized over).
fn remove_at(v: &[u64], p: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(v.len().saturating_sub(1));
    for (i, &x) in v.iter().enumerate() {
        if i != p {
            out.push(x);
        }
    }
    out
}

// Generic N-dimensional strided copy. Walks the destination in row-major order
// (last axis fastest) while maintaining the corresponding source offset via an
// incremental odometer. `src_base` is an extra source offset applied to every
// element, used when a leading axis has been peeled off for parallelism.
#[inline]
fn transpose_base<const N: usize>(
    src: &[[u8; N]],
    dst: &mut [[u8; N]],
    shape: &[u64],
    src_strides: &[u64],
    dst_strides: &[u64],
    src_base: usize,
) {
    let ndim = shape.len();
    if dst.is_empty() {
        return;
    }
    let mut idx = vec![0u64; ndim];
    let mut s_off = src_base;
    let mut d_off = 0usize;
    unsafe {
        loop {
            *dst.get_unchecked_mut(d_off) = *src.get_unchecked(s_off);
            let mut d = ndim;
            loop {
                if d == 0 {
                    return;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < shape[d] {
                    s_off += src_strides[d] as usize;
                    d_off += dst_strides[d] as usize;
                    break;
                }
                idx[d] = 0;
                s_off -= (src_strides[d] * (shape[d] - 1)) as usize;
                d_off -= (dst_strides[d] * (shape[d] - 1)) as usize;
            }
        }
    }
}

// Cache-blocked 2D transpose. Tiling keeps both the strided source reads and
// strided destination writes within cache for each TILE x TILE block.
#[inline]
fn transpose_2d<const N: usize>(
    src: &[[u8; N]],
    dst: &mut [[u8; N]],
    shape: &[u64],
    src_strides: &[u64],
    dst_strides: &[u64],
) {
    let n0 = shape[0] as usize;
    let n1 = shape[1] as usize;
    let (ss0, ss1) = (src_strides[0] as usize, src_strides[1] as usize);
    let (ds0, ds1) = (dst_strides[0] as usize, dst_strides[1] as usize);
    let mut i0b = 0;
    while i0b < n0 {
        let i0e = (i0b + TILE).min(n0);
        let mut i1b = 0;
        while i1b < n1 {
            let i1e = (i1b + TILE).min(n1);
            for i0 in i0b..i0e {
                let mut s = i0 * ss0 + i1b * ss1;
                let mut d = i0 * ds0 + i1b * ds1;
                for _ in i1b..i1e {
                    unsafe {
                        *dst.get_unchecked_mut(d) = *src.get_unchecked(s);
                    }
                    s += ss1;
                    d += ds1;
                }
            }
            i1b = i1e;
        }
        i0b = i0e;
    }
}

// Dispatch for one element size. Picks the outermost destination axis (largest
// destination stride) as the parallel/decomposition axis so each destination
// chunk is contiguous and disjoint.
fn transpose<const N: usize>(
    src: &[[u8; N]],
    dst: &mut [[u8; N]],
    shape: &[u64],
    src_strides: &[u64],
    dst_strides: &[u64],
) {
    let ndim = shape.len();
    let total = dst.len();
    if total == 0 {
        return;
    }
    if ndim <= 1 {
        dst.copy_from_slice(src);
        return;
    }

    let p = (0..ndim).max_by_key(|&i| dst_strides[i]).unwrap();
    let outer = shape[p] as usize;
    let chunk = dst_strides[p] as usize; // == total / outer for the outermost axis

    if total >= PARALLEL_THRESHOLD && outer >= 2 && chunk >= MIN_PARALLEL_CHUNK {
        // Peel off axis `p`: each contiguous destination chunk of length
        // `chunk` corresponds to one index along `p`, and reads from the
        // source starting at `j * base_step`.
        let base_step = src_strides[p] as usize;
        let sub_shape = remove_at(shape, p);
        let sub_src = remove_at(src_strides, p);
        let sub_dst = remove_at(dst_strides, p);
        dst.par_chunks_mut(chunk).enumerate().for_each(|(j, dc)| {
            transpose_base::<N>(src, dc, &sub_shape, &sub_src, &sub_dst, j * base_step);
        });
    } else if ndim == 2 {
        transpose_2d::<N>(src, dst, shape, src_strides, dst_strides);
    } else {
        transpose_base::<N>(src, dst, shape, src_strides, dst_strides, 0);
    }
}

// Byte-wise fallback for element sizes other than 1/2/4/8 (same odometer, but
// copies `ts` bytes per element instead of a fixed-size block).
fn transpose_bytes(
    src: &[u8],
    dst: &mut [u8],
    shape: &[u64],
    src_strides: &[u64],
    dst_strides: &[u64],
    ts: usize,
) {
    let ndim = shape.len();
    let total = dst.len() / ts;
    if total == 0 {
        return;
    }
    let mut idx = vec![0u64; ndim];
    let mut s_off = 0usize;
    let mut d_off = 0usize;
    loop {
        dst[d_off * ts..d_off * ts + ts].copy_from_slice(&src[s_off * ts..s_off * ts + ts]);
        let mut d = ndim;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < shape[d] {
                s_off += src_strides[d] as usize;
                d_off += dst_strides[d] as usize;
                break;
            }
            idx[d] = 0;
            s_off -= (src_strides[d] * (shape[d] - 1)) as usize;
            d_off -= (dst_strides[d] * (shape[d] - 1)) as usize;
        }
    }
}

// Select the monomorphized copy for the element size.
fn transpose_dispatch(
    src: &[u8],
    dst: &mut [u8],
    shape: &[u64],
    src_strides: &[u64],
    dst_strides: &[u64],
    type_size: usize,
) {
    match type_size {
        1 => transpose::<1>(as_chunks(src), as_chunks_mut(dst), shape, src_strides, dst_strides),
        2 => transpose::<2>(as_chunks(src), as_chunks_mut(dst), shape, src_strides, dst_strides),
        4 => transpose::<4>(as_chunks(src), as_chunks_mut(dst), shape, src_strides, dst_strides),
        8 => transpose::<8>(as_chunks(src), as_chunks_mut(dst), shape, src_strides, dst_strides),
        _ => transpose_bytes(src, dst, shape, src_strides, dst_strides, type_size),
    }
}

// Copy a C-order input buffer into an F-order MATLAB output array.
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

    let c_strides = c_strides_of(shape);
    let f_strides = f_strides_of(shape);
    // Source is C-order, destination (MATLAB) is F-order.
    transpose_dispatch(in_buf, result, shape, &c_strides, &f_strides, type_size);

    Ok(())
}

// Copy an F-order MATLAB input array into a freshly allocated C-order buffer.
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

    let f_strides = f_strides_of(shape);
    let c_strides = c_strides_of(shape);
    // Source (MATLAB) is F-order, destination is C-order.
    transpose_dispatch(in_buf, &mut result, shape, &f_strides, &c_strides, type_size);

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
    if data_type.is::<UInt8DataType>() {
        Ok(MxClassId::Uint8)
    } else if data_type.is::<UInt16DataType>() {
        Ok(MxClassId::Uint16)
    } else if data_type.is::<UInt32DataType>() {
        Ok(MxClassId::Uint32)
    } else if data_type.is::<UInt64DataType>() {
        Ok(MxClassId::Uint64)
    } else if data_type.is::<Float32DataType>() {
        Ok(MxClassId::Single)
    } else if data_type.is::<Float64DataType>() {
        Ok(MxClassId::Double)
    } else if data_type.is::<Int8DataType>() {
        Ok(MxClassId::Int8)
    } else if data_type.is::<Int16DataType>() {
        Ok(MxClassId::Int16)
    } else if data_type.is::<Int32DataType>() {
        Ok(MxClassId::Int32)
    } else if data_type.is::<Int64DataType>() {
        Ok(MxClassId::Int64)
    } else if data_type.is::<BoolDataType>() {
        Ok(MxClassId::Logical)
    } else {
        Err("Unsupported data type".to_string())
    }
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
