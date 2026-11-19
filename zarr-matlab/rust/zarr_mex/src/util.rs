use ffi::*;

use std;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use zarrs::array::data_type::{
    BoolDataType, Float32DataType, Float64DataType, Int16DataType, Int32DataType, Int64DataType,
    Int8DataType, UInt16DataType, UInt32DataType, UInt64DataType, UInt8DataType,
};
use zarrs::array::{
    Array, ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayMetadata, ArrayMetadataOptions, ArraySubset,
    ChunkShapeTraits, DataType, ElementOrder, Resources,
};
use zarrs::storage::{
    ReadableStorage, ReadableStorageTraits, ReadableWritableListableStorage,
    ReadableWritableListableStorageTraits,
};

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

// Dense F-order (column-major) strides: first axis varies fastest.
pub fn f_strides_of(shape: &[u64]) -> Vec<u64> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1u64;
    for &size in shape {
        strides.push(stride);
        stride *= size;
    }
    strides
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
        Arc::new(crate::http::HttpStore(store))
    } else {
        let store_path: PathBuf = path.into();
        let store = zarrs_result_to_str_error(
            zarrs::filesystem::FilesystemStore::new(&store_path),
            &format!("Error while opening local store at '{}'", path),
        )?;
        Arc::new(store)
    };

    // Note: we intentionally do not pre-check that `zarr.json` exists here.
    // Every caller immediately does `Array::open`, which reads and validates
    // `zarr.json` itself, so a pre-check would just double the metadata I/O
    // (a full extra request over HTTP) on every open.
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

// ---------------------------------------------------------------------------
// Per-object open-array cache.
//
// `ZarrArray` (a MATLAB handle class) opens its backing array once and reuses
// the resulting handle for all ops, freeing it in its `delete` destructor.
// The mex ops accept either a path string (transient open, unchanged behavior)
// or a numeric handle (cached), so direct-mex callers are unaffected.
//
// The cached array holds only metadata plus the store connection; chunk data
// is still read fresh on every op, so reads stay correct. A `resize` through a
// handle mutates the cached array in place, so the owning object always sees
// its own changes.
// ---------------------------------------------------------------------------

// An opened array. Local paths open a writable array (usable for read + write);
// HTTP opens a readable-only array.
pub enum OpenedArray {
    Writable(Array<dyn ReadableWritableListableStorageTraits>),
    Readable(Array<dyn ReadableStorageTraits>),
}

impl OpenedArray {
    pub fn shape(&self) -> &[u64] {
        match self {
            OpenedArray::Writable(a) => a.shape(),
            OpenedArray::Readable(a) => a.shape(),
        }
    }

    pub fn data_type(&self) -> &DataType {
        match self {
            OpenedArray::Writable(a) => a.data_type(),
            OpenedArray::Readable(a) => a.data_type(),
        }
    }

    pub fn metadata(&self) -> &ArrayMetadata {
        match self {
            OpenedArray::Writable(a) => a.metadata(),
            OpenedArray::Readable(a) => a.metadata(),
        }
    }

    pub fn chunk_shape_vec(&self, origin: &[u64]) -> Result<Vec<u64>> {
        let cs = match self {
            OpenedArray::Writable(a) => a.chunk_shape(origin),
            OpenedArray::Readable(a) => a.chunk_shape(origin),
        };
        Ok(zarrs_result_to_str_error(cs, "Error while determining chunk/shard shape")?.to_array_shape())
    }

    // Decode `subset` into `target`, whose layout (e.g. a column-major view of a
    // MATLAB array) determines where each element lands.
    pub fn retrieve_array_subset_into(
        &self,
        subset: &ArraySubset,
        target: &mut ArrayBytesDecodeIntoTarget,
    ) -> Result<()> {
        let resources = Resources::default();
        let r = match self {
            OpenedArray::Writable(a) => a.retrieve_array_subset_into(subset, target, &resources),
            OpenedArray::Readable(a) => a.retrieve_array_subset_into(subset, target, &resources),
        };
        zarrs_result_to_str_error(r, "Error while reading data from array")
    }

    // Store `bytes`, given in column-major (MATLAB) order, to `subset`.
    pub fn store_array_subset(&self, subset: &ArraySubset, bytes: ArrayBytes) -> Result<()> {
        match self {
            OpenedArray::Writable(a) => zarrs_result_to_str_error(
                a.store_array_subset(subset, bytes, &Resources::default()),
                "Error while writing data",
            ),
            OpenedArray::Readable(_) => {
                Err("HTTP URLs are not supported for write operations".to_string())
            }
        }
    }

    pub fn resize_and_store(&mut self, new_shape: Vec<u64>) -> Result<()> {
        match self {
            OpenedArray::Writable(a) => {
                zarrs_result_to_str_error(a.set_shape(new_shape), "Error while resizing array")?;
                zarrs_result_to_str_error(a.store_metadata(), "Error while writing array metadata")
            }
            OpenedArray::Readable(_) => {
                Err("HTTP URLs are not supported for write operations".to_string())
            }
        }
    }
}

// Configure an opened array for MATLAB: elements are exchanged in column-major
// order, and stored metadata omits the zarrs-specific `_zarrs` attributes.
pub fn configure_for_matlab<TStorage: ?Sized>(array: &mut Array<TStorage>) {
    array
        .set_element_order(ElementOrder::F)
        .set_metadata_options(ArrayMetadataOptions::default().with_include_zarrs_metadata(false));
}

// Open an array without caching it (writable for local paths, readable for HTTP).
fn open_opened_array(path: &str) -> Result<OpenedArray> {
    if is_http_url(path) {
        let store = create_readable_store(path)?;
        let mut array =
            zarrs_result_to_str_error(Array::open(store, "/"), "Error while opening array")?;
        configure_for_matlab(&mut array);
        Ok(OpenedArray::Readable(array))
    } else {
        let store = create_writable_store(path)?;
        let mut array =
            zarrs_result_to_str_error(Array::open(store, "/"), "Error while opening array")?;
        configure_for_matlab(&mut array);
        Ok(OpenedArray::Writable(array))
    }
}

// Registry of cached open arrays keyed by an opaque handle. The outer mutex
// guards the map only; each entry has its own mutex so the map lock is never
// held during I/O. MATLAB serializes mex calls, so contention is nil.
fn array_registry() -> &'static Mutex<HashMap<u64, Arc<Mutex<OpenedArray>>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<Mutex<OpenedArray>>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

pub fn registry_open(path: &str) -> Result<u64> {
    let opened = open_opened_array(path)?;
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    array_registry()
        .lock()
        .unwrap()
        .insert(handle, Arc::new(Mutex::new(opened)));
    Ok(handle)
}

pub fn registry_close(handle: u64) {
    array_registry().lock().unwrap().remove(&handle);
}

fn registry_get(handle: u64) -> Result<Arc<Mutex<OpenedArray>>> {
    array_registry()
        .lock()
        .unwrap()
        .get(&handle)
        .cloned()
        .ok_or_else(|| "Invalid or closed array handle".to_string())
}

// A resolved first argument: either a transient (uncached) open array or a
// reference to a cached one.
pub enum ArrayArg {
    Transient(OpenedArray),
    Cached(Arc<Mutex<OpenedArray>>),
}

impl ArrayArg {
    pub fn with<R>(&self, f: impl FnOnce(&OpenedArray) -> R) -> R {
        match self {
            ArrayArg::Transient(a) => f(a),
            ArrayArg::Cached(m) => {
                let guard = m.lock().unwrap();
                f(&guard)
            }
        }
    }

    pub fn with_mut<R>(&mut self, f: impl FnOnce(&mut OpenedArray) -> R) -> R {
        match self {
            ArrayArg::Transient(a) => f(a),
            ArrayArg::Cached(m) => {
                let mut guard = m.lock().unwrap();
                f(&mut guard)
            }
        }
    }
}

// True if `pm` is a MATLAB char array or string scalar (i.e. a path), as
// opposed to a numeric array handle.
fn is_string_like(pm: MxArray) -> bool {
    let string_class = CString::new("string").unwrap();
    unsafe { mxIsChar(pm) || mxIsClass(pm, string_class.as_ptr()) }
}

// Resolve the first mex argument to an array: a string opens transiently, a
// numeric value looks up a cached handle.
pub fn resolve_array_arg(pm: MxArray) -> Result<ArrayArg> {
    if is_string_like(pm) {
        let path = mx_array_to_str(pm)?;
        Ok(ArrayArg::Transient(open_opened_array(path)?))
    } else {
        let buf = mx_array_to_f64_slice(pm)?;
        let handle = *buf.first().ok_or("Empty array handle")? as u64;
        Ok(ArrayArg::Cached(registry_get(handle)?))
    }
}

// `open` op: open (and cache) an array by path, returning a numeric handle.
pub fn open_handle(rhs: &[MxArray]) -> Result<MxArrayMut> {
    if rhs.len() != 1 {
        return Err(format!(
            "Invalid number of input arguments. Expected 1 (path), got {}",
            rhs.len()
        ));
    }
    let path = mx_array_to_str(rhs[0])?;
    let handle = registry_open(path)?;

    let arr = create_numeric_array(&[1, 1], MxClassId::Double, MxComplexity::Real)?;
    let ptr = unsafe { mxGetPr(arr) };
    if ptr.is_null() {
        return Err("Failed to allocate array handle".to_string());
    }
    unsafe {
        *ptr = handle as f64;
    }
    Ok(arr)
}

// `close` op: drop a cached array by handle.
pub fn close_handle(rhs: &[MxArray]) -> Result<()> {
    if rhs.len() != 1 {
        return Err(format!(
            "Invalid number of input arguments. Expected 1 (handle), got {}",
            rhs.len()
        ));
    }
    let buf = mx_array_to_f64_slice(rhs[0])?;
    let handle = *buf.first().ok_or("Empty array handle")? as u64;
    registry_close(handle);
    Ok(())
}
