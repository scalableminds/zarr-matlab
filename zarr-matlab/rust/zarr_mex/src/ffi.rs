pub use libc::{c_double, c_int, c_uint, c_void, size_t};
pub use std::os::raw::c_char;

// HACK(amotta): There is no guarantee for this
#[allow(non_camel_case_types)]
type c_bool = bool;

pub enum MxArrayT {}
pub type MxArray = *const MxArrayT;
pub type MxArrayMut = *mut MxArrayT;

// HACK(amotta): This is only true for the new, 64bit API
pub type MwSize = size_t;

pub enum MxComplexity {
    Real,
    Complex,
}

#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub enum MxClassId {
    // from tyepdef enum { .. } mxClassID in
    // $MATLABROOT/extern/include/matrix.h:262 of MATLAB R2016b
    Unknown,
    Cell,
    Struct,
    Logical,
    Char,
    Void,
    Double,
    Single,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Function,
    Opaque,
    Object,
    Index,
    Sparse,
}

#[cfg_attr(target_os = "linux", link(name = "mx", kind = "dylib"))]
#[cfg_attr(target_os = "macos", link(name = "mx", kind = "dylib"))]
#[cfg_attr(target_os = "windows", link(name = "libmx", kind = "dylib"))]
extern "C-unwind" {
    // creation
    #[link_name = "mxCreateNumericArray_730"]
    pub fn mxCreateNumericArray(
        ndim: size_t,
        dims: *const size_t,
        class_id: c_int,
        complex_flag: c_int,
    ) -> MxArrayMut;

    // access
    pub fn mxGetPr(pm: MxArray) -> *mut c_double;
    pub fn mxGetData(pm: MxArray) -> *mut c_void;
    pub fn mxArrayToUTF8String(pm: MxArray) -> *const c_char;
    #[link_name = "mxGetNumberOfDimensions_730"]
    pub fn mxGetNumberOfDimensions(pm: MxArray) -> size_t;
    #[link_name = "mxGetDimensions_730"]
    pub fn mxGetDimensions(pm: MxArray) -> *const size_t;
    pub fn mxGetNumberOfElements(pm: MxArray) -> size_t;
    pub fn mxGetElementSize(pm: MxArray) -> size_t;
    pub fn mxGetClassID(pm: MxArray) -> MxClassId;

    // validation
    pub fn mxIsChar(pm: MxArray) -> c_bool;
    pub fn mxIsScalar(pm: MxArray) -> c_bool;
    pub fn mxIsDouble(pm: MxArray) -> c_bool;
    pub fn mxIsComplex(pm: MxArray) -> c_bool;

    // struct creation and access
    #[link_name = "mxCreateStructMatrix_730"]
    pub fn mxCreateStructMatrix(
        m: size_t,
        n: size_t,
        nfields: c_int,
        fieldnames: *const *const c_char,
    ) -> MxArrayMut;
    #[link_name = "mxSetField_730"]
    pub fn mxSetField(pm: MxArrayMut, index: size_t, fieldname: *const c_char, pvalue: MxArrayMut);

    // string creation
    pub fn mxCreateString(str: *const c_char) -> MxArrayMut;
}

#[cfg_attr(target_os = "linux", link(name = "mex", kind = "dylib"))]
#[cfg_attr(target_os = "macos", link(name = "mex", kind = "dylib"))]
#[cfg_attr(target_os = "windows", link(name = "libmex", kind = "dylib"))]
extern "C-unwind" {
    pub fn mexErrMsgIdAndTxt(errorid: *const c_char, errormsg: *const c_char) -> ();
}
