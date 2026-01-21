use zarrs::array::Array;

use std::ffi::CString;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ffi::*;
use crate::util::*;

fn create_double_vector(values: &[u64]) -> Result<MxArrayMut> {
    let dims: [u64; 2] = [1, values.len() as u64];
    let arr = create_numeric_array(&dims, MxClassId::Double, MxComplexity::Real)?;
    let ptr = unsafe { mxGetPr(arr) };
    if ptr.is_null() {
        return Err("Failed to get pointer to array".to_string());
    }
    unsafe {
        for (i, &val) in values.iter().enumerate() {
            *ptr.add(i) = val as f64;
        }
    }
    Ok(arr)
}

pub(crate) fn info(rhs: &[MxArray]) -> Result<MxArrayMut> {
    if rhs.len() != 1 {
        return Err(format!(
            "Invalid number of input arguments. Expected 1 (path), got {}",
            rhs.len()
        ));
    }

    let store_str = rhs[0];
    let store_path: PathBuf = mx_array_to_str(store_str)?.into();

    // Open the array
    let store: zarrs::storage::ReadableWritableListableStorage = Arc::new(
        zarrs_result_to_str_error(zarrs::filesystem::FilesystemStore::new(&store_path))?,
    );
    let array = zarrs_result_to_str_error(Array::open(store.clone(), "/"))?;

    let shape = array.shape();
    let ndim = shape.len();

    // Create the boundingBox matrix (ndim x 2)
    let bbox_dims: [u64; 2] = [ndim as u64, 2];
    let bbox_arr = create_numeric_array(&bbox_dims, MxClassId::Double, MxComplexity::Real)?;
    let bbox_ptr = unsafe { mxGetPr(bbox_arr) };
    if bbox_ptr.is_null() {
        return Err("Failed to get pointer to bounding box array".to_string());
    }
    unsafe {
        for i in 0..ndim {
            *bbox_ptr.add(i) = 1.0;
            *bbox_ptr.add(ndim + i) = (shape[i] + 1) as f64;
        }
    }

    // Get data type as string
    let data_type_str = format!("{}", array.data_type());
    let data_type_cstr = CString::new(data_type_str).unwrap();
    let data_type_arr = unsafe { mxCreateString(data_type_cstr.as_ptr()) };

    // Get chunk shape from chunk representation at origin
    let chunk_origin: Vec<u64> = vec![0; ndim];
    let shard_shape = zarrs_result_to_str_error(array.chunk_shape(&chunk_origin))?.to_array_shape();

    // Get shard shape (from sharding codec if present, otherwise same as chunk)
    let inner_chunk_shape: Option<Vec<u64>> = match array.metadata() {
        zarrs::array::ArrayMetadata::V3(metadata) => metadata.codecs.iter().find_map(|codec| {
            if codec.name() == "sharding_indexed" {
                codec
                    .configuration()
                    .and_then(|configuration| configuration.get("chunk_shape"))
                    .and_then(|inner_chunk_shape| inner_chunk_shape.as_array())
                    .map(|inner_chunk_shape| {
                        inner_chunk_shape
                            .iter()
                            .map(|d| d.as_u64().unwrap())
                            .collect()
                    })
            } else {
                None
            }
        }),
        _ => None,
    };

    // Create field names
    let field_bbox = CString::new("boundingBox").unwrap();
    let field_dtype = CString::new("dataType").unwrap();
    let field_chunk = CString::new("chunkShape").unwrap();
    let field_shard = CString::new("shardShape").unwrap();

    let field_names: [*const c_char; 4] = [
        field_bbox.as_ptr(),
        field_dtype.as_ptr(),
        field_chunk.as_ptr(),
        field_shard.as_ptr(),
    ];

    let result_struct = unsafe { mxCreateStructMatrix(1, 1, 4, field_names.as_ptr()) };
    if result_struct.is_null() {
        return Err("Failed to create output struct".to_string());
    }

    // Set fields
    unsafe {
        mxSetField(result_struct, 0, field_bbox.as_ptr(), bbox_arr);
        mxSetField(result_struct, 0, field_dtype.as_ptr(), data_type_arr);

        match inner_chunk_shape {
            Some(inner_chunk_shape) => {
                mxSetField(
                    result_struct,
                    0,
                    field_chunk.as_ptr(),
                    create_double_vector(&inner_chunk_shape)?,
                );
                mxSetField(
                    result_struct,
                    0,
                    field_shard.as_ptr(),
                    create_double_vector(&shard_shape)?,
                );
            }
            None => {
                mxSetField(
                    result_struct,
                    0,
                    field_chunk.as_ptr(),
                    create_double_vector(&shard_shape)?,
                );
                mxSetField(
                    result_struct,
                    0,
                    field_shard.as_ptr(),
                    create_double_vector(&shard_shape)?,
                );
            }
        }
    }

    Ok(result_struct)
}
