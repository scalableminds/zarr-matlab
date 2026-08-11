use zarrs::plugin::{ExtensionName, ZarrVersion};

use std::ffi::CString;

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

    let arg = resolve_array_arg(rhs[0])?;

    // Gather the plain metadata under the (possibly cached) array's lock, then
    // build the MATLAB struct once the lock is released.
    let (shape, data_type_str, shard_shape, inner_chunk_shape, zarr_format) = arg.with(|array| {
        let shape = array.shape().to_vec();
        let ndim = shape.len();

        // Data type as string (use the Zarr V3 name, e.g. "uint32")
        let data_type_str = array
            .data_type()
            .name(ZarrVersion::V3)
            .map(|name| name.into_owned())
            .unwrap_or_else(|| format!("{}", array.data_type()));

        // Chunk/shard shape from the chunk representation at the origin
        let chunk_origin: Vec<u64> = vec![0; ndim];
        let shard_shape = array.chunk_shape_vec(&chunk_origin)?;

        // Inner chunk shape (from sharding codec if present, otherwise none).
        // Zarr V2 has no sharding, so it never has an inner chunk shape.
        let (inner_chunk_shape, zarr_format): (Option<Vec<u64>>, u64) = match array.metadata() {
            zarrs::array::ArrayMetadata::V3(metadata) => {
                let inner_chunk_shape = metadata.codecs.iter().find_map(|codec| {
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
                });
                (inner_chunk_shape, 3)
            }
            zarrs::array::ArrayMetadata::V2(_) => (None, 2),
        };

        Ok::<_, String>((
            shape,
            data_type_str,
            shard_shape,
            inner_chunk_shape,
            zarr_format,
        ))
    })?;

    // Create the shape vector
    let shape_arr = create_double_vector(&shape)?;

    let data_type_cstr = CString::new(data_type_str).unwrap();
    let data_type_arr = unsafe { mxCreateString(data_type_cstr.as_ptr()) };

    let zarr_format_arr = create_double_vector(&[zarr_format])?;

    // Create field names
    let field_shape = CString::new("shape").unwrap();
    let field_dtype = CString::new("dataType").unwrap();
    let field_chunk = CString::new("chunkShape").unwrap();
    let field_shard = CString::new("shardShape").unwrap();
    let field_format = CString::new("zarrFormat").unwrap();

    let field_names: [*const c_char; 5] = [
        field_shape.as_ptr(),
        field_dtype.as_ptr(),
        field_chunk.as_ptr(),
        field_shard.as_ptr(),
        field_format.as_ptr(),
    ];

    let result_struct = unsafe { mxCreateStructMatrix(1, 1, 5, field_names.as_ptr()) };
    if result_struct.is_null() {
        return Err("Failed to create output struct".to_string());
    }

    // Set fields
    unsafe {
        mxSetField(result_struct, 0, field_shape.as_ptr(), shape_arr);
        mxSetField(result_struct, 0, field_dtype.as_ptr(), data_type_arr);
        mxSetField(result_struct, 0, field_format.as_ptr(), zarr_format_arr);

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
