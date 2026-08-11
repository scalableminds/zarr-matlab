# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "zarr>=3,<4",
#     "numcodecs>=0.16",
# ]
# ///
"""zarr-python side of the zarr-matlab <-> zarr-python interop tests.

Invoked by the MATLAB test classes in this directory via `system('uv run
zarr_interop_cli.py ...')`. `uv run` resolves the dependencies declared in the
inline script metadata above automatically -- no separate install step.

Subcommands:
  write-array   Create an array with zarr-python, filled with deterministic
                data (see deterministic_data below), for zarr-matlab to read.
  check-array   Open an array zarr-matlab wrote and verify its shape, dtype,
                data and (optionally) attributes match the same deterministic
                formula. Exits nonzero with a message on stderr on mismatch.
  write-group   Same idea, for a small two-level group hierarchy.
  check-group   Same idea, for a small two-level group hierarchy.

All arguments are plain tokens (no embedded JSON, no shell-quoting-sensitive
values) deliberately: MATLAB's `system()` invokes `cmd.exe` on Windows, which
does not treat `'` as a quote character the way POSIX shells do, so a
`--foo '{"a":1}'`-style argument that is safe on Linux/macOS is not safe
there. Compressor/filter selection is therefore a handful of named flags
rather than a JSON blob.

Deterministic data
-------------------
Both the Python side (here) and the MATLAB side (deterministicData in
ZarrPythonInteropTest.m) independently regenerate identical N-D test arrays
from (shape, dtype) alone -- no data interchange file, no shared random seed.
Zarr's codec pipeline already makes `arr[i,j,k,...]` addressing agree between
writer and reader regardless of on-disk byte layout, so the two languages only
need to agree on value-as-a-function-of-logical-index:

  1. Build the 0-based linear index, reshaped in column-major (Fortran) order
     (MATLAB's `reshape` is column-major natively; here `order="F"` is passed
     explicitly) -- so both sides get the same N-D array of index values.
  2. Apply a per-dtype-tier `mod`/subtract, all exact in float64 so both
     languages agree bit-for-bit (no rounding-sensitive functions like sin or
     rand are used anywhere).
  3. Cast to the target dtype.

KEEP THIS FORMULA AND `DEFAULT_ATTRS` IN SYNC WITH ZarrPythonInteropTest.m.
"""

import argparse
import json
import sys

import numcodecs
import numpy as np
import zarr
import zarr.codecs

# Keep in sync with ZarrPythonInteropTest.m's defaultAttrs().
DEFAULT_ATTRS = {
    "description": "zarr-matlab interop fixture",
    "scale": 2.5,
    "channel": 3,
    "flag": True,
}

DTYPE_CHOICES = [
    "bool",
    "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64",
    "float32", "float64",
]

COMPRESSOR_CHOICES = ["zstd", "gzip", "zlib", "bz2", "blosc", "none"]

# numpy dtype "kind+size" code (no byteorder prefix) per Zarr dtype name.
_NUMPY_CODE = {
    "bool": "?", "uint8": "u1", "int8": "i1",
    "uint16": "u2", "int16": "i2", "uint32": "u4", "int32": "i4",
    "uint64": "u8", "int64": "i8", "float32": "f4", "float64": "f8",
}

_BLOSC_SHUFFLE_V2 = {
    "noshuffle": numcodecs.Blosc.NOSHUFFLE,
    "shuffle": numcodecs.Blosc.SHUFFLE,
    "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
}


def fail(message):
    print(f"CHECK FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def parse_shape(text):
    return tuple(int(x) for x in text.split(",") if x.strip())


def parse_fill_value(text):
    if text is None:
        return None
    low = text.strip().lower()
    if low == "nan":
        return float("nan")
    if low in ("inf", "infinity"):
        return float("inf")
    if low in ("-inf", "-infinity"):
        return float("-inf")
    return json.loads(text)


def numpy_dtype(dtype_name, byteorder="little"):
    if dtype_name == "bool":
        return np.dtype("bool")
    prefix = "<" if byteorder == "little" else ">"
    return np.dtype(f"{prefix}{_NUMPY_CODE[dtype_name]}")


def v2_dtype_string(dtype_name, byteorder="little"):
    """Numpy-style dtype string for a Zarr V2 .zarray 'dtype' field."""
    if dtype_name == "bool":
        return "|b1"
    if dtype_name in ("uint8", "int8"):
        return f"|{_NUMPY_CODE[dtype_name]}"
    prefix = "<" if byteorder == "little" else ">"
    return f"{prefix}{_NUMPY_CODE[dtype_name]}"


def deterministic_data(shape, dtype_name, byteorder="little"):
    n = 1
    for d in shape:
        n *= d
    idx = np.arange(n, dtype=np.float64).reshape(shape, order="F")

    if dtype_name == "bool":
        return np.mod(idx, 2) != 0
    if dtype_name in ("uint8", "int8"):
        raw = np.mod(idx, 256)
        if dtype_name == "int8":
            raw = raw - 128
    elif dtype_name in ("uint16", "int16"):
        raw = np.mod(idx, 60000)
        if dtype_name == "int16":
            raw = raw - 30000
    else:
        raw = np.mod(idx, 1000003)
        if dtype_name.startswith("int") or dtype_name.startswith("float"):
            raw = raw - 500001

    return raw.astype(numpy_dtype(dtype_name, byteorder))


def build_named_compressor(name, zarr_format, dtype_size, blosc_cname, blosc_clevel, blosc_shuffle, level):
    """A single named compressor -> a numcodecs codec (v2), a zarr.codecs
    codec (v3), or None/[] for 'no compression' (v2/v3 respectively)."""
    if name is None or name == "none":
        return None if zarr_format == 2 else []

    if zarr_format == 2:
        if name == "zstd":
            return numcodecs.Zstd(level=level if level is not None else 3)
        if name == "gzip":
            return numcodecs.GZip(level=level if level is not None else 5)
        if name == "zlib":
            return numcodecs.Zlib(level=level if level is not None else 5)
        if name == "bz2":
            return numcodecs.BZ2(level=level if level is not None else 1)
        if name == "blosc":
            return numcodecs.Blosc(
                cname=blosc_cname or "lz4",
                clevel=blosc_clevel if blosc_clevel is not None else 5,
                shuffle=_BLOSC_SHUFFLE_V2[blosc_shuffle or "noshuffle"],
            )
        raise ValueError(f"Unsupported Zarr V2 compressor: {name!r}")

    # zarr_format == 3
    if name in ("zlib", "bz2"):
        raise ValueError(f"{name!r} is a Zarr V2-only (numcodecs) compressor, not available under zarr_format=3")
    if name == "zstd":
        return zarr.codecs.ZstdCodec(level=level if level is not None else 3, checksum=False)
    if name == "gzip":
        return zarr.codecs.GzipCodec(level=level if level is not None else 5)
    if name == "blosc":
        return zarr.codecs.BloscCodec(
            cname=blosc_cname or "lz4",
            clevel=blosc_clevel if blosc_clevel is not None else 5,
            shuffle=blosc_shuffle or "noshuffle",
            typesize=dtype_size,
        )
    raise ValueError(f"Unsupported Zarr V3 compressor: {name!r}")


def build_compressors(args, dtype_size):
    """Resolve --compressor / --compressor-sequence into the compressors=
    kwarg value for zarr.create_array()/Group.create_array()."""
    sequence = getattr(args, "compressor_sequence", None)
    if sequence:
        if args.zarr_format == 2:
            raise ValueError("--compressor-sequence is Zarr V3 only (Zarr V2 has a single compressor slot)")
        names = [n.strip() for n in sequence.split(",") if n.strip()]
        return [
            build_named_compressor(n, args.zarr_format, dtype_size, None, None, None, None)
            for n in names
        ]
    return build_named_compressor(
        getattr(args, "compressor", None), args.zarr_format, dtype_size,
        getattr(args, "blosc_cname", None), getattr(args, "blosc_clevel", None),
        getattr(args, "blosc_shuffle", None), getattr(args, "compressor_level", None),
    )


def build_filters(filters_name, ndim):
    """--filters {transpose,none} -> V3 filters= value, or None (meaning:
    don't pass filters= at all, let zarr-python pick its own default)."""
    if filters_name is None:
        return None
    if filters_name == "none":
        return []
    if filters_name == "transpose":
        order = tuple(range(ndim - 1, -1, -1))
        return [zarr.codecs.TransposeCodec(order=order)]
    raise ValueError(f"Unsupported Zarr V3 filter: {filters_name!r}")


def build_array_create_kwargs(args, shape):
    """Kwargs for zarr.create_array()/Group.create_array(), shared by the
    array and group write subcommands."""
    dtype_size = numpy_dtype(args.dtype).itemsize
    kwargs = {
        "chunks": parse_shape(args.chunks),
        "compressors": build_compressors(args, dtype_size),
    }

    if args.zarr_format == 2:
        kwargs["dtype"] = v2_dtype_string(args.dtype, getattr(args, "byteorder", "little"))
        if getattr(args, "order", None):
            kwargs["order"] = args.order
        separator = {"slash": "/", "dot": "."}[getattr(args, "separator", None) or "dot"]
        kwargs["chunk_key_encoding"] = {"name": "v2", "separator": separator}
    else:
        kwargs["dtype"] = args.dtype
        filters = build_filters(getattr(args, "filters", None), len(shape))
        if filters is not None:
            kwargs["filters"] = filters
        separator = {"slash": "/", "dot": "."}[getattr(args, "separator", None) or "slash"]
        kwargs["chunk_key_encoding"] = {"name": "default", "separator": separator}
        shards = getattr(args, "shards", None)
        if shards:
            kwargs["shards"] = parse_shape(shards)

    fill_value = parse_fill_value(getattr(args, "fill_value", None))
    if fill_value is not None:
        kwargs["fill_value"] = fill_value

    return kwargs


def cmd_write_array(args):
    shape = parse_shape(args.shape)
    kwargs = build_array_create_kwargs(args, shape)

    arr = zarr.create_array(store=args.path, shape=shape, zarr_format=args.zarr_format, **kwargs)

    if not args.fill_only:
        arr[...] = deterministic_data(shape, args.dtype, getattr(args, "byteorder", "little"))
    if args.attrs:
        arr.attrs.update(DEFAULT_ATTRS)

    print("OK")
    return 0


def cmd_check_array(args):
    shape = parse_shape(args.shape)
    arr = zarr.open_array(store=args.path, mode="r")

    if args.zarr_format is not None and arr.metadata.zarr_format != args.zarr_format:
        fail(f"zarr_format mismatch: expected {args.zarr_format}, got {arr.metadata.zarr_format}")
    if tuple(arr.shape) != shape:
        fail(f"shape mismatch: expected {shape}, got {tuple(arr.shape)}")

    expected_kind_size = numpy_dtype(args.dtype)
    if arr.dtype.kind != expected_kind_size.kind or arr.dtype.itemsize != expected_kind_size.itemsize:
        fail(f"dtype mismatch: expected kind/size {expected_kind_size.kind}{expected_kind_size.itemsize}, "
             f"got {arr.dtype.kind}{arr.dtype.itemsize} ({arr.dtype})")

    actual = arr[...]
    if args.fill_only:
        fill_value = parse_fill_value(args.fill_value)
        expected = np.full(shape, fill_value, dtype=actual.dtype)
    else:
        expected = deterministic_data(shape, args.dtype)

    if not np.array_equal(actual, expected, equal_nan=True):
        mismatches = int(np.sum(~np.isclose(actual, expected, equal_nan=True))) if actual.dtype.kind == "f" \
            else int(np.sum(actual != expected))
        fail(f"data mismatch: {mismatches}/{actual.size} elements differ")

    if args.attrs:
        actual_attrs = dict(arr.attrs)
        if actual_attrs != DEFAULT_ATTRS:
            fail(f"attrs mismatch: expected {DEFAULT_ATTRS}, got {actual_attrs}")

    print("OK")
    return 0


def cmd_write_group(args):
    g = zarr.open_group(store=args.path, mode="w", zarr_format=args.zarr_format)
    g.attrs.update(DEFAULT_ATTRS)

    sub = g.create_group("child")
    sub.attrs.update(DEFAULT_ATTRS)

    shape = parse_shape(args.array_shape)
    dtype_size = numpy_dtype(args.array_dtype).itemsize
    compressors = build_named_compressor(
        getattr(args, "compressor", None), args.zarr_format, dtype_size, None, None, None, None,
    )

    dtype = v2_dtype_string(args.array_dtype) if args.zarr_format == 2 else args.array_dtype
    arr = sub.create_array(
        name="data",
        shape=shape,
        chunks=parse_shape(args.array_chunks),
        dtype=dtype,
        compressors=compressors,
    )
    arr[...] = deterministic_data(shape, args.array_dtype)

    print("OK")
    return 0


def cmd_check_group(args):
    g = zarr.open_group(store=args.path, mode="r")

    if args.zarr_format is not None and g.metadata.zarr_format != args.zarr_format:
        fail(f"group zarr_format mismatch: expected {args.zarr_format}, got {g.metadata.zarr_format}")
    if dict(g.attrs) != DEFAULT_ATTRS:
        fail(f"group attrs mismatch: expected {DEFAULT_ATTRS}, got {dict(g.attrs)}")

    sub = g["child"]
    if dict(sub.attrs) != DEFAULT_ATTRS:
        fail(f"subgroup attrs mismatch: expected {DEFAULT_ATTRS}, got {dict(sub.attrs)}")

    arr = sub["data"]
    shape = parse_shape(args.array_shape)
    if tuple(arr.shape) != shape:
        fail(f"array shape mismatch: expected {shape}, got {tuple(arr.shape)}")

    actual = arr[...]
    expected = deterministic_data(shape, args.array_dtype)
    if not np.array_equal(actual, expected, equal_nan=True):
        fail("group array data mismatch")

    print("OK")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, zarr_format_required):
        sub.add_argument("--path", required=True)
        sub.add_argument("--zarr-format", type=int, choices=(2, 3), required=zarr_format_required)

    def add_array_common(sub):
        sub.add_argument("--shape", required=True)
        sub.add_argument("--dtype", required=True, choices=DTYPE_CHOICES)
        sub.add_argument("--attrs", action="store_true")
        sub.add_argument("--fill-only", action="store_true")

    def add_compressor_flags(sub):
        sub.add_argument("--compressor", choices=COMPRESSOR_CHOICES)
        sub.add_argument("--compressor-sequence", help="comma-separated compressor names, V3 only, e.g. zstd,gzip")
        sub.add_argument("--compressor-level", type=int)
        sub.add_argument("--blosc-cname")
        sub.add_argument("--blosc-clevel", type=int)
        sub.add_argument("--blosc-shuffle", choices=("noshuffle", "shuffle", "bitshuffle"))

    write_array = subparsers.add_parser("write-array")
    add_common(write_array, zarr_format_required=True)
    add_array_common(write_array)
    add_compressor_flags(write_array)
    write_array.add_argument("--chunks", required=True)
    write_array.add_argument("--filters", choices=("transpose", "none"))
    write_array.add_argument("--order", choices=("C", "F"))
    write_array.add_argument("--separator", choices=("slash", "dot"))
    write_array.add_argument("--fill-value")
    write_array.add_argument("--byteorder", choices=("little", "big"), default="little")
    write_array.add_argument("--shards")
    write_array.set_defaults(func=cmd_write_array)

    check_array = subparsers.add_parser("check-array")
    add_common(check_array, zarr_format_required=False)
    add_array_common(check_array)
    check_array.add_argument("--fill-value")
    check_array.set_defaults(func=cmd_check_array)

    write_group = subparsers.add_parser("write-group")
    add_common(write_group, zarr_format_required=True)
    write_group.add_argument("--array-shape", required=True)
    write_group.add_argument("--array-chunks", required=True)
    write_group.add_argument("--array-dtype", required=True, choices=DTYPE_CHOICES)
    write_group.add_argument("--compressor", choices=COMPRESSOR_CHOICES)
    write_group.set_defaults(func=cmd_write_group)

    check_group = subparsers.add_parser("check-group")
    add_common(check_group, zarr_format_required=False)
    check_group.add_argument("--array-shape", required=True)
    check_group.add_argument("--array-dtype", required=True, choices=DTYPE_CHOICES)
    check_group.set_defaults(func=cmd_check_group)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as err:
        # Argument-validation errors (unsupported compressor/format combos,
        # etc.) -- report cleanly rather than as a Python traceback, since
        # this message is what surfaces in the calling MATLAB test's failure.
        fail(str(err))


if __name__ == "__main__":
    sys.exit(main())
