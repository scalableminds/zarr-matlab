# MATLAB-Zarr
A Zarr v3 implementation based on [zarrs](https://zarrs.dev) for MATLAB.

## Features

- Read and write Zarr v3 arrays
- Support for chunking and sharding
- Filter codecs: transpose (default)
- Compression codecs: zstd (default), gzip, blosc
- Filesystem access (read-write) and HTTP/HTTPS remote access (read-only)
- Group hierarchy support
- Attributes for arrays and groups
- Supported data types: `bool`, `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`

## Usage

### ZarrArray

Create a new array:

```matlab
% Create with default options
arr = ZarrArray.create('/path/to/array', [200, 200, 200], 'uint16');

% Create with explicit chunk shape
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', 'chunkShape', [32, 32, 32]);

% Create a 1D array
arr = ZarrArray.create('/path/to/array', [10000], 'float64', 'chunkShape', [1000]);

% Create a 2D array
arr = ZarrArray.create('/path/to/array', [1024, 768], 'uint8', 'chunkShape', [256, 256]);

% Create with sharding (shard contains multiple chunks)
arr = ZarrArray.create('/path/to/array', [256, 256, 256], 'uint16', ...
    'chunkShape', [32, 32, 32], 'shardShape', [128, 128, 128]);

% Create without compression (zstd is the default)
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', 'compressors', 'none');

% Create with compressor and custom configuration
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'float32', ...
    'chunkShape', [32, 32, 32], 'compressors', struct('name', 'zstd', 'configuration', struct('level', 10)));

% Create without filters (no transpose, for C-order data)
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', 'filters', 'none');

% Create with no filters and no compression
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', ...
    'filters', 'none', 'compressors', 'none');

% Create with multiple compressors (sequence)
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', ...
    'compressors', {struct('name', 'zstd'), struct('name', 'crc32c')});

% Create with custom fill value for uninitialized chunks
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'float32', 'fillValue', NaN);

% Create with v2-style chunk key encoding (uses '.' separator)
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', ...
    'chunkKeyEncoding', struct('name', 'v2', 'separator', '.'));
```

Create an array from existing data (shape and data type are inferred):

```matlab
% Create from existing MATLAB array
data = uint16(rand(100, 100, 100) * 65535);
arr = ZarrArray.createFromData('/path/to/array', data);

% Create from data with gzip instead of default zstd
data = rand(64, 64, 64);  % double -> float64
arr = ZarrArray.createFromData('/path/to/array', data, 'compressors', 'gzip');

% Create from data with explicit chunk shape
arr = ZarrArray.createFromData('/path/to/array', data, 'chunkShape', [32, 32, 32]);

% Create from logical array (stored as bool)
mask = rand(64, 64, 64) > 0.5;  % logical -> bool
arr = ZarrArray.createFromData('/path/to/mask', mask);
```

Notes:
- **Filters**: Applied before bytes codec. Default is `transpose` (to store Fortran-order data). Use `'none'` to disable.
- **Compressors**: Applied after bytes codec. Default is `zstd`. Supported: `zstd`, `gzip`, `blosc`, or `'none'` to disable.
- Both `filters` and `compressors` accept: a single codec (string or struct), a cell array of codecs (sequence), or `'none'`. 
Open an existing array and read/write data:

```matlab
% Open existing array
arr = ZarrArray('/path/to/array');

% Get array info
info = arr.info();   % Returns struct with shape, dataType, chunkShape, shardShape
shape = arr.shape(); % Returns shape as vector

% Read data (bounding box is [start, end] for each dimension, 1-indexed)
bbox = [1, 33; 1, 33; 1, 33];  % Read a 32x32x32 region
data = arr.read(bbox);

% Read entire array (bbox is optional)
allData = arr.read();

% Write data to a region
data = uint16(rand(32, 32, 32) * 65535);
arr.write(data, bbox);

% Write entire array at origin (bbox is optional)
fullData = uint16(rand(100, 100, 100) * 65535);
arr.write(fullData);  % Writes starting at [1, 1, 1]

% Write with automatic resize (extends array if needed)
largeData = uint16(rand(50, 50, 50) * 65535);
arr.write(largeData, [151, 200; 151, 200; 151, 200], 'allowResize', true);

% Write to 1D arrays as column vector
arr.write([34; 94; 28]);

% Resize array explicitly
arr.resize([200, 200, 200]);
```

### ZarrGroup

Create and navigate group hierarchies:

```matlab
% Create a new group
root = ZarrGroup.create('/path/to/dataset');

% Create subgroups
segmentation = root.createGroup('segmentation');
raw = root.createGroup('raw');

% Create arrays within groups
seg_data = segmentation.createArray('data', [1000, 1000, 500], 'uint32', ...
    'chunkShape', [64, 64, 64], 'shardShape', [256, 256, 256]);
raw_data = raw.createArray('data', [1000, 1000, 500], 'uint8');

% Create array from existing data
data = uint16(rand(100, 100, 50) * 65535);
arr = root.createArrayFromData('processed', data);

% List contents
names = root.list();                    % Returns {'raw', 'segmentation'}
[groups, arrays] = root.listContents(); % Separate groups and arrays
```

Open existing groups:

```matlab
% Open existing group
root = ZarrGroup('/path/to/dataset');

% Navigate to subgroups
segmentation = root.openGroup('segmentation');

% Open arrays within groups
arr = segmentation.openArray('data');
data = arr.read([1, 65; 1, 65; 1, 65]);
```

### Remote Access (HTTP)

Read from remote Zarr arrays and groups:

```matlab
% Open remote array
arr = ZarrArray('https://example.com/data/zarr_v3/dataset/segmentation/1');
info = arr.info();
data = arr.read([1, 33; 1, 33; 1, 33]);

% Open remote group and navigate
root = ZarrGroup('https://example.com/data/zarr_v3/dataset');
segmentation = root.openGroup('segmentation');
arr = segmentation.openArray('1');
```

Note: Remote access is read-only. The `list()` and `listContents()` methods are not available for HTTP URLs.

### Attributes

Both arrays and groups support attributes:

```matlab
% Set attributes on a group
grp = ZarrGroup.create('/path/to/dataset');
grp.setAttribute('name', 'My Dataset');
grp.setAttribute('resolution', [4, 4, 30]);

% Set multiple attributes at once
grp.setAttributes(struct('version', 2, 'author', 'John Doe'));

% Get attributes
name = grp.getAttribute('name');
attrs = grp.getAttributes();  % Returns struct with all attributes

% Attributes work the same way on arrays
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16');
arr.setAttribute('units', 'nm');
arr.setAttribute('scale', [1.0, 1.0, 2.0]);
```

Note: Writing attributes is not available for HTTP URLs (read-only).

## Installation

### From Toolbox (Recommended)

Download `zarr-matlab.mltbx` from the releases page and double-click to install, or run:

```matlab
matlab.addons.install('zarr-matlab.mltbx')
```

### Building from Source

Requires Rust toolchain installed.

```matlab
cd zarr-matlab
zarrBuild()
```

## Running Tests

```matlab
cd zarr-matlab
results = runtests('tests');
```

## Credits

Uses the Rust-based [zarrs](https://zarrs.dev) library for the Zarr IO. Developed by [Lachlan Deakin](https://github.com/LDeakin) and other contributors.

Uses the Rust-MATLAB binding originally developed for the [WKW format](https://github.com/scalableminds/webknossos-wrap). Developed by Alessandro Motta at the [Max Planck Institute for Brain Research](https://brain.mpg.de/)

## License
MIT
