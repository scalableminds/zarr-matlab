# MATLAB-Zarr
A Zarr v3 implementation based on [zarrs](https://zarrs.dev) for MATLAB.

## Usage

### ZarrArray

Create a new array:

```matlab
% Create with default options
arr = ZarrArray.create('/path/to/array', [200, 200, 200], 'uint16');

% Create with explicit chunk shape
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', 'chunkShape', [32, 32, 32]);

% Create with sharding (shard contains multiple chunks)
arr = ZarrArray.create('/path/to/array', [256, 256, 256], 'uint16', ...
    'chunkShape', [32, 32, 32], 'shardShape', [128, 128, 128]);

% Create without compression (zstd is the default)
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', 'codec', 'none');

% Create with compression and custom configuration
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'float32', ...
    'chunkShape', [32, 32, 32], 'codec', struct('name', 'zstd', 'configuration', struct('level', 10)));
```

Create an array from existing data (shape and data type are inferred):

```matlab
% Create from existing MATLAB array
data = uint16(rand(100, 100, 100) * 65535);
arr = ZarrArray.createFromData('/path/to/array', data);

% Create from data with gzip instead of default zstd
data = rand(64, 64, 64);  % double -> float64
arr = ZarrArray.createFromData('/path/to/array', data, 'codec', 'gzip');

% Create from data with explicit chunk shape
arr = ZarrArray.createFromData('/path/to/array', data, 'chunkShape', [32, 32, 32], 'codec', 'zstd');
```

Notes:
- Supported data types: `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`
- Compression codecs: `zstd` (default), `gzip`, `blosc`, or `none` to disable compression
- All arrays are created with the `transpose` codec to represent Fortran-order arrays.

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

% Write data
data = uint16(rand(32, 32, 32) * 65535);
arr.write(bbox, data);

% Resize array
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
    'chunkShape', [64, 64, 64], 'shardShape', [256, 256, 256], 'codec', 'zstd');
raw_data = raw.createArray('data', [1000, 1000, 500], 'uint8');

% Create array from existing data
data = uint16(rand(100, 100, 50) * 65535);
arr = root.createArrayFromData('processed', data, 'codec', 'zstd');

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

## Building

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
