# MATLAB-Zarr
A Zarr v3 implementation based on [zarrs](https://zarrs.dev) for MATLAB.

## Usage

### ZarrArray

Create a new array:

```matlab
% Create a simple array
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', [32, 32, 32]);

% Create with sharding (shard contains multiple chunks)
arr = ZarrArray.create('/path/to/array', [256, 256, 256], 'uint16', ...
    [32, 32, 32], 'shardShape', [128, 128, 128]);

% Create with compression
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'uint16', [32, 32, 32], ...
    'codec', 'zstd');

% Create with compression and custom configuration
arr = ZarrArray.create('/path/to/array', [100, 100, 100], 'float32', [32, 32, 32], ...
    'codec', struct('name', 'zstd', 'configuration', struct('level', 10)));
```

Supported data types: `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`

Supported codecs: `zstd`, `gzip`, `blosc`

Open an existing array and read/write data:

```matlab
% Open existing array
arr = ZarrArray('/path/to/array');

% Get array info
info = arr.info();   % Returns struct with boundingBox, dataType, chunkShape, shardShape
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
seg_data = segmentation.createArray('data', [1000, 1000, 500], 'uint32', [64, 64, 64], ...
    'shardShape', [256, 256, 256], 'codec', 'zstd');
raw_data = raw.createArray('data', [1000, 1000, 500], 'uint8', [64, 64, 64]);

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

Uses the Rust-based [zarrs](https://zarrs.dev) library for the Zarr IO.

Uses the Rust-MATLAB binding originally developed for the [WKW format](https://github.com/scalableminds/webknossos-wrap). Developed by Alessandro Motta at the [Max Planck Institute for Brain Research](https://brain.mpg.de/)

## License
MIT
