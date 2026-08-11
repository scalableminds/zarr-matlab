classdef ZarrArrayV2Test < matlab.unittest.TestCase

    properties (Constant)
        % Zarr v2 copy of the dataset used by ZarrMexTest.RemotePath. The array
        % is 4-D <u4, order 'F', compressed with numcodecs blosc/zstd.
        RemoteV2Path = 'https://static.webknossos.org/data/l4_sample/segmentation/1';
    end

    properties (Access = private)
        TempDir
    end

    methods (TestClassSetup)
        function addParentToPath(~)
            parentDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(parentDir);
        end
    end

    methods (TestMethodSetup)
        function setupTempDir(testCase)
            testCase.TempDir = fullfile(tempdir, 'zarr_matlab_test');
            if isfolder(testCase.TempDir)
                rmdir(testCase.TempDir, 's');
            end
            mkdir(testCase.TempDir);
        end
    end

    methods (TestMethodTeardown)
        function removeTempDir(testCase)
            if isfolder(testCase.TempDir)
                rmdir(testCase.TempDir, 's');
            end
        end
    end

    methods (Test)
        function testCreate(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_create');

            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', ...
                'chunkShape', [32, 32, 32], 'zarrFormat', 2);

            % v2 writes .zarray and must not write a v3 zarr.json
            testCase.verifyTrue(isfile(fullfile(arrayPath, '.zarray')));
            testCase.verifyFalse(isfile(fullfile(arrayPath, 'zarr.json')));

            info = arr.info();
            testCase.verifyEqual(info.shape, [100, 100, 100]);
            testCase.verifyEqual(info.dataType, 'uint16');
            testCase.verifyEqual(info.chunkShape, [32, 32, 32]);
            testCase.verifyEqual(info.shardShape, [32, 32, 32]);
            testCase.verifyEqual(info.zarrFormat, 2);
            testCase.verifyEqual(arr.zarrFormat(), 2);
        end

        function testMetadataFields(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_metadata');
            ZarrArray.create(arrayPath, [64, 48], 'int16', ...
                'chunkShape', [32, 16], 'zarrFormat', 2);

            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));

            testCase.verifyEqual(metadata.zarr_format, 2);
            testCase.verifyEqual(metadata.shape', [64, 48]);
            testCase.verifyEqual(metadata.chunks', [32, 16]);
            testCase.verifyEqual(metadata.dtype, '<i2');
            % MATLAB is column-major, so the default is Fortran order rather
            % than the v3 path's transpose codec
            testCase.verifyEqual(metadata.order, 'F');
            testCase.verifyEqual(metadata.fill_value, 0);
            testCase.verifyEqual(metadata.dimension_separator, '/');
            testCase.verifyEmpty(metadata.filters);
            testCase.verifyEqual(metadata.compressor.id, 'zstd');
            testCase.verifyEqual(metadata.compressor.level, 3);
        end

        function testDtypeMapping(testCase)
            expected = { ...
                'bool', '|b1'; ...
                'uint8', '|u1'; 'int8', '|i1'; ...
                'uint16', '<u2'; 'int16', '<i2'; ...
                'uint32', '<u4'; 'int32', '<i4'; ...
                'uint64', '<u8'; 'int64', '<i8'; ...
                'float32', '<f4'; 'float64', '<f8' };

            for i = 1:size(expected, 1)
                dataType = expected{i, 1};
                arrayPath = fullfile(testCase.TempDir, ['v2_dtype_', dataType]);
                arr = ZarrArray.create(arrayPath, [8, 8], dataType, ...
                    'chunkShape', [4, 4], 'zarrFormat', 2);

                metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
                testCase.verifyEqual(metadata.dtype, expected{i, 2}, ...
                    sprintf('dtype mismatch for %s', dataType));
                testCase.verifyEqual(arr.dataType(), dataType);
            end
        end

        function testRoundTripAllDataTypes(testCase)
            % Zarr data type name -> MATLAB class to cast the test data to
            dataTypes = { ...
                'uint8', 'uint8'; 'int8', 'int8'; ...
                'uint16', 'uint16'; 'int16', 'int16'; ...
                'uint32', 'uint32'; 'int32', 'int32'; ...
                'uint64', 'uint64'; 'int64', 'int64'; ...
                'float32', 'single'; 'float64', 'double' };

            for i = 1:size(dataTypes, 1)
                dataType = dataTypes{i, 1};
                arrayPath = fullfile(testCase.TempDir, ['v2_roundtrip_', dataType]);
                arr = ZarrArray.create(arrayPath, [16, 20], dataType, ...
                    'chunkShape', [8, 8], 'zarrFormat', 2);

                testData = cast(randi([0, 100], [16, 20]), dataTypes{i, 2});
                arr.write(testData);

                testCase.verifyEqual(arr.read(), testData, ...
                    sprintf('round-trip mismatch for %s', dataType));
            end
        end

        function testRoundTripBool(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_bool');
            arr = ZarrArray.create(arrayPath, [32, 32, 32], 'bool', ...
                'chunkShape', [16, 16, 16], 'zarrFormat', 2);

            testData = logical(randi([0, 1], [32, 32, 32]));
            arr.write(testData);

            testCase.verifyEqual(arr.read(), testData);
        end

        function testCreate1DArray(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_1d');
            arr = ZarrArray.create(arrayPath, [1000], 'float64', ...
                'chunkShape', [100], 'zarrFormat', 2);

            testData = rand(200, 1);
            bbox = [1, 201];
            arr.write(testData, bbox);

            testCase.verifyEqual(arr.read(bbox), testData);
            testCase.verifyEqual(size(arr.read()), [1000, 1]);
        end

        function testCreate4DArray(testCase)
            % Matches the axis layout of the remote v2 fixture
            arrayPath = fullfile(testCase.TempDir, 'v2_4d');
            arr = ZarrArray.create(arrayPath, [1, 32, 32, 16], 'uint32', ...
                'chunkShape', [1, 16, 16, 8], 'zarrFormat', 2);

            testData = uint32(randi(1e6, [1, 32, 32, 16]));
            arr.write(testData);

            testCase.verifyEqual(arr.read(), testData);
        end

        function testReadWriteWithBbox(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_bbox');
            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', ...
                'chunkShape', [32, 32, 32], 'zarrFormat', 2);

            testData = uint16(reshape(1:1000, [10, 10, 10]));
            bbox = [11, 21; 21, 31; 31, 41];

            arr.write(testData, bbox);

            testCase.verifyEqual(arr.read(bbox), testData);
        end

        function testResize(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_resize');
            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', ...
                'chunkShape', [32, 32, 32], 'zarrFormat', 2);

            arr.resize([150, 120, 100]);

            testCase.verifyEqual(arr.shape(), [150, 120, 100]);

            % The rewritten metadata must still be v2
            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
            testCase.verifyEqual(metadata.shape', [150, 120, 100]);
            testCase.verifyEqual(metadata.zarr_format, 2);
            testCase.verifyFalse(isfile(fullfile(arrayPath, 'zarr.json')));
        end

        function testWriteWithAllowResize(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_allow_resize');
            arr = ZarrArray.create(arrayPath, [20, 20, 20], 'uint8', ...
                'chunkShape', [10, 10, 10], 'zarrFormat', 2);

            testData = uint8(randi(255, [15, 15, 15]));
            bbox = [10, 25; 10, 25; 10, 25];

            arr.write(testData, bbox, 'allowResize', true);

            testCase.verifyEqual(arr.shape(), [24, 24, 24]);
            testCase.verifyEqual(arr.read(bbox), testData);
        end

        function testCreateFromData(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_from_data');
            testData = uint16(randi(65535, [50, 40, 30]));

            arr = ZarrArray.createFromData(arrayPath, testData, ...
                'chunkShape', [16, 16, 16], 'zarrFormat', 2);

            testCase.verifyEqual(arr.zarrFormat(), 2);
            testCase.verifyEqual(arr.shape(), [50, 40, 30]);
            testCase.verifyEqual(arr.read(), testData);
        end

        function testOpenExisting(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_open');
            ZarrArray.create(arrayPath, [64, 64], 'uint32', ...
                'chunkShape', [32, 32], 'zarrFormat', 2);

            % Opening is format-agnostic: no zarrFormat argument needed
            arr = ZarrArray(arrayPath);

            testCase.verifyEqual(arr.shape(), [64, 64]);
            testCase.verifyEqual(arr.zarrFormat(), 2);
        end

        function testEquivalentToV3(testCase)
            % The same data written as v2 (order 'F') and as v3 (transpose
            % codec) must read back identically.
            testData = uint16(randi(65535, [40, 30, 20]));

            v2Path = fullfile(testCase.TempDir, 'equiv_v2');
            v3Path = fullfile(testCase.TempDir, 'equiv_v3');

            v2Arr = ZarrArray.createFromData(v2Path, testData, ...
                'chunkShape', [16, 16, 16], 'zarrFormat', 2);
            v3Arr = ZarrArray.createFromData(v3Path, testData, ...
                'chunkShape', [16, 16, 16], 'zarrFormat', 3);

            testCase.verifyEqual(v2Arr.read(), testData);
            testCase.verifyEqual(v2Arr.read(), v3Arr.read());
        end

        % Compressor Tests

        function testWithZstdCompressor(testCase)
            testCase.verifyCompressorRoundTrip('v2_zstd', 'zstd', 'zstd');
        end

        function testWithGzipCompressor(testCase)
            testCase.verifyCompressorRoundTrip('v2_gzip', 'gzip', 'gzip');
        end

        function testWithZlibCompressor(testCase)
            % zlib is zarr-python's classic v2 default; it needs the zarrs
            % 'zlib' cargo feature, which is off by default upstream.
            testCase.verifyCompressorRoundTrip('v2_zlib', 'zlib', 'zlib');
        end

        function testWithBz2Compressor(testCase)
            testCase.verifyCompressorRoundTrip('v2_bz2', 'bz2', 'bz2');
        end

        function testWithBloscCompressor(testCase)
            arrayPath = testCase.verifyCompressorRoundTrip('v2_blosc', 'blosc', 'blosc');

            % numcodecs blosc takes an integer shuffle, not the v3 string form,
            % and carries no typesize
            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
            testCase.verifyEqual(metadata.compressor.cname, 'lz4');
            testCase.verifyEqual(metadata.compressor.clevel, 5);
            testCase.verifyEqual(metadata.compressor.shuffle, 0);
            testCase.verifyFalse(isfield(metadata.compressor, 'typesize'));
        end

        function testWithBloscCompressorConfigured(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_blosc_config');
            bloscConfig = struct('cname', 'zstd', 'clevel', 9, 'shuffle', 1, 'blocksize', 0);
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'float32', ...
                'chunkShape', [32, 32, 32], 'zarrFormat', 2, ...
                'compressors', struct('name', 'blosc', 'configuration', bloscConfig));

            testData = single(rand(32, 32, 32));
            bbox = [1, 33; 1, 33; 1, 33];
            arr.write(testData, bbox);

            testCase.verifyEqual(arr.read(bbox), testData);

            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
            testCase.verifyEqual(metadata.compressor.cname, 'zstd');
            testCase.verifyEqual(metadata.compressor.shuffle, 1);
        end

        function testWithNoCompression(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_no_compress');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'uint8', ...
                'chunkShape', [32, 32, 32], 'zarrFormat', 2, 'compressors', 'none');

            % 'none' must produce a JSON null, not an empty array
            raw = fileread(fullfile(arrayPath, '.zarray'));
            testCase.verifyTrue(contains(raw, '"compressor": null') ...
                || contains(raw, '"compressor":null'));

            testData = uint8(randi(255, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];
            arr.write(testData, bbox);

            testCase.verifyEqual(arr.read(bbox), testData);
        end

        % Order and Chunk Key Tests

        function testWithNoFiltersUsesCOrder(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_c_order');
            arr = ZarrArray.create(arrayPath, [64, 48, 32], 'uint16', ...
                'chunkShape', [32, 16, 16], 'zarrFormat', 2, 'filters', 'none');

            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
            testCase.verifyEqual(metadata.order, 'C');

            % Byte order on disk must not change what MATLAB sees
            testData = uint16(randi(65535, [64, 48, 32]));
            arr.write(testData);
            testCase.verifyEqual(arr.read(), testData);
        end

        function testWithDotSeparator(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_dot_sep');
            arr = ZarrArray.create(arrayPath, [64, 64], 'uint8', ...
                'chunkShape', [32, 32], 'zarrFormat', 2, 'chunkKeyEncoding', '.');

            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
            testCase.verifyEqual(metadata.dimension_separator, '.');

            testData = uint8(randi(255, [32, 32]));
            arr.write(testData, [1, 33; 1, 33]);

            % Flat chunk key, dot-separated
            testCase.verifyTrue(isfile(fullfile(arrayPath, '0.0')));
            testCase.verifyEqual(arr.read([1, 33; 1, 33]), testData);
        end

        function testWithSlashSeparator(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_slash_sep');
            arr = ZarrArray.create(arrayPath, [64, 64], 'uint8', ...
                'chunkShape', [32, 32], 'zarrFormat', 2);

            testData = uint8(randi(255, [32, 32]));
            arr.write(testData, [1, 33; 1, 33]);

            % Nested chunk key, one directory level per dimension
            testCase.verifyTrue(isfile(fullfile(arrayPath, '0', '0')));
            testCase.verifyEqual(arr.read([1, 33; 1, 33]), testData);
        end

        function testWithCustomFillValue(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_fillvalue');
            arr = ZarrArray.create(arrayPath, [64, 64], 'uint16', ...
                'chunkShape', [32, 32], 'zarrFormat', 2, 'fillValue', 42);

            testCase.verifyEqual(arr.read([1, 33; 1, 33]), uint16(ones(32, 32) * 42));
        end

        function testWithNaNFillValue(testCase)
            % jsonencode maps NaN to null, which in v2 means "no fill value",
            % so a NaN fill value has to be written as the string "NaN"
            arrayPath = fullfile(testCase.TempDir, 'v2_nan_fill');
            arr = ZarrArray.create(arrayPath, [16, 16], 'float64', ...
                'chunkShape', [8, 8], 'zarrFormat', 2, 'fillValue', NaN);

            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
            testCase.verifyEqual(metadata.fill_value, 'NaN');

            testCase.verifyEqual(arr.read(), NaN(16, 16));
        end

        % Attribute Tests

        function testGetAttributesEmpty(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_attrs_empty');
            arr = ZarrArray.create(arrayPath, [32, 32], 'uint8', ...
                'chunkShape', [16, 16], 'zarrFormat', 2);

            attrs = arr.getAttributes();

            testCase.verifyTrue(isstruct(attrs));
            testCase.verifyEmpty(fieldnames(attrs));
        end

        function testSetAndGetAttributes(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_attrs');
            arr = ZarrArray.create(arrayPath, [32, 32], 'uint16', ...
                'chunkShape', [16, 16], 'zarrFormat', 2);

            arr.setAttributes(struct('description', 'test array', 'scale', 2.5));

            % v2 attributes live in their own .zattrs file
            testCase.verifyTrue(isfile(fullfile(arrayPath, '.zattrs')));

            readAttrs = arr.getAttributes();
            testCase.verifyEqual(readAttrs.description, 'test array');
            testCase.verifyEqual(readAttrs.scale, 2.5);

            % .zarray must be left alone
            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
            testCase.verifyFalse(isfield(metadata, 'attributes'));
        end

        function testAttributesPersist(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_attrs_persist');
            arr = ZarrArray.create(arrayPath, [32, 32], 'int16', ...
                'chunkShape', [16, 16], 'zarrFormat', 2);
            arr.setAttribute('key', 'value');

            arr2 = ZarrArray(arrayPath);

            testCase.verifyEqual(arr2.getAttribute('key'), 'value');
        end

        function testAttributesSurviveResize(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_attrs_resize');
            arr = ZarrArray.create(arrayPath, [32, 32], 'uint8', ...
                'chunkShape', [16, 16], 'zarrFormat', 2);
            arr.setAttribute('key', 'value');

            arr.resize([64, 64]);

            testCase.verifyEqual(arr.getAttribute('key'), 'value');
        end

        % Interop Tests

        function testOpenExternallyWrittenArray(testCase)
            % Hand-written .zarray in the shape zarr-python produces: C order,
            % dot separator, zlib compressor, and no node_type field.
            arrayPath = fullfile(testCase.TempDir, 'v2_external');
            mkdir(arrayPath);
            fid = fopen(fullfile(arrayPath, '.zarray'), 'w');
            fprintf(fid, '%s', jsonencode(struct( ...
                'zarr_format', 2, ...
                'shape', {{8, 8}}, ...
                'chunks', {{4, 4}}, ...
                'dtype', '<u2', ...
                'compressor', struct('id', 'zlib', 'level', 1), ...
                'fill_value', 7, ...
                'order', 'C', ...
                'filters', {NaN}, ...
                'dimension_separator', '.' ...
                )));
            fclose(fid);

            arr = ZarrArray(arrayPath);

            testCase.verifyEqual(arr.zarrFormat(), 2);
            testCase.verifyEqual(arr.shape(), [8, 8]);
            testCase.verifyEqual(arr.dataType(), 'uint16');
            % Unwritten chunks come back as the fill value
            testCase.verifyEqual(arr.read(), uint16(ones(8, 8) * 7));

            % And it stays readable after a write
            testData = uint16(randi(65535, [8, 8]));
            arr.write(testData);
            testCase.verifyEqual(arr.read(), testData);
        end

        function testOpenBigEndianArray(testCase)
            % Big-endian dtypes are decoded by the bytes codec, and reported
            % under their endianness-free v3 name
            arrayPath = fullfile(testCase.TempDir, 'v2_big_endian');
            mkdir(arrayPath);
            fid = fopen(fullfile(arrayPath, '.zarray'), 'w');
            fprintf(fid, '%s', jsonencode(struct( ...
                'zarr_format', 2, ...
                'shape', {{8, 8}}, ...
                'chunks', {{4, 4}}, ...
                'dtype', '>u4', ...
                'compressor', {NaN}, ...
                'fill_value', 0, ...
                'order', 'C', ...
                'filters', {NaN}, ...
                'dimension_separator', '.' ...
                )));
            fclose(fid);

            arr = ZarrArray(arrayPath);
            testCase.verifyEqual(arr.dataType(), 'uint32');

            testData = uint32(randi(1e6, [8, 8]));
            arr.write(testData);
            testCase.verifyEqual(arr.read(), testData);
        end

        % Remote Tests

        function testReadFromRemote(testCase)
            % Same voxel as ZarrMexTest.testReadFromRemote, which reads the
            % Zarr v3 copy of this dataset
            bbox = [1, 2; 3074, 3075; 3074, 3075; 514, 515];
            arr = ZarrArray(testCase.RemoteV2Path);

            testCase.verifyEqual(arr.read(bbox), uint32(988274));
        end

        function testInfoFromRemote(testCase)
            arr = ZarrArray(testCase.RemoteV2Path);
            info = arr.info();

            testCase.verifyEqual(info.zarrFormat, 2);
            testCase.verifyEqual(info.dataType, 'uint32');
            testCase.verifyEqual(info.chunkShape, [1, 32, 32, 32]);
        end

        % Error Tests

        function testShardingRejected(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_error_sharding');

            testCase.verifyError(@() ZarrArray.create(arrayPath, [128, 128, 128], 'uint16', ...
                'chunkShape', [16, 16, 16], 'shardShape', [64, 64, 64], 'zarrFormat', 2), ...
                'zarr:error');
        end

        function testCompressorSequenceRejected(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_error_seq');

            testCase.verifyError(@() ZarrArray.create(arrayPath, [64, 64], 'uint8', ...
                'chunkShape', [32, 32], 'zarrFormat', 2, 'compressors', {'zstd', 'gzip'}), ...
                'zarr:error');
        end

        function testV3OnlyCodecRejected(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_error_transpose');

            testCase.verifyError(@() ZarrArray.create(arrayPath, [64, 64], 'uint8', ...
                'chunkShape', [32, 32], 'zarrFormat', 2, 'filters', 'transpose'), ...
                'zarr:error');
        end

        function testUnknownZarrFormatRejected(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_error_format');

            % Rejected by the inputParser validator, so this raises MATLAB's own
            % identifier rather than 'zarr:error'
            testCase.verifyError(@() ZarrArray.create(arrayPath, [64, 64], 'uint8', ...
                'chunkShape', [32, 32], 'zarrFormat', 4), ?MException);
        end

        function testUnknownSeparatorRejected(testCase)
            arrayPath = fullfile(testCase.TempDir, 'v2_error_sep');

            testCase.verifyError(@() ZarrArray.create(arrayPath, [64, 64], 'uint8', ...
                'chunkShape', [32, 32], 'zarrFormat', 2, 'chunkKeyEncoding', '-'), ...
                'zarr:error');
        end
    end

    methods (Access = private)
        function arrayPath = verifyCompressorRoundTrip(testCase, name, compressor, expectedId)
            % VERIFYCOMPRESSORROUNDTRIP Create a v2 array with a compressor,
            % round-trip data through it, and check the numcodecs id on disk.
            arrayPath = fullfile(testCase.TempDir, name);
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'uint16', ...
                'chunkShape', [32, 32, 32], 'zarrFormat', 2, 'compressors', compressor);

            metadata = jsondecode(fileread(fullfile(arrayPath, '.zarray')));
            testCase.verifyEqual(metadata.compressor.id, expectedId);

            testData = uint16(randi(65535, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];
            arr.write(testData, bbox);

            testCase.verifyEqual(arr.read(bbox), testData);
        end
    end
end
