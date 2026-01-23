classdef ZarrArrayTest < matlab.unittest.TestCase

    properties (Constant)
        TempDir = '/tmp/zarr_matlab_test';
    end

    methods (TestClassSetup)
        function addParentToPath(~)
            parentDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(parentDir);
        end
    end

    methods (TestMethodSetup)
        function cleanupTempDir(testCase)
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
            arrayPath = fullfile(testCase.TempDir, 'class_create');

            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', 'chunkShape', [32, 32, 32]);

            testCase.verifyTrue(isfile(fullfile(arrayPath, 'zarr.json')));

            info = arr.info();
            testCase.verifyEqual(info.shape, [100, 100, 100]);
            testCase.verifyEqual(info.dataType, 'uint16');
            testCase.verifyEqual(info.chunkShape, [32, 32, 32]);
        end

        function testCreateWithDefaultChunkShape(testCase)
            arrayPath = fullfile(testCase.TempDir, 'default_chunk');

            % Create without specifying chunkShape - should default to min(shape, 100)
            arr = ZarrArray.create(arrayPath, [200, 50, 150], 'uint8');

            info = arr.info();
            testCase.verifyEqual(info.chunkShape, [100, 50, 100]);
        end

        function testCreateFromDataWithDefaultChunkShape(testCase)
            arrayPath = fullfile(testCase.TempDir, 'from_data_default_chunk');

            data = uint32(randi(1000, [80, 120, 60]));
            arr = ZarrArray.createFromData(arrayPath, data);

            info = arr.info();
            testCase.verifyEqual(info.chunkShape, [80, 100, 60]);
            testCase.verifyEqual(arr.shape(), [80, 120, 60]);
        end

        function testShape(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_shape');
            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', 'chunkShape', [32, 32, 32]);

            testCase.verifyEqual(arr.shape(), [100, 100, 100]);
        end

        function testResize(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_resize');
            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', 'chunkShape', [32, 32, 32]);

            arr.resize([150, 120, 100]);

            testCase.verifyEqual(arr.shape(), [150, 120, 100]);
        end

        function testWriteRead(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_write_read');
            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', 'chunkShape', [32, 32, 32]);

            testData = uint16(reshape(1:1000, [10, 10, 10]));
            bbox = [1, 11; 1, 11; 1, 11];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        function testOpenExisting(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_open');
            ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', 'chunkShape', [32, 32, 32]);

            arr = ZarrArray(arrayPath);

            testCase.verifyEqual(arr.shape(), [100, 100, 100]);
        end

        function testWithSharding(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_sharding');

            arr = ZarrArray.create(arrayPath, [128, 128, 128], 'float32', ...
                'chunkShape', [32, 32, 32], 'shardShape', [64, 64, 64]);

            info = arr.info();
            testCase.verifyEqual(info.dataType, 'float32');
            testCase.verifyEqual(info.chunkShape, [32, 32, 32]);
            testCase.verifyEqual(info.shardShape, [64, 64, 64]);
        end

        function testWithZstdCodec(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_zstd');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'uint8', ...
                'chunkShape', [32, 32, 32], 'codec', 'zstd');

            testData = uint8(randi(255, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        function testWithZstdCodecConfigured(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_zstd_cfg');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'int32', ...
                'chunkShape', [32, 32, 32], 'codec', struct('name', 'zstd', 'configuration', struct('level', 10)));

            testData = int32(randi(1000000, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        function testWithGzipCodec(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_gzip');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'float64', ...
                'chunkShape', [32, 32, 32], 'codec', struct('name', 'gzip', 'configuration', struct('level', 6)));

            testData = rand(32, 32, 32);
            bbox = [1, 33; 1, 33; 1, 33];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyLessThan(max(abs(testData(:) - readData(:))), 1e-10);
        end

        function testWithShardingAndZstd(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_shard_zstd');
            arr = ZarrArray.create(arrayPath, [128, 128, 128], 'uint16', ...
                'chunkShape', [16, 16, 16], 'shardShape', [64, 64, 64], 'codec', 'zstd');

            testData = uint16(randi(65535, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        function testCreateFromData(testCase)
            arrayPath = fullfile(testCase.TempDir, 'from_data');
            testData = uint16(randi(65535, [50, 40, 30]));

            arr = ZarrArray.createFromData(arrayPath, testData, 'chunkShape', [16, 16, 16]);

            testCase.verifyEqual(arr.shape(), [50, 40, 30]);
            info = arr.info();
            testCase.verifyEqual(info.dataType, 'uint16');

            bbox = [1, 51; 1, 41; 1, 31];
            readData = arr.read(bbox);
            testCase.verifyEqual(readData, testData);
        end

        function testCreateFromDataWithCodec(testCase)
            arrayPath = fullfile(testCase.TempDir, 'from_data_zstd');
            testData = single(rand(32, 32, 32));

            arr = ZarrArray.createFromData(arrayPath, testData, 'chunkShape', [16, 16, 16], 'codec', 'zstd');

            info = arr.info();
            testCase.verifyEqual(info.dataType, 'float32');

            bbox = [1, 33; 1, 33; 1, 33];
            readData = arr.read(bbox);
            testCase.verifyLessThan(max(abs(testData(:) - readData(:))), 1e-6);
        end

        function testCreateFromDataDouble(testCase)
            arrayPath = fullfile(testCase.TempDir, 'from_data_double');
            testData = rand(20, 20, 20);  % double by default

            arr = ZarrArray.createFromData(arrayPath, testData, 'chunkShape', [10, 10, 10]);

            info = arr.info();
            testCase.verifyEqual(info.dataType, 'float64');

            bbox = [1, 21; 1, 21; 1, 21];
            readData = arr.read(bbox);
            testCase.verifyLessThan(max(abs(testData(:) - readData(:))), 1e-10);
        end

        % Attribute Tests

        function testGetAttributesEmpty(testCase)
            arrayPath = fullfile(testCase.TempDir, 'array_attrs_empty');
            arr = ZarrArray.create(arrayPath, [32, 32, 32], 'uint8', 'chunkShape', [16, 16, 16]);

            attrs = arr.getAttributes();

            testCase.verifyTrue(isstruct(attrs));
            testCase.verifyEmpty(fieldnames(attrs));
        end

        function testSetAndGetAttributes(testCase)
            arrayPath = fullfile(testCase.TempDir, 'array_attrs');
            arr = ZarrArray.create(arrayPath, [32, 32, 32], 'uint16', 'chunkShape', [16, 16, 16]);

            attrs = struct('description', 'test array', 'scale', 2.5, 'offset', [10, 20, 30]);
            arr.setAttributes(attrs);

            readAttrs = arr.getAttributes();

            testCase.verifyEqual(readAttrs.description, 'test array');
            testCase.verifyEqual(readAttrs.scale, 2.5);
            testCase.verifyEqual(readAttrs.offset', [10, 20, 30]);
        end

        function testSetAndGetSingleAttribute(testCase)
            arrayPath = fullfile(testCase.TempDir, 'array_single_attr');
            arr = ZarrArray.create(arrayPath, [32, 32, 32], 'float32', 'chunkShape', [16, 16, 16]);

            arr.setAttribute('voxel_size', [1.0, 1.0, 2.0]);
            arr.setAttribute('unit', 'um');

            testCase.verifyEqual(arr.getAttribute('voxel_size')', [1.0, 1.0, 2.0]);
            testCase.verifyEqual(arr.getAttribute('unit'), 'um');
        end

        function testGetNonExistentAttribute(testCase)
            arrayPath = fullfile(testCase.TempDir, 'array_no_attr');
            arr = ZarrArray.create(arrayPath, [32, 32, 32], 'uint8', 'chunkShape', [16, 16, 16]);

            testCase.verifyError(@() arr.getAttribute('missing'), 'zarr:error');
        end

        function testAttributesPersist(testCase)
            arrayPath = fullfile(testCase.TempDir, 'array_attrs_persist');
            arr = ZarrArray.create(arrayPath, [32, 32, 32], 'int16', 'chunkShape', [16, 16, 16]);
            arr.setAttribute('key', 'value');

            % Reopen the array
            arr2 = ZarrArray(arrayPath);

            testCase.verifyEqual(arr2.getAttribute('key'), 'value');
        end

        % Error Tests

        function testWriteWithWrongDataType(testCase)
            arrayPath = fullfile(testCase.TempDir, 'error_wrong_dtype');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'uint16', 'chunkShape', [32, 32, 32]);

            % Try to write uint8 data to a uint16 array
            wrongData = uint8(randi(255, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];

            testCase.verifyError(@() arr.write(bbox, wrongData), 'zarr:error');
        end

        function testWriteOutOfBounds(testCase)
            arrayPath = fullfile(testCase.TempDir, 'error_out_of_bounds');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'uint32', 'chunkShape', [32, 32, 32]);

            data = uint32(ones(32, 32, 32));
            % Bounding box extends beyond array shape
            bbox = [50, 82; 1, 33; 1, 33];

            testCase.verifyError(@() arr.write(bbox, data), 'zarr:error');
        end
    end
end
