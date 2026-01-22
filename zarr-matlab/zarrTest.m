classdef ZarrTest < matlab.unittest.TestCase

    properties (Constant)
        RemotePath = 'https://static.webknossos.org/data/zarr_v3/l4_sample/segmentation/1';
        TempDir = '/tmp/zarr_matlab_test';
    end

    methods (TestClassSetup)
        function buildMex(~)
            zarrBuild();
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
        % zarrMex Tests

        function testReadFromRemote(testCase)
            bbox = [1, 2; 3074, 3075; 3074, 3075; 514, 515];
            data = zarrMex('read', testCase.RemotePath, bbox);
            testCase.verifyEqual(data, uint32(988274));
        end

        function testCreateArray(testCase)
            arrayPath = fullfile(testCase.TempDir, 'create_test');
            json = testCase.createTestMetadataJson([100, 100, 100], 'uint32', [32, 32, 32]);

            zarrMex('create', arrayPath, json);

            testCase.verifyTrue(isfile(fullfile(arrayPath, 'zarr.json')));

            written_json = fileread(fullfile(arrayPath, 'zarr.json'));
            original_struct = jsondecode(json);
            written_struct = jsondecode(written_json);
            testCase.verifyEqual(written_struct, original_struct);
        end

        function testWriteAndRead(testCase)
            arrayPath = fullfile(testCase.TempDir, 'write_test');
            json = testCase.createTestMetadataJson([100, 100, 100], 'uint32', [32, 32, 32]);
            zarrMex('create', arrayPath, json);

            data = uint32(reshape(1:1000, [10, 10, 10]));
            bbox = [1, 11; 1, 11; 1, 11];

            zarrMex('write', arrayPath, bbox, data);
            readData = zarrMex('read', arrayPath, bbox);

            testCase.verifyEqual(readData, data);
        end

        function testResize(testCase)
            arrayPath = fullfile(testCase.TempDir, 'resize_test');
            json = testCase.createTestMetadataJson([100, 100, 100], 'uint32', [32, 32, 32]);
            zarrMex('create', arrayPath, json);

            newShape = [200, 150, 100];
            zarrMex('resize', arrayPath, newShape);

            resizedJson = fileread(fullfile(arrayPath, 'zarr.json'));
            resizedStruct = jsondecode(resizedJson);
            testCase.verifyEqual(resizedStruct.shape, newShape');
        end

        function testInfo(testCase)
            arrayPath = fullfile(testCase.TempDir, 'info_test');
            json = testCase.createTestMetadataJson([100, 100, 100], 'uint32', [32, 32, 32]);
            zarrMex('create', arrayPath, json);

            info = zarrMex('info', arrayPath);

            testCase.verifyEqual(info.boundingBox, [1, 101; 1, 101; 1, 101]);
            testCase.verifyEqual(info.dataType, 'uint32');
            testCase.verifyEqual(info.chunkShape, [32, 32, 32]);
            testCase.verifyEqual(info.shardShape, [32, 32, 32]);
        end

        function testInfoFromRemote(testCase)
            info = zarrMex('info', testCase.RemotePath);

            testCase.verifyEqual(info.dataType, 'uint32');
        end

        % ZarrArray Class Tests

        function testZarrArrayCreate(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_create');

            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', [32, 32, 32]);

            testCase.verifyTrue(isfile(fullfile(arrayPath, 'zarr.json')));

            info = arr.info();
            testCase.verifyEqual(info.boundingBox, [1, 101; 1, 101; 1, 101]);
            testCase.verifyEqual(info.dataType, 'uint16');
            testCase.verifyEqual(info.chunkShape, [32, 32, 32]);
        end

        function testZarrArrayShape(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_shape');
            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', [32, 32, 32]);

            testCase.verifyEqual(arr.shape(), [100, 100, 100]);
        end

        function testZarrArrayResize(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_resize');
            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', [32, 32, 32]);

            arr.resize([150, 120, 100]);

            testCase.verifyEqual(arr.shape(), [150, 120, 100]);
        end

        function testZarrArrayWriteRead(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_write_read');
            arr = ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', [32, 32, 32]);

            testData = uint16(reshape(1:1000, [10, 10, 10]));
            bbox = [1, 11; 1, 11; 1, 11];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        function testZarrArrayOpenExisting(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_open');
            ZarrArray.create(arrayPath, [100, 100, 100], 'uint16', [32, 32, 32]);

            arr = ZarrArray(arrayPath);

            testCase.verifyEqual(arr.shape(), [100, 100, 100]);
        end

        function testZarrArrayWithSharding(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_sharding');

            arr = ZarrArray.create(arrayPath, [128, 128, 128], 'float32', ...
                [32, 32, 32], 'shardShape', [64, 64, 64]);

            info = arr.info();
            testCase.verifyEqual(info.dataType, 'float32');
            testCase.verifyEqual(info.chunkShape, [32, 32, 32]);
            testCase.verifyEqual(info.shardShape, [64, 64, 64]);
        end

        function testZarrArrayWithZstdCodec(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_zstd');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'uint8', [32, 32, 32], ...
                'codec', 'zstd');

            testData = uint8(randi(255, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        function testZarrArrayWithZstdCodecConfigured(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_zstd_cfg');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'int32', [32, 32, 32], ...
                'codec', struct('name', 'zstd', 'configuration', struct('level', 10)));

            testData = int32(randi(1000000, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        function testZarrArrayWithGzipCodec(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_gzip');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'float64', [32, 32, 32], ...
                'codec', struct('name', 'gzip', 'configuration', struct('level', 6)));

            testData = rand(32, 32, 32);
            bbox = [1, 33; 1, 33; 1, 33];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyLessThan(max(abs(testData(:) - readData(:))), 1e-10);
        end

        function testZarrArrayWithShardingAndZstd(testCase)
            arrayPath = fullfile(testCase.TempDir, 'class_shard_zstd');
            arr = ZarrArray.create(arrayPath, [128, 128, 128], 'uint16', ...
                [16, 16, 16], 'shardShape', [64, 64, 64], 'codec', 'zstd');

            testData = uint16(randi(65535, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];

            arr.write(bbox, testData);
            readData = arr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        % Error Tests

        function testWriteWithWrongDataType(testCase)
            arrayPath = fullfile(testCase.TempDir, 'error_wrong_dtype');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'uint16', [32, 32, 32]);

            % Try to write uint8 data to a uint16 array
            wrongData = uint8(randi(255, [32, 32, 32]));
            bbox = [1, 33; 1, 33; 1, 33];

            testCase.verifyError(@() arr.write(bbox, wrongData), 'zarr:error');
        end

        function testReadFromNonExistentPath(testCase)
            nonExistentPath = fullfile(testCase.TempDir, 'does_not_exist');

            testCase.verifyError(@() zarrMex('read', nonExistentPath, [1, 2; 1, 2; 1, 2]), 'zarr:error');
        end

        function testCreateWithMismatchedDimensions(testCase)
            arrayPath = fullfile(testCase.TempDir, 'error_dim_mismatch');

            % Shape has 3 dimensions but chunk shape has 2
            json = jsonencode(struct( ...
                'zarr_format', 3, ...
                'node_type', 'array', ...
                'shape', [100, 100, 100], ...
                'data_type', 'uint32', ...
                'chunk_grid', struct( ...
                    'name', 'regular', ...
                    'configuration', struct('chunk_shape', [32, 32]) ...
                ), ...
                'chunk_key_encoding', struct( ...
                    'name', 'default', ...
                    'configuration', struct('separator', '/') ...
                ), ...
                'fill_value', 0, ...
                'codecs', {{ struct('name', 'bytes', 'configuration', struct('endian', 'little')) }} ...
            ));

            testCase.verifyError(@() zarrMex('create', arrayPath, json), 'zarr:error');
        end

        function testInfoFromNonExistentPath(testCase)
            nonExistentPath = fullfile(testCase.TempDir, 'info_does_not_exist');

            testCase.verifyError(@() zarrMex('info', nonExistentPath), 'zarr:error');
        end

        function testWriteOutOfBounds(testCase)
            arrayPath = fullfile(testCase.TempDir, 'error_out_of_bounds');
            arr = ZarrArray.create(arrayPath, [64, 64, 64], 'uint32', [32, 32, 32]);

            data = uint32(ones(32, 32, 32));
            % Bounding box extends beyond array shape
            bbox = [50, 82; 1, 33; 1, 33];

            testCase.verifyError(@() arr.write(bbox, data), 'zarr:error');
        end
    end

    methods (Static, Access = private)
        function json = createTestMetadataJson(shape, dataType, chunkShape)
            json = jsonencode(struct( ...
                'zarr_format', 3, ...
                'node_type', 'array', ...
                'shape', shape, ...
                'data_type', dataType, ...
                'chunk_grid', struct( ...
                    'name', 'regular', ...
                    'configuration', struct('chunk_shape', chunkShape) ...
                ), ...
                'chunk_key_encoding', struct( ...
                    'name', 'default', ...
                    'configuration', struct('separator', '/') ...
                ), ...
                'fill_value', 0, ...
                'codecs', {{ struct( ...
                    'name', 'sharding_indexed', ...
                    'configuration', struct( ...
                        'chunk_shape', chunkShape, ...
                        'codecs', {{ struct('name', 'bytes', 'configuration', struct('endian', 'little')) }}, ...
                        'index_location', 'end', ...
                        'index_codecs', {{ ...
                            struct('name', 'bytes', 'configuration', struct('endian', 'little')), ...
                            struct('name', 'crc32c') ...
                        }} ...
                    )) ...
                }} ...
            ));
        end
    end
end
