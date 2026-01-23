classdef ZarrMexTest < matlab.unittest.TestCase

    properties (Constant)
        RemotePath = 'https://static.webknossos.org/data/zarr_v3/l4_sample/segmentation/1';
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

            testCase.verifyEqual(info.shape, [100, 100, 100]);
            testCase.verifyEqual(info.dataType, 'uint32');
            testCase.verifyEqual(info.chunkShape, [32, 32, 32]);
            testCase.verifyEqual(info.shardShape, [32, 32, 32]);
        end

        function testInfoFromRemote(testCase)
            info = zarrMex('info', testCase.RemotePath);

            testCase.verifyEqual(info.dataType, 'uint32');
        end

        % Error Tests

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
