classdef ZarrPythonInteropEdgeCasesTest < matlab.unittest.TestCase
    % ZARRPYTHONINTEROPEDGECASESTEST One-off zarr-matlab <-> zarr-python
    % interop cases not covered by the core matrix in ZarrPythonInteropTest.m:
    % v2-only compressors, explicit codec configuration, separators, byte
    % order (filters/order), fill values, big-endian dtypes, and v3-specific
    % features (filters off, codec sequences, sharding).
    %
    %   Requires `uv` on PATH -- see ZarrPythonInteropTest.m / README.md.

    properties (Access = private)
        TempDir
    end

    methods (TestClassSetup)
        function addParentToPath(~)
            parentDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(parentDir);
        end

        function checkUvAvailable(~)
            ZarrInteropHelper.assertUvAvailable();
        end
    end

    methods (TestMethodSetup)
        function setupTempDir(testCase)
            testCase.TempDir = fullfile(tempdir, 'zarr_matlab_interop_test');
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
        % --- V2-only compressors (not part of the core {zstd,gzip,blosc} matrix) ---

        function testZlibCompressor(testCase)
            testCase.verifyMatlabWriteRoundTrip('zlib_mw', [10, 8], [5, 4], 'uint16', 2, ...
                'compressors', 'zlib');
            testCase.verifyPythonWriteRoundTrip('zlib_pw', [10, 8], [5, 4], 'uint16', 2, ...
                {'--compressor', 'zlib'});
        end

        function testBz2Compressor(testCase)
            testCase.verifyMatlabWriteRoundTrip('bz2_mw', [10, 8], [5, 4], 'uint16', 2, ...
                'compressors', 'bz2');
            testCase.verifyPythonWriteRoundTrip('bz2_pw', [10, 8], [5, 4], 'uint16', 2, ...
                {'--compressor', 'bz2'});
        end

        % --- Explicit codec configuration ---

        function testBloscExplicitConfig(testCase)
            % v2 numcodecs blosc takes an integer shuffle (0/1/2); v3 takes
            % the string form -- these are genuinely different configs, not
            % just a formatting difference, so each format gets its own.
            shuffleByFormat = struct('v2', 1, 'v3', 'shuffle');
            for zarrFormat = [2, 3]
                shuffle = shuffleByFormat.(sprintf('v%d', zarrFormat));
                % blocksize has no default in the V3 BloscCodecConfigurationV1
                % schema (unlike typesize, which ZarrArray.m auto-injects when
                % missing), so it must be supplied explicitly here.
                bloscConfig = struct('cname', 'zstd', 'clevel', 9, 'shuffle', shuffle, 'blocksize', 0);
                testCase.verifyMatlabWriteRoundTrip(sprintf('blosc_cfg_mw_%d', zarrFormat), ...
                    [10, 8], [5, 4], 'float32', zarrFormat, ...
                    'compressors', struct('name', 'blosc', 'configuration', bloscConfig));
                testCase.verifyPythonWriteRoundTrip(sprintf('blosc_cfg_pw_%d', zarrFormat), ...
                    [10, 8], [5, 4], 'float32', zarrFormat, ...
                    {'--compressor', 'blosc', '--blosc-cname', 'zstd', '--blosc-shuffle', 'shuffle'});
            end
        end

        % --- No compression ---

        function testNoneCompressor(testCase)
            % uint16, not uint8: a single-byte dtype's default 'bytes' codec
            % has no configuration, and zarrs re-serializes such a codec
            % using the Zarr V3 spec's bare-string shorthand ("bytes" instead
            % of {"name":"bytes"}) -- which zarr-python 3.3.0's codec parser
            % does not accept. That's a real, narrow interop gap (confirmed
            % by inspecting the on-disk zarr.json directly), but it's about
            % single-byte dtypes specifically, not about compressor
            % selection, which is what this test means to exercise -- so it
            % uses uint16 like the rest of the suite instead of routing
            % around it silently.
            for zarrFormat = [2, 3]
                testCase.verifyMatlabWriteRoundTrip(sprintf('none_mw_%d', zarrFormat), ...
                    [10, 8], [5, 4], 'uint16', zarrFormat, 'compressors', 'none');
                testCase.verifyPythonWriteRoundTrip(sprintf('none_pw_%d', zarrFormat), ...
                    [10, 8], [5, 4], 'uint16', zarrFormat, {'--compressor', 'none'});
            end
        end

        % --- Chunk key separators (v2) ---

        function testDotSeparator(testCase)
            arrayPath = fullfile(testCase.TempDir, 'sep_dot_mw');
            arr = ZarrArray.create(arrayPath, [8, 8], 'uint8', 'chunkShape', [4, 4], ...
                'zarrFormat', 2, 'chunkKeyEncoding', '.');
            arr.write(ZarrInteropHelper.deterministicData([8, 8], 'uint8'));
            testCase.verifyTrue(isfile(fullfile(arrayPath, '0.0')));
            testCase.assertCliOk(testCase.checkArrayCmd(arrayPath, [8, 8], 'uint8', 2, {}));

            pyPath = fullfile(testCase.TempDir, 'sep_dot_pw');
            testCase.assertCliOk(testCase.writeArrayCmd(pyPath, [8, 8], [4, 4], 'uint8', 2, ...
                {'--separator', 'dot'}));
            arr2 = ZarrArray(pyPath);
            testCase.verifyEqual(arr2.read(), ZarrInteropHelper.deterministicData([8, 8], 'uint8'));
        end

        function testSlashSeparator(testCase)
            arrayPath = fullfile(testCase.TempDir, 'sep_slash_mw');
            arr = ZarrArray.create(arrayPath, [8, 8], 'uint8', 'chunkShape', [4, 4], ...
                'zarrFormat', 2, 'chunkKeyEncoding', '/');
            arr.write(ZarrInteropHelper.deterministicData([8, 8], 'uint8'));
            testCase.verifyTrue(isfile(fullfile(arrayPath, '0', '0')));
            testCase.assertCliOk(testCase.checkArrayCmd(arrayPath, [8, 8], 'uint8', 2, {}));

            pyPath = fullfile(testCase.TempDir, 'sep_slash_pw');
            testCase.assertCliOk(testCase.writeArrayCmd(pyPath, [8, 8], [4, 4], 'uint8', 2, ...
                {'--separator', 'slash'}));
            arr2 = ZarrArray(pyPath);
            testCase.verifyEqual(arr2.read(), ZarrInteropHelper.deterministicData([8, 8], 'uint8'));
        end

        % --- Byte layout (v2 order) ---

        function testCOrder(testCase)
            testCase.verifyMatlabWriteRoundTrip('corder_mw', [9, 7, 5], [9, 7, 5], 'uint16', 2, ...
                'filters', 'none');
            testCase.verifyPythonWriteRoundTrip('corder_pw', [9, 7, 5], [9, 7, 5], 'uint16', 2, ...
                {'--order', 'C'});
        end

        function testFOrder(testCase)
            testCase.verifyMatlabWriteRoundTrip('forder_mw', [9, 7, 5], [9, 7, 5], 'uint16', 2);
            testCase.verifyPythonWriteRoundTrip('forder_pw', [9, 7, 5], [9, 7, 5], 'uint16', 2, ...
                {'--order', 'F'});
        end

        % --- Fill values ---

        function testCustomFillValue(testCase)
            shape = [8, 8]; chunkShape = [4, 4];
            for zarrFormat = [2, 3]
                arrayPath = fullfile(testCase.TempDir, sprintf('fill_mw_%d', zarrFormat));
                ZarrArray.create(arrayPath, shape, 'uint16', 'chunkShape', chunkShape, ...
                    'zarrFormat', zarrFormat, 'fillValue', 42);
                cmd = testCase.checkArrayCmd(arrayPath, shape, 'uint16', zarrFormat, ...
                    {'--fill-only', '--fill-value', '42'});
                testCase.assertCliOk(cmd);

                pyPath = fullfile(testCase.TempDir, sprintf('fill_pw_%d', zarrFormat));
                cmd = testCase.writeArrayCmd(pyPath, shape, chunkShape, 'uint16', zarrFormat, ...
                    {'--fill-only', '--fill-value', '42'});
                testCase.assertCliOk(cmd);
                arr = ZarrArray(pyPath);
                testCase.verifyEqual(arr.read(), uint16(ones(shape) * 42));
            end
        end

        function testNaNFillValue(testCase)
            % zarr-matlab can only WRITE a NaN fill value under Zarr v2: it
            % goes through jsonencode, which renders NaN as JSON null, and
            % v2's fillValueForV2 special-cases that into the spec's "NaN"
            % string form. v3 has no such special-casing (a pre-existing,
            % out-of-scope-here gap, not something this suite patches), so a
            % null v3 fill_value is rejected outright by zarrs at create
            % time -- confirmed directly, not assumed. The v3 half of this
            % test therefore only covers Python-write / MATLAB-read.
            shape = [8, 8]; chunkShape = [4, 4];

            arrayPath = fullfile(testCase.TempDir, 'nanfill_mw_2');
            ZarrArray.create(arrayPath, shape, 'float64', 'chunkShape', chunkShape, ...
                'zarrFormat', 2, 'fillValue', NaN);
            cmd = testCase.checkArrayCmd(arrayPath, shape, 'float64', 2, ...
                {'--fill-only', '--fill-value', 'nan'});
            testCase.assertCliOk(cmd);

            for zarrFormat = [2, 3]
                pyPath = fullfile(testCase.TempDir, sprintf('nanfill_pw_%d', zarrFormat));
                cmd = testCase.writeArrayCmd(pyPath, shape, chunkShape, 'float64', zarrFormat, ...
                    {'--fill-only', '--fill-value', 'nan'});
                testCase.assertCliOk(cmd);
                arr = ZarrArray(pyPath);
                testCase.verifyEqual(arr.read(), NaN(shape));
            end
        end

        % --- Big-endian (Zarr v2 only expresses this; zarr-matlab can read
        % it but never writes it, so this is Python-write / MATLAB-check only) ---

        function testBigEndianV2(testCase)
            shape = [6, 6]; chunkShape = [3, 3]; dataType = 'uint32';
            pyPath = fullfile(testCase.TempDir, 'bigendian_pw');
            cmd = testCase.writeArrayCmd(pyPath, shape, chunkShape, dataType, 2, ...
                {'--byteorder', 'big', '--compressor', 'none'});
            testCase.assertCliOk(cmd);

            metadata = jsondecode(fileread(fullfile(pyPath, '.zarray')));
            testCase.verifyEqual(metadata.dtype, '>u4');

            arr = ZarrArray(pyPath);
            testCase.verifyEqual(arr.dataType(), dataType);
            testCase.verifyEqual(arr.read(), ZarrInteropHelper.deterministicData(shape, dataType));
        end

        % --- V3-specific: filters off, codec sequence, sharding ---

        function testV3FiltersOff(testCase)
            testCase.verifyMatlabWriteRoundTrip('v3_nofilt_mw', [9, 7, 5], [9, 7, 5], 'uint16', 3, ...
                'filters', 'none');
            testCase.verifyPythonWriteRoundTrip('v3_nofilt_pw', [9, 7, 5], [9, 7, 5], 'uint16', 3, ...
                {'--filters', 'none'});
        end

        function testV3CodecSequence(testCase)
            % uint16, not uint8 -- see the comment in testNoneCompressor.
            shape = [10, 10]; chunkShape = [5, 5]; dataType = 'uint16';
            arrayPath = fullfile(testCase.TempDir, 'v3_seq_mw');
            arr = ZarrArray.create(arrayPath, shape, dataType, 'chunkShape', chunkShape, ...
                'zarrFormat', 3, 'compressors', {'zstd', 'gzip'});
            arr.write(ZarrInteropHelper.deterministicData(shape, dataType));
            testCase.assertCliOk(testCase.checkArrayCmd(arrayPath, shape, dataType, 3, {}));

            pyPath = fullfile(testCase.TempDir, 'v3_seq_pw');
            cmd = testCase.writeArrayCmd(pyPath, shape, chunkShape, dataType, 3, ...
                {'--compressor-sequence', 'zstd,gzip'});
            testCase.assertCliOk(cmd);
            arr2 = ZarrArray(pyPath);
            testCase.verifyEqual(arr2.read(), ZarrInteropHelper.deterministicData(shape, dataType));
        end

        function testV3Sharding(testCase)
            shape = [16, 16]; dataType = 'uint16';
            arrayPath = fullfile(testCase.TempDir, 'v3_shard_mw');
            arr = ZarrArray.create(arrayPath, shape, dataType, 'chunkShape', [4, 4], ...
                'shardShape', [8, 8], 'zarrFormat', 3, 'compressors', 'zstd');
            arr.write(ZarrInteropHelper.deterministicData(shape, dataType));
            testCase.assertCliOk(testCase.checkArrayCmd(arrayPath, shape, dataType, 3, {}));

            pyPath = fullfile(testCase.TempDir, 'v3_shard_pw');
            cmd = testCase.writeArrayCmd(pyPath, shape, [4, 4], dataType, 3, ...
                {'--shards', '8,8', '--compressor', 'zstd'});
            testCase.assertCliOk(cmd);
            arr2 = ZarrArray(pyPath);
            testCase.verifyEqual(arr2.read(), ZarrInteropHelper.deterministicData(shape, dataType));
            info = arr2.info();
            testCase.verifyEqual(info.chunkShape, [4, 4]);
            testCase.verifyEqual(info.shardShape, [8, 8]);
        end
    end

    methods (Access = private)
        function verifyMatlabWriteRoundTrip(testCase, name, shape, chunkShape, dataType, zarrFormat, varargin)
            arrayPath = fullfile(testCase.TempDir, name);
            arr = ZarrArray.create(arrayPath, shape, dataType, 'chunkShape', chunkShape, ...
                'zarrFormat', zarrFormat, varargin{:});
            arr.write(ZarrInteropHelper.deterministicData(shape, dataType));

            cmd = testCase.checkArrayCmd(arrayPath, shape, dataType, zarrFormat, {});
            testCase.assertCliOk(cmd);
        end

        function verifyPythonWriteRoundTrip(testCase, name, shape, chunkShape, dataType, zarrFormat, extraArgs)
            arrayPath = fullfile(testCase.TempDir, name);
            cmd = testCase.writeArrayCmd(arrayPath, shape, chunkShape, dataType, zarrFormat, extraArgs);
            testCase.assertCliOk(cmd);

            arr = ZarrArray(arrayPath);
            testCase.verifyEqual(arr.read(), ZarrInteropHelper.deterministicData(shape, dataType));
        end

        function cmd = writeArrayCmd(~, arrayPath, shape, chunkShape, dataType, zarrFormat, extraArgs)
            cmd = sprintf('uv run "%s" write-array --path "%s" --shape %s --chunks %s --dtype %s --zarr-format %d %s', ...
                ZarrInteropHelper.cliPath(), arrayPath, ZarrInteropHelper.shapeArg(shape), ...
                ZarrInteropHelper.shapeArg(chunkShape), dataType, zarrFormat, strjoin(extraArgs, ' '));
        end

        function cmd = checkArrayCmd(~, arrayPath, shape, dataType, zarrFormat, extraArgs)
            cmd = sprintf('uv run "%s" check-array --path "%s" --shape %s --dtype %s --zarr-format %d %s', ...
                ZarrInteropHelper.cliPath(), arrayPath, ZarrInteropHelper.shapeArg(shape), ...
                dataType, zarrFormat, strjoin(extraArgs, ' '));
        end

        function assertCliOk(testCase, cmd)
            [status, out] = system(cmd);
            testCase.assertEqual(status, 0, sprintf('interop CLI call failed:\n%s\ncommand: %s', out, cmd));
        end
    end
end
