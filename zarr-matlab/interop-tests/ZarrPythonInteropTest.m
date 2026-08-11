classdef ZarrPythonInteropTest < matlab.unittest.TestCase
    % ZARRPYTHONINTEROPTEST Round-trip zarr-matlab against a real zarr-python
    % install, in both directions, for both Zarr v2 and v3.
    %
    %   This suite requires `uv` (https://docs.astral.sh/uv/) on PATH. It is
    %   NOT part of `runtests('tests')` -- run it explicitly via
    %   `runtests('interop-tests')`. See README.md in this directory.
    %
    %   The core compressor/format matrix lives here as a parameterized test;
    %   one-off cases (separators, order, fill values, big-endian, sharding,
    %   ...) are in ZarrPythonInteropEdgeCasesTest.m, and group hierarchy
    %   coverage is in ZarrPythonInteropGroupTest.m. All three share the
    %   deterministic-data formula and fixed attributes via ZarrInteropHelper.

    properties (TestParameter)
        ZarrFormat = struct('v2', 2, 'v3', 3);
        Compressor = struct('zstd', 'zstd', 'gzip', 'gzip', 'blosc', 'blosc');
    end

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
        function testMatlabWritePythonChecks(testCase, ZarrFormat, Compressor)
            shape = [12, 9, 5];
            chunkShape = [4, 3, 5];
            dataType = 'uint16';
            arrayPath = fullfile(testCase.TempDir, sprintf('mw_%d_%s', ZarrFormat, Compressor));

            arr = ZarrArray.create(arrayPath, shape, dataType, 'chunkShape', chunkShape, ...
                'zarrFormat', ZarrFormat, 'compressors', Compressor);
            arr.write(ZarrInteropHelper.deterministicData(shape, dataType));
            arr.setAttributes(ZarrInteropHelper.defaultAttrs());

            cmd = sprintf(['uv run "%s" check-array --path "%s" --shape %s --dtype %s ' ...
                '--zarr-format %d --attrs'], ...
                ZarrInteropHelper.cliPath(), arrayPath, ZarrInteropHelper.shapeArg(shape), ...
                dataType, ZarrFormat);
            [status, out] = system(cmd);
            testCase.assertEqual(status, 0, sprintf('python check-array failed:\n%s', out));
        end

        function testPythonWritesMatlabChecks(testCase, ZarrFormat, Compressor)
            shape = [12, 9, 5];
            chunkShape = [4, 3, 5];
            dataType = 'uint16';
            arrayPath = fullfile(testCase.TempDir, sprintf('pw_%d_%s', ZarrFormat, Compressor));

            cmd = sprintf(['uv run "%s" write-array --path "%s" --shape %s --chunks %s ' ...
                '--dtype %s --zarr-format %d --compressor %s --attrs'], ...
                ZarrInteropHelper.cliPath(), arrayPath, ZarrInteropHelper.shapeArg(shape), ...
                ZarrInteropHelper.shapeArg(chunkShape), dataType, ZarrFormat, Compressor);
            [status, out] = system(cmd);
            testCase.assertEqual(status, 0, sprintf('python write-array failed:\n%s', out));

            arr = ZarrArray(arrayPath);
            testCase.verifyEqual(arr.zarrFormat(), ZarrFormat);
            testCase.verifyEqual(arr.dataType(), dataType);
            testCase.verifyEqual(arr.read(), ZarrInteropHelper.deterministicData(shape, dataType));
            testCase.verifyEqual(arr.getAttributes(), ZarrInteropHelper.defaultAttrs());
        end
    end
end
