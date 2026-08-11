classdef ZarrPythonInteropGroupTest < matlab.unittest.TestCase
    % ZARRPYTHONINTEROPGROUPTEST Round-trip a small ZarrGroup hierarchy
    % (root group + subgroup + one array, attributes at both group levels)
    % against zarr-python, in both directions, for both Zarr v2 and v3.
    %
    %   Requires `uv` on PATH -- see ZarrPythonInteropTest.m / README.md.

    properties (TestParameter)
        ZarrFormat = struct('v2', 2, 'v3', 3);
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
        function testMatlabWritePythonChecks(testCase, ZarrFormat)
            shape = [10, 8];
            chunkShape = [5, 4];
            dataType = 'uint16';
            groupPath = fullfile(testCase.TempDir, sprintf('mw_grp_%d', ZarrFormat));

            root = ZarrGroup.create(groupPath, 'zarrFormat', ZarrFormat);
            root.setAttributes(ZarrInteropHelper.defaultAttrs());
            child = root.createGroup('child');
            child.setAttributes(ZarrInteropHelper.defaultAttrs());
            arr = child.createArray('data', shape, dataType, 'chunkShape', chunkShape);
            arr.write(ZarrInteropHelper.deterministicData(shape, dataType));

            cmd = sprintf(['uv run "%s" check-group --path "%s" --array-shape %s --array-dtype %s ' ...
                '--zarr-format %d'], ...
                ZarrInteropHelper.cliPath(), groupPath, ZarrInteropHelper.shapeArg(shape), ...
                dataType, ZarrFormat);
            [status, out] = system(cmd);
            testCase.assertEqual(status, 0, sprintf('python check-group failed:\n%s', out));
        end

        function testPythonWritesMatlabChecks(testCase, ZarrFormat)
            shape = [10, 8];
            chunkShape = [5, 4];
            dataType = 'uint16';
            groupPath = fullfile(testCase.TempDir, sprintf('pw_grp_%d', ZarrFormat));

            cmd = sprintf(['uv run "%s" write-group --path "%s" --array-shape %s --array-chunks %s ' ...
                '--array-dtype %s --zarr-format %d --compressor zstd'], ...
                ZarrInteropHelper.cliPath(), groupPath, ZarrInteropHelper.shapeArg(shape), ...
                ZarrInteropHelper.shapeArg(chunkShape), dataType, ZarrFormat);
            [status, out] = system(cmd);
            testCase.assertEqual(status, 0, sprintf('python write-group failed:\n%s', out));

            root = ZarrGroup(groupPath);
            testCase.verifyEqual(root.zarrFormat(), ZarrFormat);
            testCase.verifyEqual(root.getAttributes(), ZarrInteropHelper.defaultAttrs());

            [groups, arrays] = root.listContents();
            testCase.verifyEqual(groups, {'child'});
            testCase.verifyEmpty(arrays);

            child = root.openGroup('child');
            testCase.verifyEqual(child.getAttributes(), ZarrInteropHelper.defaultAttrs());

            arr = child.openArray('data');
            testCase.verifyEqual(arr.read(), ZarrInteropHelper.deterministicData(shape, dataType));
        end
    end
end
