classdef ZarrGroupTest < matlab.unittest.TestCase

    properties (Constant)
        RemoteGroupPath = 'https://static.webknossos.org/data/zarr_v3/l4_sample';
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
            groupPath = fullfile(testCase.TempDir, 'group_create');

            grp = ZarrGroup.create(groupPath);

            testCase.verifyTrue(isfile(fullfile(groupPath, 'zarr.json')));

            metadata = jsondecode(fileread(fullfile(groupPath, 'zarr.json')));
            testCase.verifyEqual(metadata.zarr_format, 3);
            testCase.verifyEqual(metadata.node_type, 'group');
        end

        function testOpen(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_open');
            ZarrGroup.create(groupPath);

            grp = ZarrGroup(groupPath);

            testCase.verifyEqual(grp.path, groupPath);
        end

        function testCreateSubgroup(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_parent');
            grp = ZarrGroup.create(groupPath);

            subgrp = grp.createGroup('subgroup');

            testCase.verifyTrue(isfile(fullfile(groupPath, 'subgroup', 'zarr.json')));
            testCase.verifyEqual(subgrp.path, fullfile(groupPath, 'subgroup'));
        end

        function testOpenSubgroup(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_open_sub');
            grp = ZarrGroup.create(groupPath);
            grp.createGroup('child');

            subgrp = grp.openGroup('child');

            testCase.verifyEqual(subgrp.path, fullfile(groupPath, 'child'));
        end

        function testCreateArray(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_with_array');
            grp = ZarrGroup.create(groupPath);

            arr = grp.createArray('data', [64, 64, 64], 'uint16', [32, 32, 32]);

            testCase.verifyTrue(isfile(fullfile(groupPath, 'data', 'zarr.json')));
            testCase.verifyEqual(arr.shape(), [64, 64, 64]);
        end

        function testOpenArray(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_open_array');
            grp = ZarrGroup.create(groupPath);
            grp.createArray('myarray', [32, 32, 32], 'float32', [16, 16, 16]);

            arr = grp.openArray('myarray');

            testCase.verifyEqual(arr.shape(), [32, 32, 32]);
        end

        function testList(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_list');
            grp = ZarrGroup.create(groupPath);
            grp.createGroup('subgroup1');
            grp.createGroup('subgroup2');
            grp.createArray('array1', [10, 10], 'uint8', [10, 10]);

            names = grp.list();

            testCase.verifyEqual(sort(names), {'array1', 'subgroup1', 'subgroup2'});
        end

        function testListContents(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_list_contents');
            grp = ZarrGroup.create(groupPath);
            grp.createGroup('group_a');
            grp.createGroup('group_b');
            grp.createArray('array_x', [10, 10], 'uint8', [10, 10]);
            grp.createArray('array_y', [20, 20], 'uint16', [10, 10]);

            [groups, arrays] = grp.listContents();

            testCase.verifyEqual(sort(groups), {'group_a', 'group_b'});
            testCase.verifyEqual(sort(arrays), {'array_x', 'array_y'});
        end

        function testNestedStructure(testCase)
            rootPath = fullfile(testCase.TempDir, 'nested_root');
            root = ZarrGroup.create(rootPath);

            level1 = root.createGroup('level1');
            level2 = level1.createGroup('level2');
            arr = level2.createArray('deep_array', [8, 8, 8], 'int32', [8, 8, 8]);

            % Write and read through the nested structure
            testData = int32(reshape(1:512, [8, 8, 8]));
            bbox = [1, 9; 1, 9; 1, 9];
            arr.write(bbox, testData);

            % Open through hierarchy and verify
            reopened = ZarrGroup(rootPath);
            l1 = reopened.openGroup('level1');
            l2 = l1.openGroup('level2');
            readArr = l2.openArray('deep_array');
            readData = readArr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        % Error Tests

        function testOpenNonExistent(testCase)
            nonExistentPath = fullfile(testCase.TempDir, 'group_not_exist');

            testCase.verifyError(@() ZarrGroup(nonExistentPath), 'zarr:error');
        end

        function testOpenArrayAsGroup(testCase)
            arrayPath = fullfile(testCase.TempDir, 'array_not_group');
            ZarrArray.create(arrayPath, [10, 10], 'uint8', [10, 10]);

            testCase.verifyError(@() ZarrGroup(arrayPath), 'zarr:error');
        end

        function testCreateExisting(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_exists');
            ZarrGroup.create(groupPath);

            testCase.verifyError(@() ZarrGroup.create(groupPath), 'zarr:error');
        end

        % Remote Tests

        function testOpenRemote(testCase)
            grp = ZarrGroup(testCase.RemoteGroupPath);

            testCase.verifyEqual(grp.path, testCase.RemoteGroupPath);
        end

        function testOpenRemoteSubgroup(testCase)
            grp = ZarrGroup(testCase.RemoteGroupPath);

            subgrp = grp.openGroup('segmentation');

            testCase.verifyTrue(endsWith(subgrp.path, 'segmentation'));
        end

        function testOpenRemoteArray(testCase)
            grp = ZarrGroup(testCase.RemoteGroupPath);

            subgrp = grp.openGroup('segmentation');
            arr = subgrp.openArray('1');

            info = arr.info();
            testCase.verifyEqual(info.dataType, 'uint32');
        end

        function testRemoteListError(testCase)
            grp = ZarrGroup(testCase.RemoteGroupPath);

            testCase.verifyError(@() grp.list(), 'zarr:error');
        end

        function testRemoteCreateError(testCase)
            grp = ZarrGroup(testCase.RemoteGroupPath);

            testCase.verifyError(@() grp.createGroup('newgroup'), 'zarr:error');
            testCase.verifyError(@() grp.createArray('newarray', [10, 10], 'uint8', [10, 10]), 'zarr:error');
        end
    end
end
