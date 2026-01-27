classdef ZarrGroupTest < matlab.unittest.TestCase

    properties (Constant)
        RemoteGroupPath = 'https://static.webknossos.org/data/zarr_v3/l4_sample';
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

            arr = grp.createArray('data', [64, 64, 64], 'uint16', 'chunkShape', [32, 32, 32]);

            testCase.verifyTrue(isfile(fullfile(groupPath, 'data', 'zarr.json')));
            testCase.verifyEqual(arr.shape(), [64, 64, 64]);
        end

        function testOpenArray(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_open_array');
            grp = ZarrGroup.create(groupPath);
            grp.createArray('myarray', [32, 32, 32], 'float32', 'chunkShape', [16, 16, 16]);

            arr = grp.openArray('myarray');

            testCase.verifyEqual(arr.shape(), [32, 32, 32]);
        end

        function testOpenAndReadWriteArray(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_open_readwrite_array');
            grp = ZarrGroup.create(groupPath);
            grp.createArray('myarray', [32, 32, 32], 'float32', 'chunkShape', [16, 16, 16]);

            arr = grp.openArray('myarray');
            testData = single(rand([32, 32, 32]));
            arr.write(testData);
            readData = arr.read();

            testCase.verifyEqual(readData, testData);
        end

        function testCreateArrayFromData(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_from_data');
            grp = ZarrGroup.create(groupPath);

            testData = uint32(randi(1000000, [40, 30, 20]));
            arr = grp.createArrayFromData('mydata', testData, 'chunkShape', [16, 16, 16], 'compressors', 'zstd');

            testCase.verifyTrue(isfile(fullfile(groupPath, 'mydata', 'zarr.json')));
            testCase.verifyEqual(arr.shape(), [40, 30, 20]);

            info = arr.info();
            testCase.verifyEqual(info.dataType, 'uint32');

            bbox = [1, 41; 1, 31; 1, 21];
            readData = arr.read(bbox);
            testCase.verifyEqual(readData, testData);
        end

        function testList(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_list');
            grp = ZarrGroup.create(groupPath);
            grp.createGroup('subgroup1');
            grp.createGroup('subgroup2');
            grp.createArray('array1', [10, 10], 'uint8', 'chunkShape', [10, 10]);

            names = grp.list();

            testCase.verifyEqual(sort(names), {'array1', 'subgroup1', 'subgroup2'});
        end

        function testListContents(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_list_contents');
            grp = ZarrGroup.create(groupPath);
            grp.createGroup('group_a');
            grp.createGroup('group_b');
            grp.createArray('array_x', [10, 10], 'uint8', 'chunkShape', [10, 10]);
            grp.createArray('array_y', [20, 20], 'uint16', 'chunkShape', [10, 10]);

            [groups, arrays] = grp.listContents();

            testCase.verifyEqual(sort(groups), {'group_a', 'group_b'});
            testCase.verifyEqual(sort(arrays), {'array_x', 'array_y'});
        end

        function testNestedStructure(testCase)
            rootPath = fullfile(testCase.TempDir, 'nested_root');
            root = ZarrGroup.create(rootPath);

            level1 = root.createGroup('level1');
            level2 = level1.createGroup('level2');
            arr = level2.createArray('deep_array', [8, 8, 8], 'int32', 'chunkShape', [8, 8, 8]);

            % Write and read through the nested structure
            testData = int32(reshape(1:512, [8, 8, 8]));
            bbox = [1, 9; 1, 9; 1, 9];
            arr.write(testData, bbox);

            % Open through hierarchy and verify
            reopened = ZarrGroup(rootPath);
            l1 = reopened.openGroup('level1');
            l2 = l1.openGroup('level2');
            readArr = l2.openArray('deep_array');
            readData = readArr.read(bbox);

            testCase.verifyEqual(readData, testData);
        end

        % Attribute Tests

        function testGetAttributesEmpty(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_attrs_empty');
            grp = ZarrGroup.create(groupPath);

            attrs = grp.getAttributes();

            testCase.verifyTrue(isstruct(attrs));
            testCase.verifyEmpty(fieldnames(attrs));
        end

        function testSetAndGetAttributes(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_attrs');
            grp = ZarrGroup.create(groupPath);

            attrs = struct('name', 'test_dataset', 'version', 2, 'scales', [1.0, 1.0, 2.0]);
            grp.setAttributes(attrs);

            readAttrs = grp.getAttributes();

            testCase.verifyEqual(readAttrs.name, 'test_dataset');
            testCase.verifyEqual(readAttrs.version, 2);
            testCase.verifyEqual(readAttrs.scales', [1.0, 1.0, 2.0]);
        end

        function testSetAndGetSingleAttribute(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_single_attr');
            grp = ZarrGroup.create(groupPath);

            grp.setAttribute('resolution', [4, 4, 30]);
            grp.setAttribute('unit', 'nm');

            testCase.verifyEqual(grp.getAttribute('resolution')', [4, 4, 30]);
            testCase.verifyEqual(grp.getAttribute('unit'), 'nm');
        end

        function testGetNonExistentAttribute(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_no_attr');
            grp = ZarrGroup.create(groupPath);

            testCase.verifyError(@() grp.getAttribute('missing'), 'zarr:error');
        end

        function testAttributesPersist(testCase)
            groupPath = fullfile(testCase.TempDir, 'group_attrs_persist');
            grp = ZarrGroup.create(groupPath);
            grp.setAttribute('key', 'value');

            % Reopen the group
            grp2 = ZarrGroup(groupPath);

            testCase.verifyEqual(grp2.getAttribute('key'), 'value');
        end

        % Error Tests

        function testOpenNonExistent(testCase)
            nonExistentPath = fullfile(testCase.TempDir, 'group_not_exist');

            testCase.verifyError(@() ZarrGroup(nonExistentPath), 'zarr:error');
        end

        function testOpenArrayAsGroup(testCase)
            arrayPath = fullfile(testCase.TempDir, 'array_not_group');
            ZarrArray.create(arrayPath, [10, 10], 'uint8', 'chunkShape', [10, 10]);

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
            testCase.verifyError(@() grp.createArray('newarray', [10, 10], 'uint8'), 'zarr:error');
        end

        function testRemoteGetAttributes(testCase)
            grp = ZarrGroup(testCase.RemoteGroupPath);

            attrs = grp.getAttributes();

            testCase.verifyTrue(isstruct(attrs));
        end

        function testRemoteSetAttributesError(testCase)
            grp = ZarrGroup(testCase.RemoteGroupPath);

            testCase.verifyError(@() grp.setAttributes(struct('key', 'value')), 'zarr:error');
            testCase.verifyError(@() grp.setAttribute('key', 'value'), 'zarr:error');
        end
    end
end
