function zarrTest()
    path = '../testdata/l4_sample/segmentation/1';

    % Test read command
    bbox = [1, 2; 3074, 3075; 3074, 3075; 514, 515];
    data = zarrMex('read', path, bbox);
    assert(data == 988274);

    % Test write command
    data = cast(reshape(linspace(0, 999, 1000), [1, 10, 10, 10]), "uint32");
    bbox = [1, 2; 1, 11; 1, 11; 1, 11];
    zarrMex('write', path, bbox, data);
    data_Read = zarrMex('read', path, bbox);
    assert(isequal(data, data_Read));

    % Test create command
    test_path = '/tmp/zarr_test';
    if isfolder(test_path)
        rmdir(test_path, 's');
    end
    json = jsonencode(struct( ...
        'zarr_format', 3, ...
        'node_type', 'array', ...
        'shape', [100, 100, 100], ...
        'data_type', 'uint8', ...
        'chunk_grid', struct( ...
            'name', 'regular', ...
            'configuration', struct('chunk_shape', [32, 32, 32]) ...
        ), ...
        'chunk_key_encoding', struct( ...
            'name', 'default', ...
            'configuration', struct('separator', '/') ...
        ), ...
        'fill_value', 0, ...
        'codecs', {{ struct( ...
            'name', 'sharding_indexed', ...
            'configuration', struct( ...
                'chunk_shape', [32, 32, 32], ...
                'codecs', {{ struct('name', 'bytes') }}, ...
                'index_location', 'end', ...
                'index_codecs', {{ struct('name', 'bytes'), struct('name', 'crc32c') }} ...
            )) }} ...
    ));
    zarrMex('create', test_path, json);
    assert(isfile(fullfile(test_path, 'zarr.json')));

    % Compare zarr.json contents with original JSON
    zarr_json_path = fullfile(test_path, 'zarr.json');
    written_json = fileread(zarr_json_path);
    original_struct = jsondecode(json);
    written_struct = jsondecode(written_json);
    assert(isequal(original_struct, written_struct), ...
        'zarr.json contents do not match the original JSON');

    % Test resize command
    new_shape = [200, 150, 100];
    zarrMex('resize', test_path, new_shape);
    resized_json = fileread(zarr_json_path);
    resized_struct = jsondecode(resized_json);
    assert(isequal(resized_struct.shape, new_shape'), ...
        'Shape after resize does not match expected shape');

    % Test info command
    info = zarrMex('info', test_path);
    disp(info);
    expected_bbox = [1, 201; 1, 151; 1, 101];
    assert(isequal(info.boundingBox, expected_bbox), ...
        'Bounding box does not match expected value');
    assert(strcmp(info.dataType, 'uint8'), ...
        'Data type does not match expected value');
    assert(isequal(info.chunkShape, [32, 32, 32]), ...
        'Chunk shape does not match expected value');
    assert(isequal(info.shardShape, [32, 32, 32]), ...
        'Shard shape does not match expected value');

    % Test ZarrArray class
    testZarrArrayClass();

end

function testZarrArrayClass()
    % Test ZarrArray.create without sharding
    test_path = '/tmp/zarr_class_test';
    if isfolder(test_path)
        rmdir(test_path, 's');
    end

    arr = ZarrArray.create(test_path, [100, 100, 100], 'uint16', [32, 32, 32]);
    assert(isfile(fullfile(test_path, 'zarr.json')), ...
        'zarr.json not created');

    % Test info method
    info = arr.info();
    assert(isequal(info.boundingBox, [1, 101; 1, 101; 1, 101]), ...
        'ZarrArray.info boundingBox mismatch');
    assert(strcmp(info.dataType, 'uint16'), ...
        'ZarrArray.info dataType mismatch');
    assert(isequal(info.chunkShape, [32, 32, 32]), ...
        'ZarrArray.info chunkShape mismatch');

    % Test shape method
    assert(isequal(arr.shape(), [100, 100, 100]), ...
        'ZarrArray.shape mismatch');

    % Test resize method
    arr.resize([150, 120, 100]);
    assert(isequal(arr.shape(), [150, 120, 100]), ...
        'ZarrArray.resize failed');

    % Test write and read methods
    test_data = uint16(reshape(1:1000, [10, 10, 10]));
    bbox = [1, 11; 1, 11; 1, 11];
    arr.write(bbox, test_data);
    read_data = arr.read(bbox);
    assert(isequal(test_data, read_data), ...
        'ZarrArray write/read roundtrip failed');

    % Test ZarrArray.create with sharding
    test_path_sharded = '/tmp/zarr_class_test_sharded';
    if isfolder(test_path_sharded)
        rmdir(test_path_sharded, 's');
    end

    arr_sharded = ZarrArray.create(test_path_sharded, [128, 128, 128], 'float32', ...
        [32, 32, 32], 'shardShape', [64, 64, 64]);

    info_sharded = arr_sharded.info();
    assert(strcmp(info_sharded.dataType, 'float32'), ...
        'Sharded array dataType mismatch');
    assert(isequal(info_sharded.chunkShape, [32, 32, 32]), ...
        'Sharded array chunkShape mismatch');
    assert(isequal(info_sharded.shardShape, [64, 64, 64]), ...
        'Sharded array shardShape mismatch');

    % Test opening existing array
    arr_opened = ZarrArray(test_path);
    assert(isequal(arr_opened.shape(), [150, 120, 100]), ...
        'Opening existing array failed');

    % Test ZarrArray.create with zstd codec (simple string)
    test_path_zstd = '/tmp/zarr_class_test_zstd';
    if isfolder(test_path_zstd)
        rmdir(test_path_zstd, 's');
    end
    arr_zstd = ZarrArray.create(test_path_zstd, [64, 64, 64], 'uint8', [32, 32, 32], ...
        'codec', 'zstd');
    test_data_zstd = uint8(randi(255, [32, 32, 32]));
    arr_zstd.write([1, 33; 1, 33; 1, 33], test_data_zstd);
    read_data_zstd = arr_zstd.read([1, 33; 1, 33; 1, 33]);
    assert(isequal(test_data_zstd, read_data_zstd), ...
        'ZarrArray with zstd codec write/read roundtrip failed');

    % Test ZarrArray.create with zstd codec and configuration
    test_path_zstd_cfg = '/tmp/zarr_class_test_zstd_cfg';
    if isfolder(test_path_zstd_cfg)
        rmdir(test_path_zstd_cfg, 's');
    end
    arr_zstd_cfg = ZarrArray.create(test_path_zstd_cfg, [64, 64, 64], 'int32', [32, 32, 32], ...
        'codec', struct('name', 'zstd', 'configuration', struct('level', 10)));
    test_data_zstd_cfg = int32(randi(1000000, [32, 32, 32]));
    arr_zstd_cfg.write([1, 33; 1, 33; 1, 33], test_data_zstd_cfg);
    read_data_zstd_cfg = arr_zstd_cfg.read([1, 33; 1, 33; 1, 33]);
    assert(isequal(test_data_zstd_cfg, read_data_zstd_cfg), ...
        'ZarrArray with zstd codec (configured) write/read roundtrip failed');

    % Test ZarrArray.create with gzip codec
    test_path_gzip = '/tmp/zarr_class_test_gzip';
    if isfolder(test_path_gzip)
        rmdir(test_path_gzip, 's');
    end
    arr_gzip = ZarrArray.create(test_path_gzip, [64, 64, 64], 'float64', [32, 32, 32], ...
        'codec', struct('name', 'gzip', 'configuration', struct('level', 6)));
    test_data_gzip = rand(32, 32, 32);
    arr_gzip.write([1, 33; 1, 33; 1, 33], test_data_gzip);
    read_data_gzip = arr_gzip.read([1, 33; 1, 33; 1, 33]);
    assert(max(abs(test_data_gzip(:) - read_data_gzip(:))) < 1e-10, ...
        'ZarrArray with gzip codec write/read roundtrip failed');

    % Test ZarrArray.create with sharding and zstd codec
    test_path_shard_zstd = '/tmp/zarr_class_test_shard_zstd';
    if isfolder(test_path_shard_zstd)
        rmdir(test_path_shard_zstd, 's');
    end
    arr_shard_zstd = ZarrArray.create(test_path_shard_zstd, [128, 128, 128], 'uint16', ...
        [16, 16, 16], 'shardShape', [64, 64, 64], 'codec', 'zstd');
    test_data_shard = uint16(randi(65535, [32, 32, 32]));
    arr_shard_zstd.write([1, 33; 1, 33; 1, 33], test_data_shard);
    read_data_shard = arr_shard_zstd.read([1, 33; 1, 33; 1, 33]);
    assert(isequal(test_data_shard, read_data_shard), ...
        'ZarrArray with sharding and zstd codec write/read roundtrip failed');

    disp('All ZarrArray class tests passed!');
end