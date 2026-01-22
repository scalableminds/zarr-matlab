classdef ZarrArray < handle
    properties (SetAccess = private)
        path
    end

    methods (Static)
        function arr = create(path, shape, dataType, chunkShape, varargin)
            % CREATE Create a new Zarr array
            %   arr = ZarrArray.create(path, shape, dataType, chunkShape)
            %   arr = ZarrArray.create(path, shape, dataType, chunkShape, 'shardShape', shardShape)
            %   arr = ZarrArray.create(path, shape, dataType, chunkShape, 'codec', 'zstd')
            %
            %   Arguments:
            %     path       - Path where the array will be created
            %     shape      - Array shape as a vector, e.g. [100, 100, 100]
            %     dataType   - Data type string: 'uint8', 'uint16', 'uint32', 'uint64',
            %                  'int8', 'int16', 'int32', 'int64', 'float32', 'float64'
            %     chunkShape - Chunk shape as a vector, e.g. [32, 32, 32]
            %
            %   Optional Name-Value Arguments:
            %     shardShape - Shard shape for sharded arrays (enables sharding codec)
            %     codec      - Compression codec, can be:
            %                  - String: 'zstd', 'gzip', 'blosc'
            %                  - Struct with 'name' and optional 'configuration' fields
            %                  Examples:
            %                    'zstd'
            %                    struct('name', 'zstd', 'configuration', struct('level', 5))
            %                    struct('name', 'gzip', 'configuration', struct('level', 6))
            %                    struct('name', 'blosc', 'configuration', struct( ...
            %                        'cname', 'lz4', 'clevel', 5, 'shuffle', 'shuffle'))

            p = inputParser;
            addRequired(p, 'path', @ischar);
            addRequired(p, 'shape', @isnumeric);
            addRequired(p, 'dataType', @ischar);
            addRequired(p, 'chunkShape', @isnumeric);
            addParameter(p, 'shardShape', [], @isnumeric);
            addParameter(p, 'codec', '', @(x) ischar(x) || isstruct(x));
            parse(p, path, shape, dataType, chunkShape, varargin{:});

            shardShape = p.Results.shardShape;
            codecParam = p.Results.codec;
            useSharding = ~isempty(shardShape);
            ndim = numel(shape);

            % Build transpose codec to convert between MATLAB's Fortran order and Zarr's C order
            transposeCodec = ZarrArray.buildTransposeCodec(ndim);

            % Build compression codec if specified
            compressionCodec = ZarrArray.buildCodec(codecParam, dataType);

            % Build the bytes codec with endian configuration if needed
            bytesCodec = ZarrArray.buildBytesCodec(dataType);

            % Build the inner codecs (transpose + bytes + optional compression)
            if isempty(compressionCodec)
                innerCodecs = {{ transposeCodec, bytesCodec }};
            else
                innerCodecs = {{ transposeCodec, bytesCodec, compressionCodec }};
            end

            if useSharding
                % For sharding, transpose the chunk shapes to match the transposed array
                codecs = {{ struct( ...
                    'name', 'sharding_indexed', ...
                    'configuration', struct( ...
                        'chunk_shape', chunkShape, ...
                        'codecs', innerCodecs, ...
                        'index_location', 'end', ...
                        'index_codecs', {{ ...
                            struct('name', 'bytes', 'configuration', struct('endian', 'little')), ...
                            struct('name', 'crc32c') }} ...
                    )) }};
                gridChunkShape = shardShape;
            else
                gridChunkShape = chunkShape;
                codecs = innerCodecs;
            end

            json = jsonencode(struct( ...
                'zarr_format', 3, ...
                'node_type', 'array', ...
                'shape', shape, ...
                'data_type', dataType, ...
                'chunk_grid', struct( ...
                    'name', 'regular', ...
                    'configuration', struct('chunk_shape', gridChunkShape) ...
                ), ...
                'chunk_key_encoding', struct( ...
                    'name', 'default', ...
                    'configuration', struct('separator', '/') ...
                ), ...
                'fill_value', 0, ...
                'codecs', codecs ...
            ));

            zarrMex('create', path, json);
            arr = ZarrArray(path);
        end
    end

    methods (Static, Access = private)
        function codec = buildCodec(codecParam, dataType)
            % BUILDCODEC Build a codec struct from string or struct input
            %   Adds default configuration for compression codecs if not provided
            if isempty(codecParam)
                codec = [];
                return;
            end

            if ischar(codecParam)
                % Simple string like 'zstd', 'gzip', 'blosc'
                name = codecParam;
                config = [];
            elseif isstruct(codecParam)
                % Struct with name and optional configuration
                if ~isfield(codecParam, 'name')
                    error('Codec struct must have a ''name'' field');
                end
                name = codecParam.name;
                if isfield(codecParam, 'configuration')
                    config = codecParam.configuration;
                else
                    config = [];
                end
            else
                error('Codec must be a string or struct');
            end

            % Add default configuration if not provided
            if isempty(config)
                config = ZarrArray.getDefaultCodecConfig(name, dataType);
            end

            if isempty(config)
                codec = struct('name', name);
            else
                codec = struct('name', name, 'configuration', config);
            end
        end

        function config = getDefaultCodecConfig(codecName, dataType)
            % GETDEFAULTCODECCONFIG Get default configuration for a codec
            switch codecName
                case 'zstd'
                    config = struct('level', 3);
                case 'gzip'
                    config = struct('level', 5);
                case 'blosc'
                    typesize = ZarrArray.getTypeSize(dataType);
                    config = struct('cname', 'lz4', 'clevel', 5, 'shuffle', 'noshuffle', 'typesize', typesize, 'blocksize', 0);
                otherwise
                    config = [];
            end
        end

        function size = getTypeSize(dataType)
            % GETTYPESIZE Get the size in bytes for a data type
            switch dataType
                case {'uint8', 'int8'}
                    size = 1;
                case {'uint16', 'int16'}
                    size = 2;
                case {'uint32', 'int32', 'float32'}
                    size = 4;
                case {'uint64', 'int64', 'float64'}
                    size = 8;
                otherwise
                    size = 1;
            end
        end

        function codec = buildBytesCodec(dataType)
            % BUILDBYTESCODEC Build bytes codec with endian config for multi-byte types
            %   Single-byte types (uint8, int8) don't need endian specification
            %   Multi-byte types need endian: 'little' or 'big'

            singleByteTypes = {'uint8', 'int8'};
            if ismember(dataType, singleByteTypes)
                codec = struct('name', 'bytes');
            else
                codec = struct('name', 'bytes', ...
                    'configuration', struct('endian', 'little'));
            end
        end

        function codec = buildTransposeCodec(ndim)
            % BUILDTRANSPOSECODEC Build transpose codec to reverse axis order
            %   Converts between MATLAB's Fortran (column-major) order and
            %   Zarr's C (row-major) order by reversing the axis order.
            %   For ndim dimensions, order is [ndim-1, ndim-2, ..., 1, 0]

            order = (ndim - 1):-1:0;
            codec = struct('name', 'transpose', ...
                'configuration', struct('order', order));
        end
    end

    methods
        function obj = ZarrArray(path)
            % ZARRARRAY Open an existing Zarr array
            %   arr = ZarrArray(path)
            %
            %   Arguments:
            %     path - Path to the existing Zarr array

            obj.path = path;
        end

        function info = info(obj)
            % INFO Get information about the array
            %   info = arr.info()
            %
            %   Returns a struct with fields:
            %     boundingBox - nx2 matrix with [start, end] for each dimension
            %     dataType    - Data type string
            %     chunkShape  - Chunk shape vector
            %     shardShape  - Shard shape vector (same as chunkShape if not sharded)

            info = zarrMex('info', obj.path);
        end

        function shape = shape(obj)
            % SHAPE Get the shape of the array
            %   shape = arr.shape()

            info = obj.info();
            shape = (info.boundingBox(:, 2) - info.boundingBox(:, 1))';
        end

        function resize(obj, newShape)
            % RESIZE Resize the array to a new shape
            %   arr.resize(newShape)
            %
            %   Arguments:
            %     newShape - New shape as a vector, e.g. [200, 200, 200]

            zarrMex('resize', obj.path, newShape);
        end

        function data = read(obj, bbox)
            % READ Read data from a region of the array
            %   data = arr.read(bbox)
            %
            %   Arguments:
            %     bbox - Bounding box as nx2 matrix with [start, end] for each dimension
            %            Uses 1-based indexing (MATLAB convention)
            %
            %   Returns:
            %     data - Array data from the specified region

            data = zarrMex('read', obj.path, bbox);
        end

        function write(obj, bbox, data)
            % WRITE Write data to a region of the array
            %   arr.write(bbox, data)
            %
            %   Arguments:
            %     bbox - Bounding box as nx2 matrix with [start, end] for each dimension
            %            Uses 1-based indexing (MATLAB convention)
            %     data - Data to write (must match the bbox dimensions and array data type)

            zarrMex('write', obj.path, bbox, data);
        end
    end
end
