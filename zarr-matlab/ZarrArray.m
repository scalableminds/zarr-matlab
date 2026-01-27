classdef ZarrArray < ZarrNode
    % ZARRARRAY Zarr v3 array for reading and writing chunked array data
    %   A ZarrArray represents a chunked, compressed N-dimensional array.

    methods (Static)
        function arr = create(path, shape, dataType, varargin)
            % CREATE Create a new Zarr array
            %   arr = ZarrArray.create(path, shape, dataType)
            %   arr = ZarrArray.create(path, shape, dataType, 'chunkShape', [32, 32, 32])
            %   arr = ZarrArray.create(path, shape, dataType, 'shardShape', [128, 128, 128])
            %   arr = ZarrArray.create(path, shape, dataType, 'filters', 'none')
            %   arr = ZarrArray.create(path, shape, dataType, 'compressors', 'zstd')
            %
            %   Arguments:
            %     path       - Path where the array will be created
            %     shape      - Array shape as a vector, e.g. [100, 100, 100]
            %     dataType   - Data type string: 'bool', 'uint8', 'uint16', 'uint32', 'uint64',
            %                  'int8', 'int16', 'int32', 'int64', 'float32', 'float64'
            %
            %   Optional Name-Value Arguments:
            %     chunkShape  - Chunk shape as a vector, e.g. [32, 32, 32]
            %                   Default: min(shape, 100) per dimension
            %     shardShape  - Shard shape for sharded arrays (enables sharding codec)
            %     filters     - Filter codecs applied before compression. Can be:
            %                   - Cell array of codec specs (sequence)
            %                   - Single codec spec (string or struct)
            %                   - 'none' to disable filters
            %                   Default: transpose codec with Fortran order
            %     compressors - Compression codecs. Can be:
            %                   - Cell array of codec specs (sequence)
            %                   - Single codec spec (string or struct)
            %                   - 'none' to disable compression
            %                   Default: 'zstd'
            %                   Examples:
            %                     'zstd'
            %                     'none'  % disable compression
            %                     struct('name', 'zstd', 'configuration', struct('level', 5))
            %                     {struct('name', 'gzip'), struct('name', 'crc32c')}
            %     fillValue   - Fill value for uninitialized chunks.
            %                   Default: false for bool, 0 for numeric types
            %     chunkKeyEncoding - Chunk key encoding configuration. Can be:
            %                   - String: '/' or '.' for separator (uses 'default' name)
            %                   - Struct with 'name' ('default' or 'v2') and optional 'separator'
            %                   Default: struct('name', 'default', 'separator', '/')

            p = inputParser;
            addParameter(p, 'chunkShape', [], @isnumeric);
            addParameter(p, 'shardShape', [], @isnumeric);
            addParameter(p, 'filters', [], @(x) ischar(x) || isstring(x) || isstruct(x) || iscell(x));
            addParameter(p, 'compressors', 'zstd', @(x) ischar(x) || isstring(x) || isstruct(x) || iscell(x));
            addParameter(p, 'fillValue', [], @(x) isempty(x) || isnumeric(x) || islogical(x));
            addParameter(p, 'chunkKeyEncoding', [], @(x) isempty(x) || ischar(x) || isstring(x) || isstruct(x));
            parse(p, varargin{:});

            chunkShape = p.Results.chunkShape;
            if isempty(chunkShape)
                chunkShape = ZarrArray.defaultChunkShape(shape);
            end

            shardShape = p.Results.shardShape;
            filtersParam = p.Results.filters;
            compressorsParam = p.Results.compressors;
            useSharding = ~isempty(shardShape);

            % Set default fill value based on data type
            if isempty(p.Results.fillValue)
                if strcmp(dataType, 'bool')
                    fillValue = false;
                else
                    fillValue = 0;
                end
            else
                fillValue = p.Results.fillValue;
            end
            ndim = numel(shape);

            % Build filter codecs (default: transpose for Fortran order)
            if isempty(filtersParam)
                % Default: transpose codec for MATLAB's Fortran order
                filterCodecs = { ZarrArray.buildTransposeCodec(ndim) };
            else
                filterCodecs = ZarrArray.normalizeCodecs(filtersParam, dataType);
            end

            % Build compressor codecs (default: zstd)
            compressorCodecs = ZarrArray.normalizeCodecs(compressorsParam, dataType);

            % Build the bytes codec with endian configuration if needed
            bytesCodec = ZarrArray.buildBytesCodec(dataType);

            % Build the inner codecs: filters + bytes + compressors
            % Double curly braces {{ }} are needed so struct() treats this as a single
            % cell array value rather than creating multiple struct elements
            innerCodecsList = [filterCodecs, {bytesCodec}, compressorCodecs];
            innerCodecs = {{ innerCodecsList{:} }}; %#ok<CCAT1>

            if useSharding
                % For sharding, transpose the chunk shapes to match the transposed array
                indexCodecs = {{ ...
                    struct('name', 'bytes', 'configuration', struct('endian', 'little')), ...
                    struct('name', 'crc32c') }};
                codecs = {{ struct( ...
                    'name', 'sharding_indexed', ...
                    'configuration', struct( ...
                    'chunk_shape', chunkShape, ...
                    'codecs', innerCodecs, ...
                    'index_location', 'end', ...
                    'index_codecs', indexCodecs ...
                    )) }};
                gridChunkShape = shardShape;
            else
                gridChunkShape = chunkShape;
                codecs = innerCodecs;
            end

            chunkGrid = struct( ...
                'name', 'regular', ...
                'configuration', struct('chunk_shape', gridChunkShape) ...
                );

            % Build chunk key encoding
            chunkKeyEncoding = ZarrArray.buildChunkKeyEncoding(p.Results.chunkKeyEncoding);

            json = jsonencode(struct( ...
                'zarr_format', 3, ...
                'node_type', 'array', ...
                'shape', shape, ...
                'data_type', dataType, ...
                'chunk_grid', chunkGrid, ...
                'chunk_key_encoding', chunkKeyEncoding, ...
                'fill_value', fillValue, ...
                'codecs', codecs ...
                ));

            zarrMex('create', path, json);
            arr = ZarrArray(path);
        end

        function arr = createFromData(path, data, varargin)
            % CREATEFROMDATA Create a new Zarr array from existing data
            %   arr = ZarrArray.createFromData(path, data)
            %   arr = ZarrArray.createFromData(path, data, 'chunkShape', [32, 32, 32])
            %   arr = ZarrArray.createFromData(path, data, 'compressors', 'zstd')
            %
            %   Arguments:
            %     path - Path where the array will be created
            %     data - MATLAB array to store (data type and shape are inferred)
            %
            %   Optional Name-Value Arguments:
            %     chunkShape  - Chunk shape as a vector, e.g. [32, 32, 32]
            %                   Default: min(shape, 100) per dimension
            %     shardShape  - Shard shape for sharded arrays (enables sharding codec)
            %     filters     - Filter codecs (default: transpose, use 'none' to disable)
            %     compressors - Compression codecs (default: 'zstd', use 'none' to disable)
            %     fillValue   - Fill value for uninitialized chunks
            %                   (default: false for bool, 0 for numeric types)
            %     chunkKeyEncoding - Chunk key encoding ('/' or '.' or struct)
            %
            %   Example:
            %     data = uint16(rand(100, 100, 100) * 65535);
            %     arr = ZarrArray.createFromData('/path/to/array', data, 'compressors', 'zstd');

            % Infer shape from data
            shape = size(data);

            % Infer data type from data
            dataType = ZarrArray.matlabClassToZarrType(class(data));

            % Create array and write data
            arr = ZarrArray.create(path, shape, dataType, varargin{:});

            % Write the data
            bbox = [ones(numel(shape), 1), (shape(:) + 1)];
            arr.write(bbox, data);
        end
    end

    methods (Static, Access = private)
        function chunkShape = defaultChunkShape(shape)
            % DEFAULTCHUNKSHAPE Compute default chunk shape as min(shape, 100) per dimension
            chunkShape = min(shape, 100);
        end

        function codecs = normalizeCodecs(codecParam, dataType)
            % NORMALIZECODECS Normalize codec input to a cell array of codec structs
            %   Handles: 'none', single codec (string/struct), or cell array of codecs
            %   Returns a cell array of codec structs (empty if 'none')

            if ischar(codecParam) || isstring(codecParam)
                codecParam = char(codecParam);
                if strcmp(codecParam, 'none')
                    codecs = {};
                    return;
                end
                % Single codec as string
                codec = ZarrArray.buildCodec(codecParam, dataType);
                codecs = { codec };
            elseif isstruct(codecParam)
                % Single codec as struct
                codec = ZarrArray.buildCodec(codecParam, dataType);
                codecs = { codec };
            elseif iscell(codecParam)
                % Sequence of codecs
                codecs = cell(1, numel(codecParam));
                for i = 1:numel(codecParam)
                    codecs{i} = ZarrArray.buildCodec(codecParam{i}, dataType);
                end
            else
                error('zarr:error', 'Codec parameter must be ''none'', a string, a struct, or a cell array');
            end
        end

        function zarrType = matlabClassToZarrType(matlabClass)
            % MATLABCLASSTOZARRTYPE Convert MATLAB class name to Zarr data type string
            switch matlabClass
                case 'logical'
                    zarrType = 'bool';
                case 'single'
                    zarrType = 'float32';
                case 'double'
                    zarrType = 'float64';
                case {'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64'}
                    zarrType = matlabClass;
                otherwise
                    error('zarr:error', 'Unsupported data type: %s', matlabClass);
            end
        end

        function codec = buildCodec(codecParam, dataType)
            % BUILDCODEC Build a codec struct from string or struct input
            %   Adds default configuration for compression codecs if not provided
            %   Use 'none' to explicitly disable compression
            if isempty(codecParam) || (ischar(codecParam) && strcmp(codecParam, 'none'))
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
            else
                % For blosc, add typesize if not provided
                if strcmp(name, 'blosc') && ~isfield(config, 'typesize')
                    config.typesize = ZarrArray.getTypeSize(dataType);
                end
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
                case {'bool', 'uint8', 'int8'}
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
            %   Single-byte types (bool, uint8, int8) don't need endian specification
            %   Multi-byte types need endian: 'little' or 'big'

            singleByteTypes = {'bool', 'uint8', 'int8'};
            if ismember(dataType, singleByteTypes)
                codec = struct('name', 'bytes');
            else
                codec = struct('name', 'bytes', ...
                    'configuration', struct('endian', 'little'));
            end
        end

        function encoding = buildChunkKeyEncoding(param)
            % BUILDCHUNKKEYENCODING Build chunk key encoding struct
            %   Accepts: empty (default), string (separator), or struct
            %   Returns struct with 'name' and 'configuration' fields

            if isempty(param)
                % Default: 'default' name with '/' separator
                encoding = struct( ...
                    'name', 'default', ...
                    'configuration', struct('separator', '/') ...
                    );
            elseif ischar(param) || isstring(param)
                % String specifies separator, use 'default' name
                encoding = struct( ...
                    'name', 'default', ...
                    'configuration', struct('separator', char(param)) ...
                    );
            elseif isstruct(param)
                % Struct with 'name' and optional 'separator'
                if ~isfield(param, 'name')
                    error('zarr:error', 'chunkKeyEncoding struct must have a ''name'' field');
                end
                name = param.name;
                if isfield(param, 'separator')
                    separator = param.separator;
                else
                    separator = '/';  % default separator
                end
                encoding = struct( ...
                    'name', name, ...
                    'configuration', struct('separator', separator) ...
                    );
            else
                error('zarr:error', 'chunkKeyEncoding must be a string or struct');
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
            %     shape       - Array shape as a vector
            %     dataType    - Data type string
            %     chunkShape  - Chunk shape vector
            %     shardShape  - Shard shape vector (same as chunkShape if not sharded)

            info = zarrMex('info', obj.path);
        end

        function shape = shape(obj)
            % SHAPE Get the shape of the array
            %   shape = arr.shape()

            info = obj.info();
            shape = info.shape;
        end

        function dt = dataType(obj)
            % DATATYPE Get the data type of the array
            %   dt = arr.dataType()
            %
            %   Returns:
            %     dt - Data type string (e.g., 'uint8', 'float32', 'bool')

            info = obj.info();
            dt = info.dataType;
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
            %   data = arr.read()       % Read entire array
            %   data = arr.read(bbox)   % Read specified region
            %
            %   Arguments:
            %     bbox - (Optional) Bounding box as nx2 matrix with [start, end] for each dimension
            %            Uses 1-based indexing (MATLAB convention)
            %            If omitted, reads the entire array.
            %
            %   Returns:
            %     data - Array data from the specified region

            if nargin < 2 || isempty(bbox)
                % Read entire array
                arrayShape = obj.shape();
                ndim = numel(arrayShape);
                bbox = [ones(ndim, 1), arrayShape(:) + 1];
            end

            data = zarrMex('read', obj.path, bbox);
        end

        function write(obj, varargin)
            % WRITE Write data to a region of the array
            %   arr.write(data)                        % Write at origin [1, 1, ...]
            %   arr.write(bbox, data)                  % Write at specified region
            %   arr.write(data, 'allowResize', true)   % Write and extend array if needed
            %   arr.write(bbox, data, 'allowResize', true)
            %
            %   Arguments:
            %     bbox - (Optional) Bounding box as nx2 matrix with [start, end] for each dimension
            %            Uses 1-based indexing (MATLAB convention)
            %            If omitted, writes at origin based on data size.
            %     data - Data to write (must match the bbox dimensions and array data type)
            %
            %   Optional Name-Value Arguments:
            %     allowResize - If true, extends the array if bbox exceeds current shape.
            %                   Only extends, never shrinks. Default: false

            % Parse arguments: first determine if bbox was provided
            if size(varargin{1}, 2) == 2 && size(varargin{1}, 1) >= 1 && isnumeric(varargin{1}) && ~isvector(varargin{1})
                % First arg is bbox (nx2 matrix)
                bbox = varargin{1};
                data = varargin{2};
                extraArgs = varargin(3:end);
            else
                % First arg is data
                data = varargin{1};
                dataShape = size(data);
                bbox = [ones(numel(dataShape), 1), dataShape(:) + 1];
                extraArgs = varargin(2:end);
            end

            % Parse optional name-value arguments
            p = inputParser;
            addParameter(p, 'allowResize', false, @islogical);
            parse(p, extraArgs{:});

            % Resize if needed and allowed
            if p.Results.allowResize
                currentShape = obj.shape();
                bboxEnd = bbox(:, 2) - 1;  % Convert to 0-based end index
                newShape = max(currentShape(:), bboxEnd(:))';
                if any(newShape > currentShape)
                    obj.resize(newShape);
                end
            end

            zarrMex('write', obj.path, bbox, data);
        end
    end
end
