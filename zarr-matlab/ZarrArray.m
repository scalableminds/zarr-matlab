classdef ZarrArray < ZarrNode
    % ZARRARRAY Zarr array for reading and writing chunked array data
    %   A ZarrArray represents a chunked, compressed N-dimensional array.
    %   Existing arrays are opened in either Zarr v3 or v2 format, detected
    %   from the metadata on disk. New arrays default to v3; pass
    %   'zarrFormat', 2 to create v2 arrays.

    properties (Access = private, Transient)
        % Cached mex array handle, opened lazily on first use and released in
        % the destructor. Transient so a save/loaded object re-opens lazily
        % rather than carrying a stale handle from another session.
        handle = []
    end

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
            %                   For zarrFormat 2 only the separator applies, and it
            %                   becomes the 'dimension_separator' field.
            %     zarrFormat  - Zarr format version to create: 2 or 3. Default: 3
            %                   Zarr v2 has no sharding and a single compressor;
            %                   'filters' selects the chunk byte order instead of a
            %                   transpose codec (default order 'F', 'none' means 'C').

            p = inputParser;
            addParameter(p, 'chunkShape', [], @isnumeric);
            addParameter(p, 'shardShape', [], @isnumeric);
            addParameter(p, 'filters', [], @(x) ischar(x) || isstring(x) || isstruct(x) || iscell(x));
            addParameter(p, 'compressors', 'zstd', @(x) ischar(x) || isstring(x) || isstruct(x) || iscell(x));
            addParameter(p, 'fillValue', [], @(x) isempty(x) || isnumeric(x) || islogical(x));
            addParameter(p, 'chunkKeyEncoding', [], @(x) isempty(x) || ischar(x) || isstring(x) || isstruct(x));
            addParameter(p, 'zarrFormat', 3, @(x) isnumeric(x) && isscalar(x) && ismember(x, [2, 3]));
            parse(p, varargin{:});

            chunkShape = p.Results.chunkShape;
            if isempty(chunkShape)
                chunkShape = ZarrArray.defaultChunkShape(shape);
            end

            shardShape = p.Results.shardShape;
            filtersParam = p.Results.filters;
            compressorsParam = p.Results.compressors;

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

            if p.Results.zarrFormat == 2
                json = ZarrArray.buildMetadataJsonV2(shape, dataType, chunkShape, ...
                    shardShape, filtersParam, compressorsParam, fillValue, ...
                    p.Results.chunkKeyEncoding);
            else
                json = ZarrArray.buildMetadataJsonV3(shape, dataType, chunkShape, ...
                    shardShape, filtersParam, compressorsParam, fillValue, ...
                    p.Results.chunkKeyEncoding);
            end

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
            %     zarrFormat  - Zarr format version to create: 2 or 3 (default: 3)
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

            % Write the data at origin
            arr.write(data);
        end
    end

    methods (Static, Access = private)
        function json = buildMetadataJsonV3(shape, dataType, chunkShape, shardShape, ...
                filtersParam, compressorsParam, fillValue, chunkKeyEncodingParam)
            % BUILDMETADATAJSONV3 Build Zarr v3 zarr.json array metadata as JSON

            ndim = numel(shape);
            useSharding = ~isempty(shardShape);

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
                    'chunk_shape', {num2cell(chunkShape)}, ...
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
                'configuration', struct('chunk_shape', {num2cell(gridChunkShape)}) ...
                );

            % Build chunk key encoding
            chunkKeyEncoding = ZarrArray.buildChunkKeyEncoding(chunkKeyEncodingParam);

            json = jsonencode(struct( ...
                'zarr_format', 3, ...
                'node_type', 'array', ...
                'shape', {num2cell(shape)}, ...
                'data_type', dataType, ...
                'chunk_grid', chunkGrid, ...
                'chunk_key_encoding', chunkKeyEncoding, ...
                'fill_value', fillValue, ...
                'codecs', codecs ...
                ));
        end

        function json = buildMetadataJsonV2(shape, dataType, chunkShape, shardShape, ...
                filtersParam, compressorsParam, fillValue, chunkKeyEncodingParam)
            % BUILDMETADATAJSONV2 Build Zarr v2 .zarray array metadata as JSON

            if ~isempty(shardShape)
                error('zarr:error', ['Sharding is not supported by Zarr v2. Omit ' ...
                    '''shardShape'' or use ''zarrFormat'', 3.']);
            end

            [filters, order] = ZarrArray.buildFiltersAndOrderV2(filtersParam);
            compressor = ZarrArray.buildCompressorV2(compressorsParam);
            separator = ZarrArray.buildDimensionSeparatorV2(chunkKeyEncodingParam);

            % 'compressor' is a required field in the v2 metadata schema, so an
            % absent compressor has to be written as an explicit JSON null.
            % jsonencode renders NaN as null (its ConvertInfAndNaN default),
            % which is how buildCompressorV2 and buildFiltersAndOrderV2 signal
            % absence. Single braces around each value keep struct() from
            % expanding cell arrays into a struct array.
            json = jsonencode(struct( ...
                'zarr_format', 2, ...
                'shape', {num2cell(shape)}, ...
                'chunks', {num2cell(chunkShape)}, ...
                'dtype', ZarrArray.zarrTypeToDtypeV2(dataType), ...
                'compressor', {compressor}, ...
                'fill_value', {ZarrArray.fillValueForV2(fillValue)}, ...
                'order', order, ...
                'filters', {filters}, ...
                'dimension_separator', separator ...
                ));
        end

        function dtype = zarrTypeToDtypeV2(dataType)
            % ZARRTYPETODTYPEV2 Convert a Zarr data type name to a v2 dtype string
            %   Single-byte types take the '|' (not applicable) endianness prefix;
            %   multi-byte types are written little-endian, matching the endian
            %   configuration the v3 path gives its bytes codec.

            switch dataType
                case 'bool'
                    dtype = '|b1';
                case 'uint8'
                    dtype = '|u1';
                case 'int8'
                    dtype = '|i1';
                case 'uint16'
                    dtype = '<u2';
                case 'int16'
                    dtype = '<i2';
                case 'uint32'
                    dtype = '<u4';
                case 'int32'
                    dtype = '<i4';
                case 'uint64'
                    dtype = '<u8';
                case 'int64'
                    dtype = '<i8';
                case 'float32'
                    dtype = '<f4';
                case 'float64'
                    dtype = '<f8';
                otherwise
                    error('zarr:error', 'Unsupported data type for Zarr v2: %s', dataType);
            end
        end

        function value = fillValueForV2(fillValue)
            % FILLVALUEFORV2 Encode a fill value for Zarr v2 metadata
            %   jsonencode maps NaN and +/-Inf to null, but in v2 a null fill
            %   value means "no fill value at all", so non-finite floats are
            %   written using the spec's string forms instead.

            if isnumeric(fillValue) && isscalar(fillValue) && ~isfinite(fillValue)
                if isnan(fillValue)
                    value = 'NaN';
                elseif fillValue > 0
                    value = 'Infinity';
                else
                    value = '-Infinity';
                end
            else
                value = fillValue;
            end
        end

        function [filters, order] = buildFiltersAndOrderV2(filtersParam)
            % BUILDFILTERSANDORDERV2 Map the filters parameter onto v2 filters and order
            %   Zarr v2 has no transpose codec: the byte layout within a chunk is
            %   set by the 'order' field instead. The default therefore mirrors the
            %   v3 path's transpose filter by writing Fortran order, while an
            %   explicit filter list (or 'none') falls back to C order.
            %
            %   Returns NaN for the filters when there are none, which jsonencode
            %   writes as JSON null.

            if isempty(filtersParam)
                filters = NaN;
                order = 'F';
                return;
            end

            if (ischar(filtersParam) || isstring(filtersParam)) && strcmp(char(filtersParam), 'none')
                filters = NaN;
                order = 'C';
                return;
            end

            if iscell(filtersParam)
                specs = filtersParam;
            else
                specs = { filtersParam };
            end

            if isempty(specs)
                filters = NaN;
                order = 'C';
                return;
            end

            filters = cell(1, numel(specs));
            for i = 1:numel(specs)
                filters{i} = ZarrArray.buildCodecV2(specs{i});
            end
            order = 'C';
        end

        function compressor = buildCompressorV2(compressorsParam)
            % BUILDCOMPRESSORV2 Map the compressors parameter onto the v2 compressor
            %   Zarr v2 has a single compressor slot, so a sequence is rejected.
            %   Returns NaN when there is no compressor, which jsonencode writes
            %   as JSON null.

            if isempty(compressorsParam)
                compressor = NaN;
                return;
            end

            if (ischar(compressorsParam) || isstring(compressorsParam)) ...
                    && strcmp(char(compressorsParam), 'none')
                compressor = NaN;
                return;
            end

            if iscell(compressorsParam)
                if isempty(compressorsParam)
                    compressor = NaN;
                    return;
                end
                if numel(compressorsParam) > 1
                    error('zarr:error', ['Zarr v2 supports a single compressor, got %d. ' ...
                        'Use ''zarrFormat'', 3 for a codec sequence.'], numel(compressorsParam));
                end
                compressor = ZarrArray.buildCodecV2(compressorsParam{1});
                return;
            end

            compressor = ZarrArray.buildCodecV2(compressorsParam);
        end

        function codec = buildCodecV2(codecParam)
            % BUILDCODECV2 Build a v2 numcodecs metadata struct from a codec spec
            %   Zarr v2 codec metadata is flat: the codec name lives in an 'id'
            %   field and its configuration keys sit alongside it, rather than
            %   nested under 'configuration' as in v3.

            if ischar(codecParam) || isstring(codecParam)
                name = char(codecParam);
                config = [];
            elseif isstruct(codecParam)
                if ~isfield(codecParam, 'name')
                    error('zarr:error', 'Codec struct must have a ''name'' field');
                end
                name = codecParam.name;
                if isfield(codecParam, 'configuration')
                    config = codecParam.configuration;
                else
                    config = [];
                end
            else
                error('zarr:error', 'Codec must be a string or struct');
            end

            if ismember(name, {'transpose', 'bytes', 'crc32c', 'sharding_indexed'})
                error('zarr:error', ['The ''%s'' codec is Zarr v3 only. In v2 the byte order ' ...
                    'comes from the dtype and the chunk layout from the ''order'' field.'], name);
            end

            if isempty(config)
                config = ZarrArray.getDefaultCodecConfigV2(name);
            end

            codec = struct('id', name);
            if ~isempty(config)
                names = fieldnames(config);
                for i = 1:numel(names)
                    codec.(names{i}) = config.(names{i});
                end
            end
        end

        function config = getDefaultCodecConfigV2(codecName)
            % GETDEFAULTCODECCONFIGV2 Get default numcodecs configuration for a codec
            %   This cannot reuse getDefaultCodecConfig: numcodecs blosc takes an
            %   integer 'shuffle' (0 none, 1 byte-wise, 2 bit-wise) rather than the
            %   v3 string form, and takes no 'typesize'.

            switch codecName
                case 'zstd'
                    config = struct('level', 3);
                case {'gzip', 'zlib'}
                    config = struct('level', 5);
                case 'bz2'
                    config = struct('level', 1);
                case 'blosc'
                    config = struct('cname', 'lz4', 'clevel', 5, 'shuffle', 0, 'blocksize', 0);
                otherwise
                    config = [];
            end
        end

        function separator = buildDimensionSeparatorV2(param)
            % BUILDDIMENSIONSEPARATORV2 Map chunkKeyEncoding onto dimension_separator
            %   Zarr v2 chunk keys are always flat, so only the separator is
            %   configurable. Defaults to '/' to match the v3 path's default.

            if isempty(param)
                separator = '/';
            elseif ischar(param) || isstring(param)
                separator = char(param);
            elseif isstruct(param)
                if isfield(param, 'name') && ~ismember(param.name, {'default', 'v2'})
                    error('zarr:error', ['Zarr v2 supports only a chunk key separator, got ' ...
                        'chunk key encoding ''%s''.'], param.name);
                end
                if isfield(param, 'separator')
                    separator = char(param.separator);
                else
                    separator = '/';
                end
            else
                error('zarr:error', 'chunkKeyEncoding must be a string or struct');
            end

            if ~ismember(separator, {'/', '.'})
                error('zarr:error', ...
                    'Chunk key separator must be ''/'' or ''.'', got ''%s''.', separator);
            end
        end

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
                % For zstd, the Zarr V3 codec requires a 'checksum' field
                if strcmp(name, 'zstd') && ~isfield(config, 'checksum')
                    config.checksum = false;
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
                    config = struct('level', 3, 'checksum', false);
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
                'configuration', struct('order', {num2cell(order)}));
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

        function delete(obj)
            % DELETE Destructor: release the cached mex array handle.
            if ~isempty(obj.handle)
                try
                    zarrMex('close', obj.handle);
                catch
                    % Never throw from a destructor (e.g. after `clear mex`).
                end
                obj.handle = [];
            end
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
            %     zarrFormat  - Zarr format version of the array on disk (2 or 3)

            info = zarrMex('info', obj.getHandle());
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

            info = obj.info();
            dt = info.dataType;
        end

        function format = zarrFormat(obj)
            % ZARRFORMAT Get the Zarr format version of the array (2 or 3)
            %   format = arr.zarrFormat()

            info = obj.info();
            format = info.zarrFormat;
        end

        function resize(obj, newShape)
            % RESIZE Resize the array to a new shape
            %   arr.resize(newShape)
            %
            %   Arguments:
            %     newShape - New shape as a vector, e.g. [200, 200, 200]

            zarrMex('resize', obj.getHandle(), newShape);
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

            h = obj.getHandle();
            if nargin < 2 || isempty(bbox)
                % Read entire array. The mex derives the whole-array region
                % from the opened array itself, so we avoid a separate
                % obj.shape() open just to build the bbox here.
                data = zarrMex('read', h);
            else
                data = zarrMex('read', h, bbox);
            end
        end

        function write(obj, data, varargin)
            % WRITE Write data to a region of the array
            %   arr.write(data)                        % Write at origin [1, 1, ...]
            %   arr.write(data, bbox)                  % Write at specified region
            %   arr.write(data, 'allowResize', true)   % Write and extend array if needed
            %   arr.write(data, bbox, 'allowResize', true)
            %
            %   Arguments:
            %     data - Data to write (must match the bbox dimensions and array data type)
            %     bbox - (Optional) Bounding box as nx2 matrix with [start, end] for each dimension
            %            Uses 1-based indexing (MATLAB convention)
            %            If omitted, writes at origin based on data size.
            %
            %   Optional Name-Value Arguments:
            %     allowResize - If true, extends the array if bbox exceeds current shape.
            %                   Only extends, never shrinks. Default: false

            % Determine whether an explicit bbox was provided (nx2 numeric first vararg)
            hasBbox = ~isempty(varargin) && isnumeric(varargin{1}) && size(varargin{1}, 2) == 2;
            if hasBbox
                bbox = varargin{1};
                extraArgs = varargin(2:end);
            else
                extraArgs = varargin;
            end

            % Parse optional name-value arguments
            p = inputParser;
            addParameter(p, 'allowResize', false, @islogical);
            parse(p, extraArgs{:});
            allowResize = p.Results.allowResize;

            if ~hasBbox && ~allowResize
                % Fast path: write at origin, letting the mex derive the region
                % from the data's own dimensions. Avoids a separate obj.shape()
                % open just to build the bbox.
                zarrMex('write', obj.getHandle(), data);
                return;
            end

            if ~hasBbox
                % Origin write, but allowResize needs the current shape below,
                % so compute the bbox here. Use the array's dimensionality to
                % handle MATLAB's minimum 2D arrays (trim trailing singletons).
                arrayNdim = numel(obj.shape());
                dataShape = size(data);
                if numel(dataShape) > arrayNdim
                    dataShape = dataShape(1:arrayNdim);
                end
                bbox = [ones(numel(dataShape), 1), dataShape(:) + 1];
            end

            % Resize if needed and allowed
            if allowResize
                currentShape = obj.shape();
                bboxEnd = bbox(:, 2) - 1;  % Convert to 0-based end index
                newShape = max(currentShape(:), bboxEnd(:))';
                if any(newShape > currentShape)
                    obj.resize(newShape);
                end
            end

            zarrMex('write', obj.getHandle(), bbox, data);
        end
    end

    methods (Access = private)
        function h = getHandle(obj)
            % GETHANDLE Lazily open the array and cache its mex handle.
            if isempty(obj.handle)
                obj.handle = zarrMex('open', obj.path);
            end
            h = obj.handle;
        end
    end
end
