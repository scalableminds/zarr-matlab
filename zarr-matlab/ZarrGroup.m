classdef ZarrGroup < handle
    properties (SetAccess = private)
        path
    end

    methods (Static)
        function grp = create(path)
            % CREATE Create a new Zarr group
            %   grp = ZarrGroup.create(path)
            %
            %   Arguments:
            %     path - Path where the group will be created

            if ZarrGroup.isHttpUrl(path)
                error('zarr:error', 'Cannot create groups at HTTP URLs');
            end

            if isfolder(path)
                error('zarr:error', 'Path already exists: %s', path);
            end

            mkdir(path);

            metadata = struct( ...
                'zarr_format', 3, ...
                'node_type', 'group' ...
            );

            jsonPath = fullfile(path, 'zarr.json');
            fid = fopen(jsonPath, 'w');
            if fid == -1
                error('zarr:error', 'Failed to create zarr.json at %s', path);
            end
            fprintf(fid, '%s', jsonencode(metadata));
            fclose(fid);

            grp = ZarrGroup(path);
        end
    end

    methods (Static, Access = private)
        function result = isHttpUrl(path)
            % ISHTTPURL Check if path is an HTTP/HTTPS URL
            result = startsWith(path, 'http://') || startsWith(path, 'https://');
        end

        function result = joinPath(basePath, name)
            % JOINPATH Join path components, handling both filesystem and HTTP paths
            if ZarrGroup.isHttpUrl(basePath)
                % For HTTP URLs, use forward slash and ensure no double slashes
                if endsWith(basePath, '/')
                    result = [basePath, name];
                else
                    result = [basePath, '/', name];
                end
            else
                result = fullfile(basePath, name);
            end
        end

        function metadata = fetchMetadata(path)
            % FETCHMETADATA Fetch and parse zarr.json from local or HTTP path
            if ZarrGroup.isHttpUrl(path)
                jsonUrl = ZarrGroup.joinPath(path, 'zarr.json');
                try
                    options = weboptions('ContentType', 'json');
                    metadata = webread(jsonUrl, options);
                catch ME
                    error('zarr:error', 'Failed to fetch zarr.json from ''%s'': %s', path, ME.message);
                end
            else
                jsonPath = fullfile(path, 'zarr.json');
                if ~isfile(jsonPath)
                    error('zarr:error', 'No zarr.json found at ''%s''. Not a valid Zarr v3 group.', path);
                end
                metadata = jsondecode(fileread(jsonPath));
            end
        end
    end

    methods
        function obj = ZarrGroup(path)
            % ZARRGROUP Open an existing Zarr group
            %   grp = ZarrGroup(path)
            %
            %   Arguments:
            %     path - Path to the existing Zarr group (local path or HTTP URL)

            metadata = ZarrGroup.fetchMetadata(path);

            if ~isfield(metadata, 'node_type') || ~strcmp(metadata.node_type, 'group')
                error('zarr:error', 'Path ''%s'' is not a Zarr group (node_type is not ''group'').', path);
            end

            obj.path = path;
        end

        function grp = openGroup(obj, name)
            % OPENGROUP Open a subgroup within this group
            %   subgrp = grp.openGroup(name)
            %
            %   Arguments:
            %     name - Name of the subgroup (relative path)
            %
            %   Returns:
            %     subgrp - ZarrGroup object for the subgroup

            subpath = ZarrGroup.joinPath(obj.path, name);
            grp = ZarrGroup(subpath);
        end

        function grp = createGroup(obj, name)
            % CREATEGROUP Create a new subgroup within this group
            %   subgrp = grp.createGroup(name)
            %
            %   Arguments:
            %     name - Name of the subgroup to create
            %
            %   Returns:
            %     subgrp - ZarrGroup object for the new subgroup

            if ZarrGroup.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot create groups at HTTP URLs');
            end

            subpath = fullfile(obj.path, name);
            grp = ZarrGroup.create(subpath);
        end

        function arr = openArray(obj, name)
            % OPENARRAY Open an array within this group
            %   arr = grp.openArray(name)
            %
            %   Arguments:
            %     name - Name of the array (relative path)
            %
            %   Returns:
            %     arr - ZarrArray object for the array

            subpath = ZarrGroup.joinPath(obj.path, name);
            arr = ZarrArray(subpath);
        end

        function arr = createArray(obj, name, shape, dataType, chunkShape, varargin)
            % CREATEARRAY Create a new array within this group
            %   arr = grp.createArray(name, shape, dataType, chunkShape)
            %   arr = grp.createArray(name, shape, dataType, chunkShape, 'shardShape', shardShape)
            %   arr = grp.createArray(name, shape, dataType, chunkShape, 'codec', 'zstd')
            %
            %   Arguments:
            %     name       - Name of the array to create
            %     shape      - Array shape as a vector
            %     dataType   - Data type string
            %     chunkShape - Chunk shape as a vector
            %
            %   Optional Name-Value Arguments:
            %     shardShape - Shard shape for sharded arrays
            %     codec      - Compression codec
            %
            %   Returns:
            %     arr - ZarrArray object for the new array

            if ZarrGroup.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot create arrays at HTTP URLs');
            end

            subpath = fullfile(obj.path, name);
            arr = ZarrArray.create(subpath, shape, dataType, chunkShape, varargin{:});
        end

        function arr = createArrayFromData(obj, name, data, chunkShape, varargin)
            % CREATEARRAYFROMDATA Create a new array from existing data within this group
            %   arr = grp.createArrayFromData(name, data, chunkShape)
            %   arr = grp.createArrayFromData(name, data, chunkShape, 'shardShape', shardShape)
            %   arr = grp.createArrayFromData(name, data, chunkShape, 'codec', 'zstd')
            %
            %   Arguments:
            %     name       - Name of the array to create
            %     data       - MATLAB array to store (data type and shape are inferred)
            %     chunkShape - Chunk shape as a vector
            %
            %   Optional Name-Value Arguments:
            %     shardShape - Shard shape for sharded arrays
            %     codec      - Compression codec
            %
            %   Returns:
            %     arr - ZarrArray object for the new array

            if ZarrGroup.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot create arrays at HTTP URLs');
            end

            subpath = fullfile(obj.path, name);
            arr = ZarrArray.createFromData(subpath, data, chunkShape, varargin{:});
        end

        function names = list(obj)
            % LIST List all items (groups and arrays) in this group
            %   names = grp.list()
            %
            %   Returns:
            %     names - Cell array of item names
            %
            %   Note: This method is not available for HTTP URLs

            if ZarrGroup.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot list contents of HTTP URLs (directory listing not supported)');
            end

            items = dir(obj.path);
            names = {};
            for i = 1:numel(items)
                item = items(i);
                if item.isdir && ~strcmp(item.name, '.') && ~strcmp(item.name, '..')
                    % Check if it has a zarr.json (valid zarr node)
                    jsonPath = fullfile(obj.path, item.name, 'zarr.json');
                    if isfile(jsonPath)
                        names{end + 1} = item.name; %#ok<AGROW>
                    end
                end
            end
        end

        function [groups, arrays] = listContents(obj)
            % LISTCONTENTS List groups and arrays separately
            %   [groups, arrays] = grp.listContents()
            %
            %   Returns:
            %     groups - Cell array of group names
            %     arrays - Cell array of array names
            %
            %   Note: This method is not available for HTTP URLs

            if ZarrGroup.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot list contents of HTTP URLs (directory listing not supported)');
            end

            items = dir(obj.path);
            groups = {};
            arrays = {};
            for i = 1:numel(items)
                item = items(i);
                if item.isdir && ~strcmp(item.name, '.') && ~strcmp(item.name, '..')
                    jsonPath = fullfile(obj.path, item.name, 'zarr.json');
                    if isfile(jsonPath)
                        metadata = jsondecode(fileread(jsonPath));
                        if isfield(metadata, 'node_type')
                            if strcmp(metadata.node_type, 'group')
                                groups{end + 1} = item.name; %#ok<AGROW>
                            elseif strcmp(metadata.node_type, 'array')
                                arrays{end + 1} = item.name; %#ok<AGROW>
                            end
                        end
                    end
                end
            end
        end
    end
end
