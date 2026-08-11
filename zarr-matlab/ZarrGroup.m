classdef ZarrGroup < ZarrNode
    % ZARRGROUP Zarr group for organizing arrays and subgroups
    %   A ZarrGroup represents a node in the Zarr hierarchy that can contain
    %   arrays and other groups. Existing groups are opened in either Zarr v3
    %   or v2 format, detected from the metadata on disk. New groups default to
    %   v3; pass 'zarrFormat', 2 to create v2 groups.

    properties (Access = private)
        % Zarr format version of this group on disk (2 or 3), detected when the
        % group is opened. Child groups and arrays created through this group
        % inherit it by default. Read it via zarrFormat().
        format
    end

    methods (Static)
        function grp = create(path, varargin)
            % CREATE Create a new Zarr group
            %   grp = ZarrGroup.create(path)
            %   grp = ZarrGroup.create(path, 'zarrFormat', 2)
            %
            %   Arguments:
            %     path - Path where the group will be created
            %
            %   Optional Name-Value Arguments:
            %     zarrFormat - Zarr format version to create: 2 or 3 (default: 3)

            p = inputParser;
            addParameter(p, 'zarrFormat', 3, @(x) isnumeric(x) && isscalar(x) && ismember(x, [2, 3]));
            parse(p, varargin{:});

            if ZarrNode.isHttpUrl(path)
                error('zarr:error', 'Cannot create groups at HTTP URLs');
            end

            if isfolder(path)
                error('zarr:error', 'Path already exists: %s', path);
            end

            mkdir(path);

            if p.Results.zarrFormat == 2
                % Zarr v2 groups are a bare .zgroup; attributes live in .zattrs.
                ZarrNode.writeJsonFile(path, '.zgroup', struct('zarr_format', 2));
            else
                ZarrNode.writeJsonFile(path, 'zarr.json', struct( ...
                    'zarr_format', 3, ...
                    'node_type', 'group' ...
                    ));
            end

            grp = ZarrGroup(path);
        end
    end

    methods
        function obj = ZarrGroup(path)
            % ZARRGROUP Open an existing Zarr group
            %   grp = ZarrGroup(path)
            %
            %   Arguments:
            %     path - Path to the existing Zarr group (local path or HTTP URL)

            [~, zarrFormat, nodeType] = ZarrNode.fetchNodeMetadata(path);

            if ~strcmp(nodeType, 'group')
                error('zarr:error', 'Path ''%s'' is not a Zarr group.', path);
            end

            obj.path = path;
            obj.format = zarrFormat;
        end

        function format = zarrFormat(obj)
            % ZARRFORMAT Get the Zarr format version of the group (2 or 3)
            %   format = grp.zarrFormat()

            format = obj.format;
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

            subpath = ZarrNode.joinPath(obj.path, name);
            grp = ZarrGroup(subpath);
        end

        function grp = createGroup(obj, name, varargin)
            % CREATEGROUP Create a new subgroup within this group
            %   subgrp = grp.createGroup(name)
            %   subgrp = grp.createGroup(name, 'zarrFormat', 2)
            %
            %   Arguments:
            %     name - Name of the subgroup to create
            %
            %   Optional Name-Value Arguments:
            %     zarrFormat - Zarr format version to create: 2 or 3
            %                  Default: this group's own format
            %
            %   Returns:
            %     subgrp - ZarrGroup object for the new subgroup

            if ZarrNode.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot create groups at HTTP URLs');
            end

            subpath = fullfile(obj.path, name);
            args = obj.withInheritedFormat(varargin);
            grp = ZarrGroup.create(subpath, args{:});
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

            subpath = ZarrNode.joinPath(obj.path, name);
            arr = ZarrArray(subpath);
        end

        function arr = createArray(obj, name, shape, dataType, varargin)
            % CREATEARRAY Create a new array within this group
            %   arr = grp.createArray(name, shape, dataType)
            %   arr = grp.createArray(name, shape, dataType, 'chunkShape', [32, 32, 32])
            %   arr = grp.createArray(name, shape, dataType, 'compressors', 'zstd')
            %
            %   Arguments:
            %     name     - Name of the array to create
            %     shape    - Array shape as a vector
            %     dataType - Data type string
            %
            %   Optional Name-Value Arguments:
            %     chunkShape  - Chunk shape as a vector (default: min(shape, 100))
            %     shardShape  - Shard shape for sharded arrays
            %     filters     - Filter codecs (default: transpose, use 'none' to disable)
            %     compressors - Compression codecs (default: 'zstd', use 'none' to disable)
            %     fillValue   - Fill value for uninitialized chunks
            %                   (default: false for bool, 0 for numeric types)
            %     chunkKeyEncoding - Chunk key encoding ('/' or '.' or struct)
            %     zarrFormat  - Zarr format version to create: 2 or 3
            %                   Default: this group's own format
            %
            %   Returns:
            %     arr - ZarrArray object for the new array

            if ZarrNode.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot create arrays at HTTP URLs');
            end

            subpath = fullfile(obj.path, name);
            args = obj.withInheritedFormat(varargin);
            arr = ZarrArray.create(subpath, shape, dataType, args{:});
        end

        function arr = createArrayFromData(obj, name, data, varargin)
            % CREATEARRAYFROMDATA Create a new array from existing data within this group
            %   arr = grp.createArrayFromData(name, data)
            %   arr = grp.createArrayFromData(name, data, 'chunkShape', [32, 32, 32])
            %   arr = grp.createArrayFromData(name, data, 'compressors', 'zstd')
            %
            %   Arguments:
            %     name - Name of the array to create
            %     data - MATLAB array to store (data type and shape are inferred)
            %
            %   Optional Name-Value Arguments:
            %     chunkShape  - Chunk shape as a vector (default: min(shape, 100))
            %     shardShape  - Shard shape for sharded arrays
            %     filters     - Filter codecs (default: transpose, use 'none' to disable)
            %     compressors - Compression codecs (default: 'zstd', use 'none' to disable)
            %     fillValue   - Fill value for uninitialized chunks
            %                   (default: false for bool, 0 for numeric types)
            %     chunkKeyEncoding - Chunk key encoding ('/' or '.' or struct)
            %     zarrFormat  - Zarr format version to create: 2 or 3
            %                   Default: this group's own format
            %
            %   Returns:
            %     arr - ZarrArray object for the new array

            if ZarrNode.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot create arrays at HTTP URLs');
            end

            subpath = fullfile(obj.path, name);
            args = obj.withInheritedFormat(varargin);
            arr = ZarrArray.createFromData(subpath, data, args{:});
        end

        function names = list(obj)
            % LIST List all items (groups and arrays) in this group
            %   names = grp.list()
            %
            %   Returns:
            %     names - Cell array of item names
            %
            %   Note: This method is not available for HTTP URLs

            if ZarrNode.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot list contents of HTTP URLs (directory listing not supported)');
            end

            items = dir(obj.path);
            names = {};
            for i = 1:numel(items)
                item = items(i);
                if item.isdir && ~strcmp(item.name, '.') && ~strcmp(item.name, '..')
                    % Keep directories that are Zarr nodes in either format
                    nodeType = ZarrNode.detectNodeType(fullfile(obj.path, item.name));
                    if ~isempty(nodeType)
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

            if ZarrNode.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot list contents of HTTP URLs (directory listing not supported)');
            end

            items = dir(obj.path);
            groups = {};
            arrays = {};
            for i = 1:numel(items)
                item = items(i);
                if item.isdir && ~strcmp(item.name, '.') && ~strcmp(item.name, '..')
                    nodeType = ZarrNode.detectNodeType(fullfile(obj.path, item.name));
                    if strcmp(nodeType, 'group')
                        groups{end + 1} = item.name; %#ok<AGROW>
                    elseif strcmp(nodeType, 'array')
                        arrays{end + 1} = item.name; %#ok<AGROW>
                    end
                end
            end
        end
    end

    methods (Access = private)
        function args = withInheritedFormat(obj, args)
            % WITHINHERITEDFORMAT Default a child's zarrFormat to this group's
            %   Children of a v2 group are created as v2 unless the caller passes
            %   an explicit 'zarrFormat', which always wins. The name is matched
            %   the way inputParser would (case-insensitively, allowing a partial
            %   name), so appending our own copy can never collide with theirs.

            isFormatName = @(a) (ischar(a) || isstring(a)) && ~isempty(char(a)) ...
                && startsWith('zarrformat', lower(char(a)));
            if ~any(cellfun(isFormatName, args))
                args = [args, {'zarrFormat', obj.format}];
            end
        end
    end
end
