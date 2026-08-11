classdef ZarrNode < handle
    % ZARRNODE Base class for Zarr nodes (arrays and groups)
    %   This class provides common functionality for ZarrArray and ZarrGroup,
    %   including path handling, HTTP support, and attribute management.

    properties (SetAccess = protected)
        path
    end

    methods (Static, Access = protected)
        function result = isHttpUrl(path)
            % ISHTTPURL Check if path is an HTTP/HTTPS URL
            result = startsWith(path, 'http://') || startsWith(path, 'https://');
        end

        function result = joinPath(basePath, name)
            % JOINPATH Join path components, handling both filesystem and HTTP paths
            if ZarrNode.isHttpUrl(basePath)
                if endsWith(basePath, '/')
                    result = [basePath, name];
                else
                    result = [basePath, '/', name];
                end
            else
                result = fullfile(basePath, name);
            end
        end

        function text = readJsonFile(path, name)
            % READJSONFILE Read a JSON file from a node directory or URL
            %   Returns '' if the file does not exist. Over HTTP any failure is
            %   treated as "not present", since a 404 cannot be distinguished
            %   from other errors without inspecting the exception.
            if ZarrNode.isHttpUrl(path)
                try
                    options = weboptions('ContentType', 'text');
                    text = webread(ZarrNode.joinPath(path, name), options);
                catch
                    text = '';
                end
            else
                filePath = fullfile(path, name);
                if isfile(filePath)
                    text = fileread(filePath);
                else
                    text = '';
                end
            end
        end

        function writeJsonFile(path, name, value)
            % WRITEJSONFILE Write a value as JSON into a node directory
            filePath = fullfile(path, name);
            fid = fopen(filePath, 'w');
            if fid == -1
                error('zarr:error', 'Failed to write %s at %s', name, path);
            end
            fprintf(fid, '%s', jsonencode(value));
            fclose(fid);
        end

        function [metadata, zarrFormat, nodeType] = fetchNodeMetadata(path)
            % FETCHNODEMETADATA Fetch node metadata, detecting Zarr v3 or v2
            %   Looks for zarr.json (v3), then .zarray and .zgroup (v2).
            %
            %   Returns:
            %     metadata   - Decoded metadata. For v2 the contents of .zattrs
            %                  are merged in as metadata.attributes, so callers
            %                  see the same shape for both formats.
            %     zarrFormat - 2 or 3
            %     nodeType   - 'array', 'group', or '' if v3 metadata omits it

            text = ZarrNode.readJsonFile(path, 'zarr.json');
            if ~isempty(text)
                metadata = jsondecode(text);
                zarrFormat = ZarrNode.normalizeZarrFormat(metadata, 3);
                if isfield(metadata, 'node_type')
                    nodeType = metadata.node_type;
                else
                    nodeType = '';
                end
                return;
            end

            text = ZarrNode.readJsonFile(path, '.zarray');
            if ~isempty(text)
                metadata = jsondecode(text);
                zarrFormat = ZarrNode.normalizeZarrFormat(metadata, 2);
                metadata.attributes = ZarrNode.fetchV2Attributes(path);
                nodeType = 'array';
                return;
            end

            text = ZarrNode.readJsonFile(path, '.zgroup');
            if ~isempty(text)
                metadata = jsondecode(text);
                zarrFormat = ZarrNode.normalizeZarrFormat(metadata, 2);
                metadata.attributes = ZarrNode.fetchV2Attributes(path);
                nodeType = 'group';
                return;
            end

            error('zarr:error', ['No Zarr metadata found at ''%s'' ' ...
                '(looked for zarr.json, .zarray and .zgroup).'], path);
        end

        function zarrFormat = normalizeZarrFormat(metadata, defaultFormat)
            % NORMALIZEZARRFORMAT Read zarr_format, tolerating a quoted number
            %   Some writers emit {"zarr_format": "2"} in .zgroup, so a char or
            %   string value is converted rather than rejected.

            if ~isfield(metadata, 'zarr_format')
                zarrFormat = defaultFormat;
                return;
            end

            value = metadata.zarr_format;
            if ischar(value) || isstring(value)
                zarrFormat = str2double(value);
            else
                zarrFormat = double(value);
            end

            if ~ismember(zarrFormat, [2, 3])
                error('zarr:error', 'Unsupported zarr_format ''%s''. Expected 2 or 3.', string(value));
            end
        end

        function attrs = fetchV2Attributes(path)
            % FETCHV2ATTRIBUTES Read a Zarr v2 .zattrs file
            %   Returns an empty struct if the node has no .zattrs.
            text = ZarrNode.readJsonFile(path, '.zattrs');
            if isempty(text)
                attrs = struct();
            else
                attrs = jsondecode(text);
            end
        end

        function nodeType = detectNodeType(path)
            % DETECTNODETYPE Classify a local directory as a Zarr node
            %   Returns 'array', 'group', or '' if the directory is not a Zarr
            %   node. Local paths only; used for directory listing.

            jsonPath = fullfile(path, 'zarr.json');
            if isfile(jsonPath)
                metadata = jsondecode(fileread(jsonPath));
                if isfield(metadata, 'node_type')
                    nodeType = metadata.node_type;
                else
                    nodeType = '';
                end
            elseif isfile(fullfile(path, '.zarray'))
                nodeType = 'array';
            elseif isfile(fullfile(path, '.zgroup'))
                nodeType = 'group';
            else
                nodeType = '';
            end
        end
    end

    methods
        function attrs = getAttributes(obj)
            % GETATTRIBUTES Get all attributes of this node
            %   attrs = node.getAttributes()
            %
            %   Returns:
            %     attrs - Struct containing all attributes (empty struct if none)

            metadata = ZarrNode.fetchNodeMetadata(obj.path);
            if isfield(metadata, 'attributes')
                attrs = metadata.attributes;
            else
                attrs = struct();
            end
        end

        function setAttributes(obj, attrs)
            % SETATTRIBUTES Set all attributes of this node
            %   node.setAttributes(attrs)
            %
            %   Arguments:
            %     attrs - Struct containing attributes to set
            %
            %   Note: This method is not available for HTTP URLs

            if ZarrNode.isHttpUrl(obj.path)
                error('zarr:error', 'Cannot write attributes to HTTP URLs');
            end

            [~, zarrFormat] = ZarrNode.fetchNodeMetadata(obj.path);

            if zarrFormat == 2
                % Zarr v2 keeps attributes in their own .zattrs file, so the
                % whole file is the attribute object.
                ZarrNode.writeJsonFile(obj.path, '.zattrs', attrs);
            else
                metadata = jsondecode(fileread(fullfile(obj.path, 'zarr.json')));
                metadata.attributes = attrs;
                ZarrNode.writeJsonFile(obj.path, 'zarr.json', metadata);
            end
        end

        function value = getAttribute(obj, name)
            % GETATTRIBUTE Get a single attribute by name
            %   value = node.getAttribute(name)
            %
            %   Arguments:
            %     name - Name of the attribute
            %
            %   Returns:
            %     value - Value of the attribute
            %
            %   Throws error if attribute does not exist

            attrs = obj.getAttributes();
            if ~isfield(attrs, name)
                error('zarr:error', 'Attribute ''%s'' not found', name);
            end
            value = attrs.(name);
        end

        function setAttribute(obj, name, value)
            % SETATTRIBUTE Set a single attribute
            %   node.setAttribute(name, value)
            %
            %   Arguments:
            %     name  - Name of the attribute
            %     value - Value to set
            %
            %   Note: This method is not available for HTTP URLs

            attrs = obj.getAttributes();
            attrs.(name) = value;
            obj.setAttributes(attrs);
        end
    end
end
