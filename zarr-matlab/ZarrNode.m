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

        function metadata = fetchMetadata(path)
            % FETCHMETADATA Fetch and parse zarr.json from local or HTTP path
            if ZarrNode.isHttpUrl(path)
                jsonUrl = ZarrNode.joinPath(path, 'zarr.json');
                try
                    options = weboptions('ContentType', 'json');
                    metadata = webread(jsonUrl, options);
                catch ME
                    error('zarr:error', 'Failed to fetch zarr.json from ''%s'': %s', path, ME.message);
                end
            else
                jsonPath = fullfile(path, 'zarr.json');
                if ~isfile(jsonPath)
                    error('zarr:error', 'No zarr.json found at ''%s''.', path);
                end
                metadata = jsondecode(fileread(jsonPath));
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

            metadata = ZarrNode.fetchMetadata(obj.path);
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

            jsonPath = fullfile(obj.path, 'zarr.json');
            metadata = jsondecode(fileread(jsonPath));
            metadata.attributes = attrs;

            fid = fopen(jsonPath, 'w');
            if fid == -1
                error('zarr:error', 'Failed to write zarr.json at %s', obj.path);
            end
            fprintf(fid, '%s', jsonencode(metadata));
            fclose(fid);
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
