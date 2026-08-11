classdef ZarrInteropHelper
    % ZARRINTEROPHELPER Shared helpers for the zarr-python interop test suite.
    %   Centralized here (rather than duplicated per test class) specifically
    %   so the deterministic-data formula and the fixed attribute set exist in
    %   exactly one place on the MATLAB side, matching zarr_interop_cli.py's
    %   single Python-side definition. Not itself a matlab.unittest.TestCase.

    methods (Static)
        function assertUvAvailable()
            % ASSERTUVAVAILABLE Fail hard, with install instructions, if `uv`
            %   or the interop CLI's dependencies aren't available. Intended
            %   for use from each test class's TestClassSetup.

            [status, out] = system('uv --version');
            if status ~= 0
                error('zarr:interop:uvMissing', ...
                    ['uv is required to run the zarr-python interop tests.\n' ...
                     'Install it: https://docs.astral.sh/uv/getting-started/installation/\n%s'], out);
            end

            % Front-load dependency resolution here so any zarr/numcodecs
            % install problem is reported once, clearly, rather than inside
            % the first test method's failure message.
            [status, out] = system(sprintf('uv run "%s" --help', ZarrInteropHelper.cliPath()));
            if status ~= 0
                error('zarr:interop:uvMissing', ...
                    'uv could not resolve the interop CLI''s dependencies (zarr, numcodecs):\n%s', out);
            end
        end

        function p = cliPath()
            p = fullfile(fileparts(mfilename('fullpath')), 'zarr_interop_cli.py');
        end

        function s = shapeArg(shape)
            % SHAPEARG Format a shape vector as a CLI-friendly comma list, e.g. "12,9,5"
            s = strjoin(arrayfun(@num2str, shape, 'UniformOutput', false), ',');
        end

        function attrs = defaultAttrs()
            % DEFAULTATTRS Fixed attribute set written/checked by both sides.
            %   Keep in sync with zarr_interop_cli.py's DEFAULT_ATTRS.
            attrs = struct('description', 'zarr-matlab interop fixture', ...
                'scale', 2.5, 'channel', 3, 'flag', true);
        end

        function data = deterministicData(shape, dataType)
            % DETERMINISTICDATA Regenerate the same N-D test array as
            %   zarr_interop_cli.py's deterministic_data(), from (shape,
            %   dataType) alone -- no shared file, no shared random seed.
            %   Keep this formula in sync with that function.

            n = prod(shape);
            idx = double(reshape(0:(n - 1), shape));
            switch dataType
                case 'bool'
                    data = mod(idx, 2) ~= 0;
                case {'uint8', 'int8'}
                    raw = mod(idx, 256);
                    if strcmp(dataType, 'int8')
                        raw = raw - 128;
                    end
                    data = cast(raw, dataType);
                case {'uint16', 'int16'}
                    raw = mod(idx, 60000);
                    if strcmp(dataType, 'int16')
                        raw = raw - 30000;
                    end
                    data = cast(raw, dataType);
                otherwise
                    raw = mod(idx, 1000003);
                    if startsWith(dataType, 'int') || startsWith(dataType, 'float')
                        raw = raw - 500001;
                    end
                    if strcmp(dataType, 'float32')
                        data = single(raw);
                    elseif strcmp(dataType, 'float64')
                        data = raw;
                    else
                        data = cast(raw, dataType);
                    end
            end
        end
    end
end
