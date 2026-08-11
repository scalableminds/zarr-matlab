function zarrPackage(version)
    % ZARRPACKAGE Package zarr-matlab as a MATLAB toolbox (.mltbx)
    %   zarrPackage()          - Package with version from Contents.m
    %   zarrPackage('0.2.0')   - Package with specified version
    %
    %   This function:
    %     1. Discovers all .m and .mex* files in the toolbox folder
    %     2. Updates version in Contents.m
    %     3. Packages the toolbox as zarr-matlab.mltbx
    %
    %   Prerequisites:
    %     - At least one MEX file must be present (run zarrBuild first)
    %     - MATLAB R2022a or later
    %
    %   The resulting .mltbx file can be:
    %     - Installed by double-clicking
    %     - Shared via MATLAB File Exchange
    %     - Distributed directly to users

    toolboxFolder = fileparts(mfilename('fullpath'));

    % Check that at least one MEX file exists
    mexFiles = dir(fullfile(toolboxFolder, 'zarrMex.mex*'));
    if isempty(mexFiles)
        error('zarr:error', 'No MEX files found. Run zarrBuild() first.');
    end
    fprintf('Found %d MEX file(s):\n', numel(mexFiles));
    for i = 1:numel(mexFiles)
        fprintf('  - %s\n', mexFiles(i).name);
    end

    % Get or validate version
    if nargin < 1
        version = getVersionFromContents(toolboxFolder);
        fprintf('Using version from Contents.m: %s\n', version);
    else
        validateVersion(version);
        updateContentsVersion(toolboxFolder, version);
        fprintf('Updated version to: %s\n', version);
    end

    % Create toolbox options
    opts = matlab.addons.toolbox.ToolboxOptions(toolboxFolder, 'a1b2c3d4-e5f6-7890-abcd-ef1234567890');

    % Set toolbox metadata
    opts.ToolboxName = 'zarr-matlab';
    opts.ToolboxVersion = version;
    opts.Summary = 'Zarr v3 and v2 implementation for MATLAB based on zarrs';
    opts.Description = sprintf([...
        'A high-performance Zarr v3 and v2 implementation for MATLAB, powered by the Rust-based zarrs library.\n\n' ...
        'Features:\n' ...
        '- Read and write Zarr v3 and Zarr v2 arrays, with the format detected on open\n' ...
        '- Support for chunking and sharding (sharding is v3 only)\n' ...
        '- Compression codecs: zstd (default), gzip, blosc; v2 additionally supports zlib and bz2\n' ...
        '- HTTP/HTTPS remote access (read-only)\n' ...
        '- Group hierarchy support\n' ...
        '- Attributes for arrays and groups\n\n' ...
        'Supported data types: bool, uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64']);
    opts.AuthorName = 'scalable minds';
    opts.AuthorEmail = 'hello@scalableminds.com';
    opts.AuthorCompany = 'scalable minds';

    % Set MATLAB version requirements
    opts.MinimumMatlabRelease = 'R2022a';
    opts.MaximumMatlabRelease = '';

    % Collect files to include
    filesToInclude = {};

    % Add all .m files (except build/package scripts)
    mFiles = dir(fullfile(toolboxFolder, '*.m'));
    excludeFiles = {'zarrBuild.m', 'zarrPackage.m'};
    for i = 1:numel(mFiles)
        if ~ismember(mFiles(i).name, excludeFiles)
            filesToInclude{end+1} = fullfile(toolboxFolder, mFiles(i).name); %#ok<AGROW>
        end
    end

    % Add all MEX files
    for i = 1:numel(mexFiles)
        filesToInclude{end+1} = fullfile(toolboxFolder, mexFiles(i).name); %#ok<AGROW>
    end

    opts.ToolboxFiles = filesToInclude;
    opts.ToolboxMatlabPath = toolboxFolder;

    % Set output file
    outputFile = fullfile(toolboxFolder, 'zarr-matlab.mltbx');
    opts.OutputFile = outputFile;

    % Package the toolbox
    fprintf('Packaging toolbox with %d files...\n', numel(filesToInclude));
    matlab.addons.toolbox.packageToolbox(opts);

    fprintf('Successfully created: %s\n', outputFile);
    fprintf('\nTo install, double-click the .mltbx file or run:\n');
    fprintf('  matlab.addons.install(''%s'')\n', outputFile);
end

function version = getVersionFromContents(toolboxFolder)
    contentsFile = fullfile(toolboxFolder, 'Contents.m');
    if ~isfile(contentsFile)
        error('zarr:error', 'Contents.m not found');
    end

    text = fileread(contentsFile);
    tokens = regexp(text, 'Version\s+(\d+\.\d+\.\d+)', 'tokens');
    if isempty(tokens)
        error('zarr:error', 'Could not parse version from Contents.m');
    end
    version = tokens{1}{1};
end

function validateVersion(version)
    if ~regexp(version, '^\d+\.\d+\.\d+$')
        error('zarr:error', 'Invalid version format. Expected: X.Y.Z (e.g., 0.1.0)');
    end
end

function updateContentsVersion(toolboxFolder, version)
    contentsFile = fullfile(toolboxFolder, 'Contents.m');
    text = fileread(contentsFile);

    % Update version line
    dateStr = datestr(now, 'dd-mmm-yyyy'); %#ok<TNOW1,DATST>
    newVersionLine = sprintf('Version %s %s', version, dateStr);
    text = regexprep(text, 'Version\s+\d+\.\d+\.\d+\s+\d+-\w+-\d+', newVersionLine);

    fid = fopen(contentsFile, 'w');
    fprintf(fid, '%s', text);
    fclose(fid);
end
