function zarrPackage(version)
    % ZARRPACKAGE Package zarr-matlab as a MATLAB toolbox (.mltbx)
    %   zarrPackage()          - Package with version from Contents.m
    %   zarrPackage('0.2.0')   - Package with specified version
    %
    %   This function:
    %     1. Updates version in Contents.m and the project file
    %     2. Packages the toolbox as zarr-matlab.mltbx
    %
    %   Prerequisites:
    %     - MEX file must be built (run zarrBuild first)
    %     - MATLAB R2022a or later
    %
    %   The resulting .mltbx file can be:
    %     - Installed by double-clicking
    %     - Shared via MATLAB File Exchange
    %     - Distributed directly to users

    toolboxFolder = fileparts(mfilename('fullpath'));
    prjFile = fullfile(toolboxFolder, 'zarr-matlab.prj');

    if ~isfile(prjFile)
        error('zarr:error', 'Project file not found: %s', prjFile);
    end

    % Check that MEX file exists
    mexFiles = dir(fullfile(toolboxFolder, 'zarrMex.mex*'));
    if isempty(mexFiles)
        error('zarr:error', 'MEX file not found. Run zarrBuild() first.');
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

    % Update version in project file
    updateProjectVersion(prjFile, version);

    % Package the toolbox
    fprintf('Packaging toolbox...\n');
    outputFile = fullfile(toolboxFolder, 'zarr-matlab.mltbx');

    matlab.addons.toolbox.packageToolbox(prjFile, outputFile);

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

function updateProjectVersion(prjFile, version)
    text = fileread(prjFile);

    % Update version in project file
    text = regexprep(text, '<param\.version>[^<]*</param\.version>', ...
        sprintf('<param.version>%s</param.version>', version));

    fid = fopen(prjFile, 'w');
    fprintf(fid, '%s', text);
    fclose(fid);
end
