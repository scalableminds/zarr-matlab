function zarrBuild()
    % Written by
    %   Benedikt Staffler <benedikt.staffler@brain.mpg.de>
    %   Alessandro Motta <alessandro.motta@brain.mpg.de>

    % export path to library
    matlabRoot = matlabroot();
    
    % path to .so / .dll files
    arch = computer('arch');
    extraLinkPaths = {fullfile(matlabRoot, 'bin', arch)};
    
    if ispc
        % on Windows, we also require
        % - the corresponding .lib files
        extraLinkPaths{end + 1} = ...
            fullfile(matlabRoot, 'extern', 'lib', arch, 'microsoft');
    end
    
    % make link paths available for cargo
    exportExtraLinkPaths(extraLinkPaths);
    
    buildWithCargo('zarr_mex', 'zarrMex');
end

function buildWithCargo(oldName, newName)
    prevDir = pwd();
    prevRestore = onCleanup(@() cd(prevDir));
    
    thisDir = fileparts(mfilename('fullpath'));
    cargoDir = fullfile(thisDir, 'rust', oldName);
    
    % build project
    cd(cargoDir);
    %system('cargo clean');
    %system('cargo update');
    
    if ismac
        [~, unameResult] = system('uname -m');
        if strcmp(strtrim(unameResult),'arm64')
            system('cargo build --release --target=aarch64-apple-darwin');
            libDir = fullfile(thisDir, '..', 'target', 'aarch64-apple-darwin', 'release');
        else
            system('cargo build --release --target=x86_64-apple-darwin');
            libDir = fullfile(thisDir, '..', 'target', 'x86_64-apple-darwin', 'release');
        end
    else
        system('cargo build --release');
        libDir = fullfile(thisDir, '..', 'target', 'release');
    end
    
    % rename library
    if ismac
        libPath = fullfile(libDir, strcat('lib', oldName, '.dylib'));
    elseif isunix
        libPath = fullfile(libDir, strcat('lib', oldName, '.so'));
    elseif ispc
        libPath = fullfile(libDir, strcat(oldName, '.dll'));
    else
        error('Platform not supported');
    end
    
    mexPath = fullfile(thisDir, strcat(newName, '.', mexext()));
    copyfile(libPath, mexPath);
    
    cd(prevDir);
end

function exportExtraLinkPaths(paths)
    extraLinkPathsStr = strjoin(paths, ';');
    setenv('EXTRALINKPATHS', extraLinkPathsStr);
end
