% Get a list of all files and folders in this folder.
files = dir();
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
% Print folder names to command window.
h=waitbar(0,'Please be patient!');

for k = 1 : length(subFolders)
    waitbar(k/length(subFolders));
    folder = strcat(subFolders(k).name, '/');
    workingDir = strcat(subFolders(k).name, '/BW_v3_color/');
    workingDir = '1494452415578787519/antipar/';
    

    imageNames = dir(fullfile(workingDir,'*.jpg'));
    imageNames = {imageNames.name}';
    newImageNames = cell(size(imageNames));
    for ii = 1:length(imageNames)
        if ii == 1
            newImageNames{1} = imageNames{ii};
        elseif ii >= 2 && ii <= 11
            newImageNames{ii + 8} = imageNames{ii};
        elseif ii == 12
            newImageNames{2} = imageNames{ii};
        elseif ii == 13
            newImageNames{20} = imageNames{ii};
        else
            newImageNames{ii - 11} = imageNames{ii};
        end
    end
%     leftVideo = VideoWriter(fullfile(workingDir,'left.avi'));
%     leftVideo.FrameRate = 20;
%     open(leftVideo)
%     rightVideo = VideoWriter(fullfile(workingDir,'right.avi'));
%     rightVideo.FrameRate = 20;
%     open(rightVideo)
%     partialVideo = VideoWriter(fullfile(workingDir,'part.avi'));
%     partialVideo.FrameRate = 20;
%     open(partialVideo)
    outputVideo = VideoWriter(fullfile(workingDir,'out.avi'));
    outputVideo.FrameRate = 20;
    open(outputVideo)
    for ii = 1:length(newImageNames)
       img = imread(fullfile(workingDir,newImageNames{ii}));
       
%        leftImg = img(:,1:end/2,:);
%        rightImg = img(:,end/2+1:end,:);
       for jj = 1:outputVideo.FrameRate / 4
%            writeVideo(leftVideo,leftImg)
%            writeVideo(rightVideo,rightImg)
           writeVideo(outputVideo,img)
       end
    end
    %close(partialVideo)
    close(outputVideo)
    %shuttleAvi = VideoReader(fullfile(workingDir,'shuttle_out.avi'));
    %open(outputVideo)
    break;
end