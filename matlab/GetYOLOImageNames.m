prefix = '/Users/taojia/Google_Drive/Course/CS/CS221/Project/Code/darknet/VOCdevkit/VOC2012/JEPGImages'
source_folder = './0601/';
%target_folder = 'YOLO_Label';
% Get a list of all files and folders in this folder.
files = dir(source_folder);
% Print folder names to command window.

fid = fopen( 'laneImgNames.txt', 'wt' );

for k = 3 : length(files)
    output_string = strcat(prefix, files(k).name, '\n');
    fprintf(fid, output_string);
    %fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
end


fclose(fid);