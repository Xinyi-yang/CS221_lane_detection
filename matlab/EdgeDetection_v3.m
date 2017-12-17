% % Get a list of all files and folders in this folder.
% files = dir();
% % Get a logical vector that tells which is a directory.
% dirFlags = [files.isdir];
% % Extract only those that are directories.
% subFolders = files(dirFlags);
% % Print folder names to command window.
% h=waitbar(0,'Please be patient!');
% for k = 1 : length(subFolders)
%     waitbar(k/length(subFolders));
%     clipBatch = '';
%     folder = strcat(subFolders(k).name, '/');
%     folder = '1494452415578787519/';
%     files = dir(strcat(clipBatch, folder, '*.jpg'));
%     
%     height = 720;
%     width = 1280;
%     clip = zeros(720, 1280, length(files));
%     BWFolder = 'BW_v3_color/';
%     mkdir(strcat(clipBatch, folder), BWFolder);
%     for i = 1:length(files)
%         
%         
%         filename = files(i).name;
%         if filename(1) == 'B'
%             continue;
%         end
%         RGB = imread(strcat(clipBatch, folder, filename));
%         %RGB = imread('1495492644603734252_20.jpg'); %curvy
%         %RGB = imread('1494452385593783358_20.jpg'); %dim
%         %RGB = imread('1495058787529671793_20.jpg');%yellow lane
%         %RGB = imread('1494452391590105691_20.jpg');%Super dim
%         %RGB = imread('1494452415578787519/20.jpg');
%         %RGB = imread(strcat(clipBatch, folder,'20.jpg'));%Last photo of clip
% %         figure;
% %         imshow(RGB)
%         I = rgb2gray(RGB);
%         [height, width] = size(I);
%         lowerI = I(floor(height):height, :);
% %         figure;
% %         imshow(I)
%         %I = imgaussfilt(I);
%         % figure;
%         % imshow(I)
%         % figure;
% 
%         % thresholds = 0.02:0.02:0.1;
%         % for i = 1:5
%         %     threshold = thresholds(i)
%         %     EDGE = edge(I,'sobel',threshold );
%         %     
%         %     subplot(2, 3, i)
%         %     imshow(EDGE)
%         % end
% 
%         %level = graythresh(I)
%         % BW = im2bw(I,0.75);
%         % figure;
%         % imshow(BW)
%         % levels = 0.6:0.03:0.78;
%         % for i = 1:6
%         %     level = levels(i);
%         %     BW = im2bw(I,level);
%         %     
%         %     subplot(2, 3, i)
%         %     imshow(BW)
%         % end
%         % 
%         bestThreshold = 0.06;
%         vertThreshold = 0.02;
%         EDGE_vert = edge(I,'sobel',vertThreshold, 'vertical');
%         EDGE_isotropic = edge(I,'sobel',bestThreshold);
% 
%         EDGE = and(EDGE_vert, EDGE_isotropic);
%         %EDGE = EDGE_isotropic;
% %         figure;
% %         imshow(EDGE)
% 
%         %adaptEdge = not(adaptivethreshold(I, 30, 0.10));
%     %     figure;
%     %     imshow(adaptEdge)
%     %     figure;
%     %     imshow(EDGE_vert)
%     %     figure;
%     %     imshow(EDGE_isotropic)
% 
%         % bestLevel = 0.3
%         bestLevel = min(graythresh(lowerI) * 1.7, 0.8);
%         if bestLevel == 0.8
%             warning = 'Warning! Level too high!'
%         end
% 
%         %adpLevel = adaptthresh(I, 0.4);
%         BW = im2bw(I, bestLevel);
% %         figure;
% %         imshow(BW)
%         %figure;
%         %imshow(or(or(BW, EDGE), adaptEdge))
%         FINAL = or(BW, EDGE);
%         FINAL_COLOR = im2double(RGB).*repmat(double(FINAL),[1,1,3]);
%         
%         clip(:, :, i) = FINAL;
% %         figure;
% %         imshow(FINAL)
%         %imwrite(FINAL_COLOR,strcat(clipBatch, folder, BWFolder, 'BW_', filename));
%         
%         conditionHigh = logical(ones(size(FINAL)));
%         conditionLeft = logical(ones(size(FINAL)));
%         conditionHigh(1:200, :, :) = false;
%         conditionLeft = flipud(tril(conditionLeft));
%         conditionRight = fliplr(conditionLeft);
% %         partialImg = FINAL;
% %         partialImg(not(or(conditionLeft, conditionRight))) = 0;
% %         partialImg(not(conditionHigh)) = 0;
% %         partialCOLOR = im2double(RGB).*repmat(double(partialImg),[1,1,3]);
%         antipartialImg = FINAL;
%         antipartialImg(or(conditionLeft, conditionRight)) = 0;
%         antipartialImg(not(conditionHigh)) = 0;
%         antipartialCOLOR = im2double(RGB).*repmat(double(antipartialImg),[1,1,3]);
%         parFolder = 'par/';
%         antiparFolder = 'antipar/';
%         mkdir(folder, antiparFolder);
%         %imwrite(partialCOLOR,strcat(clipBatch, folder, parFolder, 'BW_', filename));
%         imwrite(antipartialCOLOR,strcat(clipBatch, folder, antiparFolder, 'BW_', filename));
%     end
%     break;
% end


%Legacy code for single clip
clipBatch = '';
folder = '1494452385593783358/';
files = dir(strcat(clipBatch, folder, '*.jpg'));
h=waitbar(0,'Please be patient!');
    height = 720;
    width = 1280;
    clip = zeros(720, 1280, length(files));
    BWFolder = 'BW_v3_color/';
    mkdir(strcat(clipBatch, folder), BWFolder);
    for i = 1:length(files)
        
        waitbar(i/length(files))
        filename = files(i).name;
        if filename(1) == 'B'
            continue;
        end
        RGB = imread(strcat(clipBatch, folder, filename));
        %RGB = imread('1495492644603734252_20.jpg'); %curvy
        %RGB = imread('1494452385593783358_20.jpg'); %dim
        %RGB = imread('1495058787529671793_20.jpg');%yellow lane
        %RGB = imread('1494452391590105691_20.jpg');%Super dim
        %RGB = imread(strcat(clipBatch, folder,'20.jpg'));%Last photo of clip
%         figure;
%         imshow(RGB)
        I = rgb2gray(RGB);
        [height, width] = size(I);
        lowerI = I(floor(height):height, :);
%         figure;
%         imshow(I)
        %I = imgaussfilt(I);
        % figure;
        % imshow(I)
        % figure;

        % thresholds = 0.02:0.02:0.1;
        % for i = 1:5
        %     threshold = thresholds(i)
        %     EDGE = edge(I,'sobel',threshold );
        %     
        %     subplot(2, 3, i)
        %     imshow(EDGE)
        % end

        %level = graythresh(I)
        % BW = im2bw(I,0.75);
        % figure;
        % imshow(BW)
        % levels = 0.6:0.03:0.78;
        % for i = 1:6
        %     level = levels(i);
        %     BW = im2bw(I,level);
        %     
        %     subplot(2, 3, i)
        %     imshow(BW)
        % end
        % 
        bestThreshold = 0.06;
        vertThreshold = 0.02;
        EDGE_vert = edge(I,'sobel',vertThreshold, 'vertical');
        EDGE_isotropic = edge(I,'sobel',bestThreshold);

        EDGE = and(EDGE_vert, EDGE_isotropic);
        %EDGE = EDGE_isotropic;
%         figure;
%         imshow(EDGE)

        %adaptEdge = not(adaptivethreshold(I, 30, 0.10));
    %     figure;
    %     imshow(adaptEdge)
    %     figure;
    %     imshow(EDGE_vert)
    %     figure;
    %     imshow(EDGE_isotropic)

        % bestLevel = 0.3
        bestLevel = min(graythresh(lowerI) * 1.7, 0.8);
        if bestLevel == 0.8
            warning = 'Warning! Level too high!'
        end

        %adpLevel = adaptthresh(I, 0.4);
        BW = im2bw(I, bestLevel);
%         figure;
%         imshow(BW)
        %figure;
        %imshow(or(or(BW, EDGE), adaptEdge))
        FINAL = or(BW, EDGE);
        % cut the top
        conditionHigh = logical(ones(size(FINAL)));
        conditionHigh(1:200, :, :) = false;
        FINAL(not(conditionHigh)) = 0;
        FINAL_COLOR = im2double(RGB).*repmat(double(FINAL),[1,1,3]);
        clip(:, :, i) = FINAL;
%         figure;
%         imshow(FINAL)
        imwrite(FINAL_COLOR,strcat(clipBatch, folder, BWFolder, 'BW_', filename));
        %break;
    end
%     AVE_FINAL = zeros(height, width);
%     for i = 1:length(files)
%         AVE_FINAL = or(AVE_FINAL, clip(:,:,i));
%     end
%     imwrite(AVE_FINAL,strcat(clipBatch, folder, 'BW_overlap.jpg'));