folder = '1494452415578787519/';
filename = '20.jpg';
RGB = imread(strcat(folder, filename));
%RGB = imread(strcat(workingDir, '20.jpg'));
figure;
imshow(RGB);
hold on
lanes = [1019, 120, 0, 431; 270, 120, 1279, 428; 785, 120, 96, 719; 512, 120, 1201, 719];
height = 720;
width = 1280;
axis([0, 1280, 0, 720]);
plot([1019, 0], [120, 431], 'r--', 'LineWidth', 2);
plot([270, 1279], [120, 428], 'r--', 'LineWidth', 2);
plot([785, 96], [120, 719], 'r--', 'LineWidth', 2);
plot([512, 1201], [120, 719], 'r--', 'LineWidth', 2);
% for i = 1:4
%     plot(lanes(i, 