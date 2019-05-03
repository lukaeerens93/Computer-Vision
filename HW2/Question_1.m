% Don't understand if there is suppose to be a script from which we read
% all these functions. So I made this one... Answers all of question 1,
% this is basically the main function of the code. 

% ---------------------- Question 1.2 Start -----------------------------
img = imread('../data/rainforest/sun_agcfctbkefnoasmy.jpg');    % I like parrots
filtered_img = extractFilterResponses(img, createFilterBank());
% Apply all 4 filters on the L channel
imshow(filtered_img(:,:,1),[]);         % Gaussian
pause(2);
imshow(filtered_img(:,:,16),[]);        % Log
pause(2);
imshow(filtered_img(:,:,31),[]);        % dX
pause(2);
imshow(filtered_img(:,:,46),[]);        % dY
pause(2);
% ---------------------- Question 1.2 End -------------------------------



% ---------------------- Question 1.3 Start -----------------------------
imgs{1}='../data/football_stadium/sun_bvmltinzudvwkjyh.jpg';
imgs{2}='../data/campus/sun_aciggnzupbzygsaw.jpg';
imgs{3}='../data/auditorium/sun_aflgfyywvxbpeyxl.jpg';
for i = 1:3
    image = imread(imgs{i});
    % Random
    randomPoints = getRandomPoints(image, 100); % Code is being a dick (randperm is a chod)
    imshow(image);
    hold on;
    plot(harrisPoints(:,2), harrisPoints(:,1), 'r*', 'LineWidth', 3, 'MarkerSize', 3);
    hold off;
    pause(3);
    % Harris
    harrisPoints = getHarrisPoints(image, 500, 0.04);
    imshow(image);
    hold on;
    plot(harrisPoints(:,2), harrisPoints(:,1), 'r*', 'LineWidth', 3, 'MarkerSize', 3);
    hold off;
    pause(3);
end
% ---------------------- Question 1.3 end -------------------------------



% ---------------------- Question 1.4 end -------------------------------
% alpha = 50
% Establish a directory route for all images to be fed into this bloody
% thing
imgDataset = load('../data/traintest.mat');
directory = imgDataset.all_imagenames;

% Random point dictionary
pixelResponses = getDictionary( directory, 50, 0.05,'random');
disp('allahu akbar');
[~, dictionary] = kmeans( pixelResponses, 100, 'EmptyAction', 'drop' );
save('dictionaryRandom.mat','dictionary');

% Harris point dictionary
pixelResponses=getDictionary(directory,50,0.05,'harris');
[~, dictionary] = kmeans(pixelResponses, 100, 'EmptyAction', 'drop');
save('dictionaryHarris.mat','dictionary');


