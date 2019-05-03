% I decided to create a matlab file for each question
% One for question 1, one for question 2, and one for question 3.

% ---------------------- Question 2.1 Start -----------------------------
% Load the random point and harris point dictionaries
load('dictionaryRandom.mat','dictionary');
dict_R = dictionary;
load('dictionaryHarris.mat','dictionary');
dict_H = dictionary;

% Class 1: Campus
imgs{1}='../data/campus/sun_abwbqwhaqxskvyat.jpg';
imgs{2}='../data/campus/sun_blymjduhjifmtvkm.jpg';
imgs{3}='../data/campus/sun_dylabfyfkiigefwx.jpg';
% Class 2: Landscape
imgs{4}='../data/landscape/sun_bgsxrmctuubdruhb.jpg';
imgs{5}='../data/landscape/sun_brifhdkmssmnyxqp.jpg';
imgs{6}='../data/landscape/sun_bzmneytvytyhgant.jpg';

for i = 1:6
    image = imread(imgs{i});
    imshow(image);
    %pause(2);
    % Random Point Dictionary
    image = getVisualWords( image, dict_R, createFilterBank() );
    lbl_to_rgb = label2rgb(image);
    imshow(lbl_to_rgb);
    %pause(2);
    % Harris Point Dictionary
    image = getVisualWords( image, dict_H, createFilterBank() );
    lbl_to_rgb = label2rgb(image);
    imshow(lbl_to_rgb);
    %pause(2);
end
% ---------------------- Question 2.1 End -------------------------------



% ---------------------- Question 2.2 Start -----------------------------
for i = 1:6
    % Delete the .jpg portion of the string, as is not saved this
    % way in the the dictionary. Replace .jpg with .mat
    address = imgs{i}(1:end-4);
    address1 = char( strcat( address, '.mat' ) );
    address2 = load(address1);
    %batchToVisualWords(2);
    [h] = getImageFeatures( address2.wordMap, 100 );
    pause(2);
    disp('tkane');
end
% ---------------------- Question 2.2 End -------------------------------

