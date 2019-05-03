% buildRecognitionSystem.m
% This script loads the visual word dictionary (in dictionaryRandom.mat or dictionaryHarris.mat) and processes
% the training images so as to build the recognition system. The result is
% stored in visionRandom.mat and visionHarris.mat.

for i = 1:2

    % Process random in batch across all images and output word count
    %batchToVisualWords(2);
    load('dictionaryRandom.mat','dictionary');
    
    % Process harris in batch across all images and output word count
    %batchToVisualWords(2);
    %load('dictionaryHarris.mat','dictionary');
    
    imgs = load('../data/traintest.mat');
    
    if (i == 1)
        img_count = size(imgs.train_imagenames);
        trainFeatures=[];
    end
    if (i == 2)
        img_count = size(imgs.test_imagenames);
        testFeatures=[];
    end
    img_count = img_count(2);
    set(0,'DefaultFigureVisible','off');    % Don't show histograms
    
    for j = 1:img_count
        if (i == 1)
            Eemage = extractBefore(imgs.train_imagenames(j), '.jpg' );
            t =string('../data/') + Eemage + string('.mat');
            disp(t);
        end
        if (i == 2)
            Eemage = extractBefore(imgs.test_imagenames(j), '.jpg');
            t = string('../data/') + Eemage + string('.mat');
            disp(t);
        end
        words = load(t);
        [h]=getImageFeatures(words.wordMap, 100);
        
        if (i == 1)
            trainFeatures = [trainFeatures; h];
        end
        if (i == 2)
            testFeatures = [testFeatures; h];
        end
        disp('Percent Complete: ' + string( (j/img_count)*100) );
    end
    
    if (i == 1)
        trainLabels = imgs.train_labels;
        train_random = {dictionary,filter,trainFeatures,trainLabels};
        %train_harris = {dictionary,filter,trainFeatures,trainLabels};
    end

    if (i == 2)
        % Testing Labels and Create Struct with everything
        testLabels = imgs.test_labels;
        test_random = {dictionary,filter,testFeatures,testLabels};
        %test_harris = {dictionary,filter,testFeatures,testLabels};
    end

end

% Save Variables
save('Train_Random_Recog.mat', 'train_random');
save('Test_Random_Recog.mat', 'test_random');
%save('Train_Harris_Recog.mat', 'train_harris');
%save('Test_Harris_Recog.mat', 'test_harris');