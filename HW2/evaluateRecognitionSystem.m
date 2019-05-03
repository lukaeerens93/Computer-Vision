% evaluateRecognitionSystem.m
% This script evaluates nearest neighbour recognition system on test images
% load traintest.mat and classify each of the test_imagenames files.
% Report both accuracy and confusion matrix




% Load random train and test structs as well as harris train and test structs
Train_Random = load('Train_Random_Recog.mat');
Test_Random = load('visionRandom.mat');

Train_Harris = load('Train_Harris_Recog.mat');
Test_Harris = load('visionHarris.mat');

%Euclidean = 1-2, Chisq = 3-4
for a = 1:4
    disp(Train_Random);
    if (a == 1 || a == 3)
        % Conduct euclidean distance test on random points based word map
        [i,w] = size(Test_Random{3});
        if (a == 1)
            Dist = getImageDistance(Train_Random{3}, Test_Random{3}, 'euclidean');
        end
        if (a == 3)
            Dist = getImageDistance(Train_Random{3}, Test_Random{3}, 'chisq');
        end
    end
    if (a == 2 || a == 4)
        % Conduct euclidean distance test on harris points based word map
        [i,w] = size(Test_Harris{3});
        if (a == 2)
            Dist = getImageDistance(Train_Harris{3}, Test_Harris{3}, 'euclidean');
        end
        if (a == 4)
            Dist = getImageDistance(Train_Harris{3}, Test_Harris{3}, 'chisq');
        end
    end
    [x,y] = size(Dist);
    IndexMin = find(Dist == min(Dist,[],1));
    [I,J] = ind2sub([x,y], IndexMin);
    if (a == 1)
        map = [(1:x)',(Train_Random{4})'];
    end
    if (a == 2)
        map = [(1:x)',(Train_Harris{4})'];
    end
    LUT(map( :, 1) ) = map(:, 2);
    result  = (LUT(I'))';
    if (a == 1)
        class = cat(2, (1:y)', Test_Random{4}', I, result);
    end
    if (a == 2)
        class = cat(2, (1:y)', Test_Harris{4}', I, result);
    end
    [c, o] = confusionmat( class(:,2), class(:,4) );
    if (a == 1)
        disp('1) Accuracy: euclid random');
        disp('2) Confusion Matrix: euclid random');
    end
    if (a == 2)
        disp('1) Accuracy: euclid harris');
        disp('2) Confusion Matrix: euclid harris');
    end
    if (a == 3)
        disp('1) Accuracy: chisq random');
        disp('2) Confusion Matrix: chisq random');
    end
    if (a == 4)
        disp('1) Accuracy: chisq harris');
        disp('2) Confusion Matrix: chisq harris');
    end
    
end

