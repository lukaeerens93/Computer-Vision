function [h] = getImageFeatures(wordMap, dictionarySize)
% Convert an wordMap to its feature vector. In this case, it is a histogram
% of the visual words
% Input:
%   wordMap:            an H * W matrix with integer values between 1 and K
%   dictionarySize:     the total number of words in the dictionary, K
% Outputs:
%   h:                  the feature vector for this image


	% -----fill in your implementation here --------
    histo = histogram(wordMap, dictionarySize);
    h_binCount = histo.BinCounts;
    Normalise = sum(h_binCount);
    h = h_binCount / Normalise;
    % ------------------------------------------

end
