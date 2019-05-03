function [wordMap] = getVisualWords(I, dictionary, filterBank)
% Convert an RGB or grayscale image to a visual words representation, with each
% pixel converted to a single integer label.   
% Inputs:
%   I:              RGB or grayscale image of size H * W * C
%   filterBank:     cell array of matrix filters used to make the visual words.
%                   generated from getFilterBankAndDictionary.m
%   dictionary:     matrix of size 3*length(filterBank) * K representing the
%                   visual words computed by getFilterBankAndDictionary.m
% Outputs:
%   wordMap:        a matrix of size H * W with integer entries between
%                   1 and K

    % -----fill in your implementation here --------
    % Define the size of the filter that will be used
    filter = extractFilterResponses(I, filterBank);
    [x,y,d] = size(filter);

    % Find the distance to nearest word for each pixel
    X_ = reshape(filter, x*y, d);
    dist = pdist2(X_, dictionary, 'euclidean');
    [M, I] = min(dist, [], 2);  

    wordMap = zeros(x, y);
    wordMap = reshape(I, x, y);
    % ------------------------------------------
end
