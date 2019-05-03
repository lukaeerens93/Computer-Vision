function [points] = getRandomPoints(I, alpha)
% Generates random points in the image
% Input:
%   I:                      grayscale image
%   alpha:                  random points
% Output:
%   points:                    point locations
%
	% -----fill in your implementation here --------
    % If there is 3 color channels (ie still in RGB format) convert to
    % grayscale
    if size(I, 3) == 3 
        img = rgb2gray(I);
        [x,y] = size(img);
        xMatrix = randperm(x-1, alpha);
        yMatrix = randperm(y-1, alpha);
        points = [ xMatrix', yMatrix' ];
    end
    % If there is 1 color channel (ie already converted to grayscale)
    if size(I, 3) == 1 
        img = I;
        [x,y] = size(img);
        xMatrix = randperm(x-1, alpha);
        yMatrix = randperm(y-1, alpha);
        points = [ xMatrix', yMatrix' ];
    end
    % ------------------------------------------

end

