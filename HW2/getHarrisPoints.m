function [points] = getHarrisPoints(I, alpha, k)
% Finds the corner points in an image using the Harris Corner detection algorithm
% Input:
%   I:                      grayscale image
%   alpha:                  number of points
%   k:                      Harris parameter
% Output:
%   points:                    point locations
%
    % -----fill in your implementation here --------
  
    % If there is 3 color channels (ie still in RGB format) convert to
    % grayscale
    if size(I, 3) == 3 
        img = rgb2gray(I);
    end
    % If there is 1 color channel (ie already converted to grayscale)
    if size(I, 3) == 1 
        img = I;
    end

    % Compute image gradient (x, y direction) and subtract the mean
    [gradient_X, gradient_y] = imgradientxy(img);
    gradient_y = gradient_y - mean(gradient_y);
    gradient_X = gradient_X - mean(gradient_X);
    
    % Define each element that goes into the covariange matrix
    Ixx = zeros( size(gradient_X) );
    Iyy = zeros( size(gradient_X) );
    Iyx = zeros( size(gradient_X) );
    Ixy = zeros( size(gradient_X) );
    
    % Create vectorization matrix inputs
    Ixx = conv2(gradient_X .* gradient_X, [1,1,1; 1,1,1; 1,1,1], "Same");
    Iyy = conv2(gradient_y .* gradient_y, [1,1,1; 1,1,1; 1,1,1], "Same"); 
    Iyx = conv2(gradient_y .* gradient_X, [1,1,1; 1,1,1; 1,1,1], "Same");
    Ixy = conv2(gradient_X .* gradient_y, [1,1,1; 1,1,1; 1,1,1], "Same"); 
    
    % Compute Vectorization matrix relevant metrics
    trace_H = Ixx + Iyy;
    determinant_H = Ixx .* Iyy - Iyx .* Ixy;
    
    R = determinant_H - k * trace_H .* trace_H;
    % Find the largest values, so sort in descending order and then pick
    % the biggest alphas (from 1 to alpha)
    [s, sortIndex] = sort( R(:), 'descend' );    
    [x2, y2] = ind2sub( size(R), sortIndex( 1:alpha ) );
    points = [x2,y2];   
    % ------------------------------------------

end
