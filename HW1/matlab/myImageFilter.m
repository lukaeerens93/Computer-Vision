function [img1] = myImageFilter(img0, h)
%Question 3.1 is here, img0 is the image, h is the filter matrix

% Determine size of image
[x,y] = size(img0);

% Rotate the filter 90 degrees to configure it for convolution
h = rot90(h, 2);

% Determine size of convolution matrix
[w,z] = size(h);
%disp("Filter Size: ");
%disp([w,z]);

% --- Now here is where the padding takes place ----
% My strategy for the padding involves the following process
% 1)    Place original image in the middle of an empty padded matrix,
%       so that is is bordered by empty cells
% 2)    Make matrix top border row = img top border row
% 3)    Do the same for right, bottom, left
% 4)    You will then be left with the corners, just make them equal to the
%       corner value for the original image. (Who cares)
%       
% Start by initializing this padded matrix and the matrix that will contain
% the padded convolution output with 0s everywhere. They are the same size
% obviously
Pad_Mat = zeros( x+(w-1) , y+(z-1) );

%disp("Padded Matrix Size");
%disp([x+(w-1), y+(z-1)]);

% Do step 1)
Pad_Mat( 1+(w-1)/2 : x+(w-1)/2 , 1+(z-1)/2 : y+(z-1)/2 ) = img0;

% Do step 2)
% Pad_Mat( 1 : (w-1)/2 , 1+(z-1)/2 : y+(z-1)/2 ) = repmat( img0(1, 1:y) , (w-1)/2,1 );

% Do step 3) 
% Pad_Mat( 1+(w-1)/2 : x+(w-1)/2 , y+(z-1)/2+1 : y+z-1 ) = repmat( img0(1:x, y) , 1 , (z-1)/2 );     % (right)
% Pad_Mat( x+(w-1)/2+1 : x+(w-1) , 1+(z-1)/2 : y+(z-1)/2 ) = repmat( img0(x, 1:y) , (w-1)/2, 1 );    % (bottom)
% Pad_Mat( 1+(w-1)/2 : x+(w-1)/2 , 1 : (z-1)/2 ) = repmat( img0(1:x, 1) , 1 , (z-1)/2 );              % (left)

% Do step 4) 
% Pad_Mat( 1 : (w-1)/2 , y+(z-1)/2+1 : y+z-1 ) = img0(1,y);            % (upper right)
% Pad_Mat( 1+(w-1)/2+x : x+w-1 , y+(z-1)/2+1 : y+z-1 ) = img0(x,y);    % (lower right)
% Pad_Mat( 1+(w-1)/2+x : x+w-1 , 1 : (z-1)/2) = img0(x,1);             % (lower left)
% Pad_Mat( 1 : (w-1)/2 , 1 : (z-1)/2 ) = img0(1,1);                    % (upper left)


% Now that padding is done, convolve the matrix that contains
% the original image with the padded border in Pad_Mat
% half_k_width = (w-1)/2;
% half_k_height = (z-1)/2;
% a = 1 + half_k_width : x - half_k_width;
% b = 1 + half_k_height : y - half_k_height;

%disp( "Padded Matrix convolution start and end" );
% disp( [ [1+half_k_width, 1+half_k_height], [x-half_k_width , y-half_k_height] ] );
cols = im2col(Pad_Mat,size(h),'sliding');
k = h(:);

covcol = k'*cols;
img1 = reshape(covcol,x,y);

%disp('A');
%imshow(img0);
%kk = conv2(img0, h);
%disp('B');
%imshow(kk);
%disp('C');

imshow(img1);
end
