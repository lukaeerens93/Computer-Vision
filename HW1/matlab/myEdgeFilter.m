function [Im, Io, Ix, Iy] = myEdgeFilter(img, sigma)
%Your implemention


% Define the specs for the gaussian filter and apply it on the image
size_filter = 2 * ceil( 3*sigma ) + 1;               % Define Gaussian Filter Size
filter = fspecial('gaussian', size_filter, sigma);   % Define the Gaussian Filter
[g_image] = myImageFilter(img, filter);                 % Apply gaussian filter to image

% Define the Sobel Operators and apply them to the convolved image
Sobel_x = [1 2 1; 0 0 0; -1 -2 -1];     % x- direction sobel
Sobel_y = Sobel_x';


% Element by element division of Sobel Convolution now
% found in order to calculate the gradient angle
Iy = myImageFilter( g_image, Sobel_y );      % Apply y-directions sobel convolution
Ix = myImageFilter( g_image, Sobel_x );      % Apply x-directions sobel convolution
val_division = Iy./Ix;
Io = atan(val_division);

% Create the matrix onto which the edges will be projected
% Create the diagonal line that passes through the matrix
Im = power( power(Ix,2) + power(Iy,2) , 1/2);

%imshow(Io);
[m,n] = size(Io);

%disp([m,n]);

Imres = Im;
       % To make this SOB less verbose

% Map gradient angle to closest of 4 cases 0, pi/4, pi/2, 5pi/4 
% be careful because everything is in the other direction in matlab, so 90
% degrees is actually -90, 45 is -45, 135 etc
for i = 2:m-1
    for j = 2:n-1
        check = Io(i,j);
        if check >= -0.125*pi &&   check < 0.125*pi
            if Im(i,j)<Im(i+1,j) || Im(i,j)<Im(i-1,j)
                Imres(i,j) = 0;  
            end
        end 
        
        if check >= pi * 0.125 && check <= pi*0.375
            if Im(i,j)<Im(i+1,j+1)||Im(i,j)<Im(i-1,j-1)
                Imres(i,j) = 0; 
            end
        end
        
        if check >=-0.375*pi && check <= -0.125*pi  
            if Im(i,j)<Im(i-1,j+1)||Im(i,j)<Im(i+1,j-1)
                Imres(i,j) =0;  
            end
        end
        
        if check >= 0.375*pi || check <= -0.375*pi
            if Im(i,j)<Im(i,j+1) || Im(i,j)<Im(i,j-1)
                Imres(i,j)=0;
            end
        end
        
  
    
    end
end
Im = Imres;

end
    
                
        
        
