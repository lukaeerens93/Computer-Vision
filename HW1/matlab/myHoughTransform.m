function [H, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)
%Your implementation here
%Im - grayscale image - 
%threshold - prevents low gradient magnitude points from being included
%rhoRes - resolution of rhos - scalar
%thetaRes - resolution of theta - scalar


% First treshold the image
% Each pixel above the threshold is a potential point on a line
Im( Im<threshold ) = 0;

% Find dimensions of image and then threshold it
[x,y] = size(Im);   %first one is: [480,640]
% Pythogras sqrt(x^2 + y^2) where x,y are dimensions of the image
disp([x,y]);
disp([x^2, y^2]);
diagonal_through_image = power( x^2 + y^2 , 1/2 );
disp(diagonal_through_image);   % = 800

%parameters
% sigma     = 2;
% threshold = 0.03;
% rhoRes    = 2;
% thetaRes  = pi/90;
% nLines    = 50;
%end of parameters

% Create accumulator array A(m,c)
thetaScale = 0:thetaRes:pi*2;       % Set dimensions (90 degrees)
c = floor(2*pi/thetaRes)+1;
rhoScale = 0:rhoRes:diagonal_through_image;
m = 1 + floor( diagonal_through_image / rhoRes );     % for the first photo: 1+2floor(800/2) = 801


% Set all elements -> A(m,c) = 0 for all m,c
H = zeros(m,c);   
%disp(size(A));

for i = 1:x             % For each image edge (x(i),y(j))
    for j = 1:y     
        for f = 1:c     % For each element in A(m,c)
            %disp(f);
            
            % If (m,c) lies on the on the line: c = x(i)m + y(j)
            if Im(i,j) > 0
                
                % Definition of each edge point
%                 X = j/rhoRes;
%                 Y = i/rhoRes;
                
                % Define the line in parametric form,
                angle = thetaScale(f);
                p =  j * cos(angle) + i * sin(angle);
                if p < 0
                    continue
                end
                p = floor(p/rhoRes)+1;
        
                
                % Redefine the x,y coordinate of the accumulator array
                %row = floor( (m+p) / 2);
%                 row = floor( p + m/2);
%                 collumn = f;                
                %disp([m,p]);
                
                % Increment A(m,c) = A(m,c) + 1
                %disp( [row, collumn] );
                %disp( [row, p] );  % <------------------------------ PUNK!
                H(p,f) = H(p,f) + 1;
            end
        end
    end
end
% imshow(H,[]);   % the bloody [] transforms the image from binary to grayscale
% remove the [] and witness the stupidity of a "feature" of matlab
% To make it less verbose:
end
        
        
