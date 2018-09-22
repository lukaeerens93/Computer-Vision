function [rhos, thetas] = myHoughLines(H, nLines)
%Your implemention here

% - YOU NEED TO MODIFY THIS CODE ----------------------
count=1;                % The number of maximums
vector=[0,0,0];         % Contains all of detected maximums
[m,n]=size(H);
Output = zeros(m,n);

% For every cell in the accumulator corresponding to a real line 
% (likely to be a locally maximal value), there will probably be 
% a number of cells in the  neighborhood that also scored highly 
% but shouldn?t be selected. These non maximal neighbors can be 
% removed using non maximal suppression. 

%The first step is to find the local maximum.
for i = 1:m       % You are scanning a 3x3 kernel along the Hough transform accumulator
    for j = 1:n   % in order to sample all pixels around each pixel
        % Handle edge and corner cases
        if i == 1 
            if j == 1   % if in top left corner
                %only consider right, lower right, lower pixels
                max_gandu = max( H(2,1), H(2, 2) );
                max_arrey_bai = max( H(1,2), max_gandu );
                if H(i,j) > max_arrey_bai           % if center pixel is bigger than others, arrei bai
                    count = count + 1;
                    Output(i,j) = H(i,j);
                    vector = vertcat( vector, [H(i,j) , i , j] );
                end
            end
            if j == n   % if in the bottom left corner
                %only consider right, upper right, top pixels
                max_gandu = max( H(1,n-1), H(2, n) );
                max_arrey_bai = max( H(2,n-1), max_gandu );
                if H(i,j) > max_arrey_bai           % if center pixel is bigger than others, arrei bai
                    count = count + 1;
                    Output(i,j) = H(i,j);
                    vector = vertcat( vector, [H(i,j) , i , j] );
                end
            end
            if j ~= 1 && j ~= n  % if neither
                %only consider right, upper right, top pixels, lower, lower left
                max_gandu_1 = max( H(2,j), H(2, j-1) );
                max_gandu_2 = max( H(1,j-1), H(2, j+1) );
                max_gandu_1 = max( H(1,j), max_gandu_1 );
                max_arrey_bai = max( max_gandu_1, max_gandu_2 );
                if H(i,j) > max_arrey_bai           % if center pixel is bigger than others, arrei bai
                    count = count + 1;
                    Output(i,j) = H(i,j);
                    vector = vertcat( vector, [H(i,j) , i , j] ); 
                end
            end
        end
        if i == m
            if j == 1   % if in top right corner
                %only consider right, lower right, lower pixels
                max_gandu = max( H(m-1,1), H(m-1, 2) );
                max_arrey_bai = max( H(m,2), max_gandu );
                if H(i,j) > max_arrey_bai           % if center pixel is bigger than others, arrei bai
                    count = count + 1;
                    Output(i,j) = H(i,j);
                    vector = vertcat( vector, [H(i,j) , i , j] );
                end
            end
            if j == n   % if in the bottom right corner
                %only consider right, upper right, top pixels
                max_gandu = max( H(m-1,n-1), H(m-1, n) );
                max_arrey_bai = max( H(m,n-1), max_gandu );
                if H(i,j) > max_arrey_bai           % if center pixel is bigger than others, arrei bai
                    count = count + 1;
                    Output(i,j) = H(i,j);
                    vector = vertcat( vector, [H(i,j) , i , j] );
                end
            end
            if j ~= 1 && j ~= n  % if neither
                %only consider left, upper left, top pixels, down, lower left
                disp([m,j-1,j+1]);
                max_gandu_1 = max( H(m,j-1), H(m-1, j-1) );
                max_gandu_2 = max( H(m-1,j), H(m-1, j+1) );
                max_gandu_1 = max( H(m,j+1), max_gandu_1 );
                max_arrey_bai = max( max_gandu_1, max_gandu_2 );
                if H(i,j) > max_arrey_bai           % if center pixel is bigger than others, arrei bai
                    count = count + 1;
                    Output(i,j) = H(i,j);
                    vector = vertcat( vector, [H(i,j) , i , j] );
                end
            end
        end
        if j == 1 
            if i ~= 1 && i ~= m      % if top row
                %only consider right, lower right, lower, lower left, left
                max_gandu_1 = max( H(i-1,1), H(i+1, 1) );
                max_gandu_2 = max( H(i-1,2), H(i+1, 2) );
                max_gandu_2 = max( H(i,2), max_gandu_2 );
                max_arrey_bai = max( max_gandu_1, max_gandu_2 );
                if H(i,j) > max_arrey_bai
                    count = count + 1;
                    Output(i,j) = H(i,j);
                    vector = vertcat( vector, [H(i,j) , i , j] ); 
                end
            end
        end
        if j == n 
            if i ~= 1 && i ~= m      % if top row
                %only consider right, lower right, lower, lower left, left
                max_gandu_1 = max( H(i-1,n), H(i+1, n) );
                max_gandu_2 = max( H(i-1,n-1), H(i+1, n-1) );
                max_gandu_2 = max( H(i,n-1), max_gandu_2 );
                max_arrey_bai = max( max_gandu_1, max_gandu_2 );
                if H(i,j) > max_arrey_bai
                    count = count + 1;
                    Output(i,j) = H(i,j);
                    vector = vertcat( vector, [H(i,j) , i , j] );
                end
            end
        end
        % Everywhere else but the borders of the image
        if i ~= 1 && i ~= m && j ~= 1 && j ~= n
            max_gandu_1 = max([H(i-1,j-1),H(i-1,j),H(i-1,j+1),H(i,j-1),H(i,j+1),H(i+1,j-1),H(i+1,j),H(i+1,j+1)]);
            max_gandu_2 = max( H(i+1,j-1), H(i+1,j) );
            max_gandu_3 = max(  H(i+1,j-1), H(i+1,j));
            max_gandu_4 = max( H(i-1,j+1), H(i-1,j) );
            max_arrey_bai_1 = max( max_gandu_1 , max_gandu_4);
            max_arrey_bai_2 = max( max_gandu_2 , max_gandu_3);
            max_arrey_bai_3 = max( max_arrey_bai_1 , max_arrey_bai_2);
            
            if H(i,j) > max_gandu_1
                count = count + 1;
                Output(i,j) = H(i,j);
                vector = vertcat( vector, [H(i,j) , i ,j] );
            end
        end
        %max1=max( H(i+1,j), H(i-1,j) );
        %max2=max( H(i,j+1), H(i,j-1) );
        %max3=max( H(i+1,j+1), H(i-1,j-1) );
        %max4=max( H(i+1,j-1), H(i-1,j+1) );
        %if H(i,j) > max(max(max1,max2),max(max3,max4))
        %    count = count + 1;
        %    Output(i,j) = H(i,j);
        %    vector = vertcat( vector, [H(i,j) , i ,j] );
        %end
    end
end
% Sort maximum found and avoid 0 vectors
Sorted = sortrows(vector, 1 , 'descend');
mpty = min(nLines, count-1);

% Return rhos and thetas
rhos = Sorted(1:mpty, 2);
thetas = Sorted(1:mpty, 3);
end
        