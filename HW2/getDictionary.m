function [dictionary] = getDictionary(imgPaths, alpha, K, method)
% Generate the filter bank and the dictionary of visual words
% Inputs:
%   imgPaths:        array of strings that repesent paths to the images
%   alpha:          num of points
%   K:              K means parameters
%   method:         string 'random' or 'harris'
% Outputs:
%   dictionary:         a length(imgPaths) * K matrix where each column
%                       represents a single visual word
    % -----fill in your implementation here --------
    dictionary = [];
    
    [r,c] = size(imgPaths);
    disp([r,c]);
    
    img_dict = [];
    pnts = [];              % List of Points
    disp(imgPaths(1:50));

    for i = 1:c
        % Read images
        %disp('chowder');
        img_name = string( imgPaths(1,i) ); 
        disp(char( strcat( '../data/', img_name ) ));
        img = imread( char( strcat( '../data/', img_name ) ) );
        %disp('powder');
        % Apply the filter on each of the images
        img_filtered = extractFilterResponses( img, createFilterBank() );

        % Get random points and harris points from the images
        if method == 'random' 
            pnts = getRandomPoints( img, alpha ); 
        end
        if method == 'harris' 
            pnts = getHarrisPoints( img, alpha, K ); 
        end
        
        % Return region of interest, index, append them to the dictionary
        img_size = size(img_filtered);
        indexes = sub2ind( img_size(1:2), pnts(:,1)', pnts(:,2)' );
        binary_function = @plus;
        input_array = ( 0:( img_size(3)-1) ).' * prod( img_size(1:2) );
        indexes = bsxfun( binary_function, indexes, input_array );
        img_dict = ( img_filtered(indexes) )';
        dictionary = [  dictionary ; img_dict];
        %disp('chowder');
    end
    % ------------------------------------------
    
end
