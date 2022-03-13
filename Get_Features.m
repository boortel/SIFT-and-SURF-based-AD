function [featuresOut, featuresOut2] = Get_Features(input_Datastore, method, sigma)

    % Initialize the arrays with results
    size_ds = size(input_Datastore.Files, 1);
    featuresOut = zeros(size_ds, 25);
    featuresOut2 = zeros(size_ds, 10);
    
    for i = 1:size_ds
        img = rgb2gray(readimage(input_Datastore, i));
        
        % Get SIFT features and its descriptors
        switch method
            case 'SIFT'
                points = detectSIFTFeatures(img, 'Sigma', sigma);
            case 'SURF'
                points = detectSURFFeatures(img);
            otherwise
                error('Unknown method: %s', method)
        end

%         % Code for the point features extraction - no longer used
%         [features, ~] = extractFeatures(img, points);
%         
%         % Get 5 strongest points and its features
%         [~, index] = maxk(points.Scale, 5);
%         features = features(index, :)';
% 
%         % Perform PCA, vectorize and save the feature vector
%         features_pca = pca(features);
%         featuresOut(i, :) = features_pca(:);

        % Get the point features
        try
            points = points.selectStrongest(5);
            featuresOut2(i, 1:5) = points.Scale;
            featuresOut2(i, 6:10) = points.Metric;
        catch
            diff = 5 - points.Count;
            featuresOut2(i, 1:5) = [points.Scale; zeros(diff, 1)];
            featuresOut2(i, 6:10) = [points.Metric; zeros(diff, 1)];
            disp('Warning - not enough feature points!');
        end
    end
end

