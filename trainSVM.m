function [SVMModel, score, acc_max, boundary] = trainSVM(datasetTrain, labelsTrain, datasetTest, labelsTest, nu, OCC)
    % Train an SVM model and returns its score, maximal accuracy and
    % the optimal decission boundary

    if OCC
        SVMModel = fitcsvm(datasetTrain, labelsTrain,'Standardize',true,'KernelFunction','gaussian',...
        'KernelScale','auto', 'Nu', nu);
    else
        SVMModel = fitcsvm(datasetTrain, labelsTrain,'Standardize',true,'KernelFunction','gaussian',...
        'KernelScale','auto');
    end

    [label, score] = predict(SVMModel, datasetTest);
    
    acc = zeros(200, 1);

    if size(score, 2) == 2
        score = score(:, 2);
    end

    for i = 1:200
        label(score <= (-1 + i/100)) = -1;
        label(score >= (-1 + i/100)) = 1;
        acc(i) = size(find(label == labelsTest), 1)/size(labelsTest, 1);
    end
    
    [acc_max, index] = max(acc);
    boundary = (-1 + index/100);
end