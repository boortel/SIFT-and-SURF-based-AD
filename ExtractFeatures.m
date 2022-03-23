%% Cleanup
clc;
clear all;
close all;

%% Set up the path constants
pathTest = "D:\Programovani\Datasets\IndustryBiscuit_KerasApp\test";
pathTrainSemi = "D:\Programovani\Datasets\IndustryBiscuit_KerasApp\trainSemi";
pathTrainOCC = "D:\Programovani\Datasets\IndustryBiscuit_KerasApp\trainOCC";

%% Load the image datasets
imds_test = imageDatastore(pathTest, "IncludeSubfolders", true, ...
    "FileExtensions", ".jpg", "LabelSource", "foldernames");

imds_trainSemi = imageDatastore(pathTrainSemi, "IncludeSubfolders", true, ...
    "FileExtensions", ".jpg", "LabelSource", "foldernames");

imds_trainOCC = imageDatastore(pathTrainOCC, "IncludeSubfolders", true, ...
    "FileExtensions", ".jpg", "LabelSource", "foldernames");

%% Get the features from data
% Get the train features
[~, SIFT_trainSemi] = Get_Features(imds_trainSemi, 'SIFT', 2);
[~, SURF_trainSemi] = Get_Features(imds_trainSemi, 'SURF', 0);

[~, SIFT_trainOCC] = Get_Features(imds_trainOCC, 'SIFT', 2);
[~, SURF_trainOCC] = Get_Features(imds_trainOCC, 'SURF', 0);

labels_trainSemi = imds_trainSemi.Labels;
labels_trainOCC = imds_trainOCC.Labels;

% Convert the labels to the numerical values (for the OCC experiment)
index_ok = labels_trainSemi == 'ok';
index_nok = labels_trainSemi == 'nok';

labels_trainSemiNum = zeros(size(labels_trainSemi));
labels_trainSemiNum(index_ok) = 1;
labels_trainSemiNum(index_nok) = -1;

index_ok = labels_trainOCC == 'ok';
index_nok = labels_trainOCC == 'nok';

labels_trainOCCNum = zeros(size(labels_trainOCC));
labels_trainOCCNum(index_ok) = 1;
labels_trainOCCNum(index_nok) = -1;

% Get the test features
[~, SIFT_test] = Get_Features(imds_test, 'SIFT', 2);
[~, SURF_test] = Get_Features(imds_test, 'SURF', 0);

labels_test = imds_test.Labels;

% Convert the labels to the numerical values (for the OCC experiment)
index_ok = labels_test == 'ok';
index_nok = labels_test == 'nok';

labels_testNum = zeros(size(labels_test));
labels_testNum(index_ok) = 1;
labels_testNum(index_nok) = -1;

%% Semi-supervised learning experiment

% Open the Classification learner and experiment with the models
% Following function contains models with the highest test accuracy

% SIFT - logistic regresion function
[trainedClassifierLR, validationAccuracyLR] = trainLogisticRegression(SIFT_trainSemi, labels_trainSemi);

% SURF - Naive Bayes (Gaussian)
[trainedClassifierNB, validationAccuracyNB] = trainNaiveBayes(SURF_trainSemi, labels_trainSemi);

%% One-class classification experiment

%% One-class SVDD
% please download the https://www.mathworks.com/matlabcentral/fileexchange/69296-support-vector-data-description-svdd
% and add the Svdd directory to the project folder
addpath([pwd, '\Svdd'])
svplot = SvddVisualization();

% Normalize the inputs
SIFT_trainOCC_N = normalize(SIFT_trainOCC);
SURF_trainOCC_N = normalize(SURF_trainOCC);

SIFT_trainSemi_N = normalize(SIFT_trainSemi);
SURF_trainSemi_N = normalize(SURF_trainSemi);

SIFT_test_N = normalize(SIFT_test);
SURF_test_N = normalize(SURF_test);

% OCC - SIFT
kernel = Kernel('type', 'laplacian', 'gamma', 0.05);
cost = 0.1;

svddParameter = struct('cost', cost, 'kernelFunc', kernel, 'KFold', 5);
svdd_SIFT_OCC = BaseSVDD(svddParameter);

% Train and test the SVDD model
svdd_SIFT_OCC.train(SIFT_trainOCC_N, labels_trainOCCNum);
results_SIFT_OCC = svdd_SIFT_OCC.test(SIFT_test_N, labels_testNum);


% OCC - SURF
kernel = Kernel('type', 'laplacian', 'gamma', 0.05);
cost = 0.1;

svddParameter = struct('cost', cost, 'kernelFunc', kernel, 'KFold', 5);
svdd_SURF_OCC = BaseSVDD(svddParameter);

% Train and test the SVDD model
svdd_SURF_OCC.train(SURF_trainOCC_N, labels_trainOCCNum);
results_SURF_OCC = svdd_SURF_OCC.test(SURF_test_N, labels_testNum);


% Semi-supervised - SIFT
kernel = Kernel('type', 'laplacian', 'gamma', 0.05);
cost = 0.1;

svddParameter = struct('cost', cost, 'kernelFunc', kernel, 'KFold', 5);
svdd_SIFT_Semi = BaseSVDD(svddParameter);

% Train and test the SVDD model
svdd_SIFT_Semi.train(SIFT_trainSemi_N, labels_trainSemiNum);
results_SIFT_Semi = svdd_SIFT_Semi.test(SIFT_test_N, labels_testNum);

%svplot.ROC(svdd_SIFT_Semi);


% Semi-supervised - SURF
kernel = Kernel('type', 'laplacian', 'gamma', 0.05);
cost = 0.1;

svddParameter = struct('cost', cost, 'kernelFunc', kernel, 'KFold', 5);
svdd_SURF_Semi = BaseSVDD(svddParameter);

% Train and test the SVDD model
svdd_SURF_Semi.train(SURF_trainSemi_N, labels_trainSemiNum);
results_SURF_Semi = svdd_SURF_Semi.test(SURF_test_N, labels_testNum);

%svplot.ROC(svdd_SURF_Semi);

% Visualise the sample distance to the sphere

% Change the header of the distance function to the:
% function distance(obj, svdd, results, f) and add f to the first line
% instead of figure
figure(1)
subplot(2, 2, 1)
svplot.distance(svdd_SIFT_OCC, results_SIFT_OCC, subplot(2, 2, 1));
title('OC-SVDD SIFT')
subplot(2, 2, 2)
svplot.distance(svdd_SURF_OCC, results_SURF_OCC, subplot(2, 2, 2));
title('OC-SVDD SURF')
subplot(2, 2, 3)
svplot.distance(svdd_SIFT_Semi, results_SIFT_Semi, subplot(2, 2, 3));
title('Semi-supervised SVDD SIFT')
subplot(2, 2, 4)
svplot.distance(svdd_SURF_Semi, results_SURF_Semi, subplot(2, 2, 4));
title('Semi-supervised SVDD SURF')

sgtitle('SVDD sphere radius and samples distances') 

%% One-class SVM

% OCC - SIFT
[SVMModel1, score1, acc1_max, boundary1] = trainSVM(SIFT_trainOCC, labels_trainOCCNum, SIFT_test, labels_testNum, 0.04, 1);

% OCC - SURF
[SVMModel2, score2, acc2_max, boundary2] = trainSVM(SURF_trainOCC, labels_trainOCCNum, SURF_test, labels_testNum, 0.04, 1);

% Semi-supervised - SIFT
[SVMModel3, score3, acc3_max, boundary3] = trainSVM(SIFT_trainSemi, labels_trainSemiNum, SIFT_test, labels_testNum, 0, 0);

% Semi-supervised - SURF
[SVMModel4, score4, acc4_max, boundary4] = trainSVM(SURF_trainSemi, labels_trainSemiNum, SURF_test, labels_testNum, 0, 0);

% Plot the scores
figure(2)
subplot(2, 2, 1)
plot(score1)
hold on
plot(boundary1 * ones(size(SIFT_test)))
axis([0 400 -1.5 3])
legend({'Classification score', 'Decision boundary'})
xlabel('Samples')
ylabel('Score')
title('OC-SVDD SIFT')

subplot(2, 2, 2)
plot(score2)
hold on
plot(boundary2 * ones(size(SURF_test)))
axis([0 400 -1.5 3])
legend({'Classification score', 'Decision boundary'})
xlabel('Samples')
ylabel('Score')
title('OC-SVDD SURF')

subplot(2, 2, 3)
plot(score3)
hold on
plot(boundary3 * ones(size(SIFT_test)))
axis([0 400 -1.5 3])
legend({'Classification score', 'Decision boundary'})
xlabel('Samples')
ylabel('Score')
title('Semi-supervised SVDD SIFT')

subplot(2, 2, 4)
plot(score4)
hold on
plot(boundary4 * ones(size(SURF_test)))
axis([0 400 -1.5 3])
legend({'Classification score', 'Decision boundary'})
xlabel('Samples')
ylabel('Score')
title('Semi-supervised SVDD SURF')

sgtitle('SVM classification scores')

%% Get the ROC curves

% SVDD
figure(3)

subplot(2, 2, 1)
[X, Y, ~, AUC] = perfcurve(labels_testNum, results_SIFT_OCC.predictedLabel, 1);
plot(X, Y)
axis([0 1 0 1])
xlabel('True positive rate')
ylabel('False positive rate')
title(['OC-SVDD SIFT, AUC: ', num2str(AUC)])

subplot(2, 2, 2)
[X, Y, ~, AUC] = perfcurve(labels_testNum, results_SURF_OCC.predictedLabel, 1);
plot(X, Y)
axis([0 1 0 1])
xlabel('True positive rate')
ylabel('False positive rate')
title(['OC-SVDD SURF, AUC: ', num2str(AUC)])

subplot(2, 2, 3)
[X, Y, ~, AUC] = perfcurve(labels_testNum, results_SIFT_Semi.predictedLabel, 1);
plot(X, Y)
axis([0 1 0 1])
xlabel('True positive rate')
ylabel('False positive rate')
title(['Semi-supervised SVDD SIFT, AUC: ', num2str(AUC)])

subplot(2, 2, 4)
[X, Y, ~, AUC] = perfcurve(labels_testNum, results_SURF_Semi.predictedLabel, 1);
plot(X, Y)
axis([0 1 0 1])
xlabel('True positive rate')
ylabel('False positive rate')
title(['Semi-supervised SVDD SURF, AUC: ', num2str(AUC)])

sgtitle('SVDD ROC curves and AUCs')

figure(4)

subplot(2, 2, 1)
[X, Y, ~, AUC] = perfcurve(labels_testNum, score1, 1);
plot(X, Y)
axis([0 1 0 1])
xlabel('True positive rate')
ylabel('False positive rate')
title(['OC-SVM SIFT, AUC: ', num2str(AUC)])

subplot(2, 2, 2)
[X, Y, ~, AUC] = perfcurve(labels_testNum, score2, 1);
plot(X, Y)
axis([0 1 0 1])
xlabel('True positive rate')
ylabel('False positive rate')
title(['OC-SVM SURF, AUC: ', num2str(AUC)])

subplot(2, 2, 3)
[X, Y, ~, AUC] = perfcurve(labels_testNum, score3, 1);
plot(X, Y)
axis([0 1 0 1])
xlabel('True positive rate')
ylabel('False positive rate')
title(['Semi-supervised SVM SIFT, AUC: ', num2str(AUC)])

subplot(2, 2, 4)
[X, Y, ~, AUC] = perfcurve(labels_testNum, score4, 1);
plot(X, Y)
axis([0 1 0 1])
xlabel('True positive rate')
ylabel('False positive rate')
title(['Semi-supervised SVM SURF, AUC: ', num2str(AUC)])

sgtitle('SVM ROC curves and AUCs')