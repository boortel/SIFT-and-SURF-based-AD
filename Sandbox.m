clc;
clear all;
close all;

%% Load and show the cookie data
okDataFile = dir('CookieData\ok\*.jpg');      
nokDataFile = dir('CookieData\nok\*.jpg');

sample_OK1 = rgb2gray(imread([okDataFile(1).folder, '\', okDataFile(1).name]));
sample_OK1_c = imread([okDataFile(1).folder, '\', okDataFile(1).name]);
sample_OK2 = rgb2gray(imread([okDataFile(3).folder, '\', okDataFile(3).name]));
%sample_OK2 = imresize(imrotate(sample_OK1,-20),1.2);

sample_SH1 = rgb2gray(imread([nokDataFile(1).folder, '\', nokDataFile(1).name]));
sample_SH1_c = imread([nokDataFile(1).folder, '\', nokDataFile(1).name]);
sample_CD1 = rgb2gray(imread([nokDataFile(28).folder, '\', nokDataFile(28).name]));
sample_CD1_c = imread([nokDataFile(28).folder, '\', nokDataFile(28).name]);
sample_SO1 = rgb2gray(imread([nokDataFile(38).folder, '\', nokDataFile(38).name]));
sample_SO1_c = imread([nokDataFile(38).folder, '\', nokDataFile(38).name]);


%% Get the SIFT features
disp('.........................');
disp('SIFT Features: ');
tic
SIFTpoints_OK1 = detectSIFTFeatures(sample_OK1);
SIFTpoints_OK2 = detectSIFTFeatures(sample_OK2);

SIFTpoints_SH1 = detectSIFTFeatures(sample_SH1);
SIFTpoints_CD1 = detectSIFTFeatures(sample_CD1);
SIFTpoints_SO1 = detectSIFTFeatures(sample_SO1);
toc

figure(1)
hold on
title('SURF features')
subplot(2, 2, 1)
imshow(sample_OK1_c)
hold on
plot(SIFTpoints_OK1.selectStrongest(5))

subplot(2, 2, 2)
imshow(sample_SH1_c)
hold on
plot(SIFTpoints_SH1.selectStrongest(5))

subplot(2, 2, 3)
imshow(sample_CD1_c)
hold on
plot(SIFTpoints_CD1.selectStrongest(5))

subplot(2, 2, 4)
imshow(sample_SO1_c)
hold on
plot(SIFTpoints_SO1.selectStrongest(5))

%% Get the SURF features
disp('.........................');
disp('SURF Features: ');
tic
SURFpoints_OK1 = detectSURFFeatures(sample_OK1);
SURFpoints_SH1 = detectSURFFeatures(sample_SH1);
SURFpoints_CD1 = detectSURFFeatures(sample_CD1);
SURFpoints_SO1 = detectSURFFeatures(sample_SO1);
toc

figure(2)
subplot(2, 2, 1)
imshow(sample_OK1_c)
hold on
plot(SURFpoints_OK1.selectStrongest(5))

subplot(2, 2, 2)
imshow(sample_SH1_c)
hold on
plot(SURFpoints_SH1.selectStrongest(5))

subplot(2, 2, 3)
imshow(sample_CD1_c)
hold on
plot(SURFpoints_CD1.selectStrongest(5))

subplot(2, 2, 4)
imshow(sample_SO1_c)
hold on
plot(SURFpoints_SO1.selectStrongest(5))

%% Get the Harris features
disp('.........................');
disp('Harris Features: ');
tic
HARRpoints_OK1 = detectHarrisFeatures(sample_OK1);
HARRpoints_SH1 = detectHarrisFeatures(sample_SH1);
HARRpoints_CD1 = detectHarrisFeatures(sample_CD1);
HARRpoints_SO1 = detectHarrisFeatures(sample_SO1);
toc

figure(3)
subplot(2, 2, 1)
imshow(sample_OK1)
hold on
plot(HARRpoints_OK1.selectStrongest(10))

subplot(2, 2, 2)
imshow(sample_SH1)
hold on
plot(HARRpoints_SH1.selectStrongest(10))

subplot(2, 2, 3)
imshow(sample_CD1)
hold on
plot(HARRpoints_CD1.selectStrongest(10))

subplot(2, 2, 4)
imshow(sample_SO1)
hold on
plot(HARRpoints_SO1.selectStrongest(10))

%% Match features
I1 = sample_OK1;
I2 = sample_OK2;

points1 = SIFTpoints_OK1;
points2 = SIFTpoints_OK2;

[features1,valid_points1] = extractFeatures(I1, points1);
[features2,valid_points2] = extractFeatures(I2, points2);

% Match the features
indexPairs = matchFeatures(features1,features2);

% Retrieve the locations of the corresponding points for each image
matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

% Visualize the corresponding points. You can see the effect of translation between the two images despite several erroneous matches.
figure(4); 
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);