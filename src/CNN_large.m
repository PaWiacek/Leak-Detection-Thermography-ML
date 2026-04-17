% Image input
clc; clearvars;

% define data directory
dataDir = 'OK_dataset2026_color_train_aug';

% Load image datastore
imds = imageDatastore(dataDir, 'IncludeSubfolders', true, 'LabelSource','foldernames');

% Display label count
disp(countEachLabel(imds));

% Define input size for the network
inputSize = [640 480 3];

% Fixed data split
imdsTrain = imageDatastore(fullfile(dataDir, 'Train'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsValidation = imageDatastore(fullfile(dataDir, 'Validation'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(fullfile(dataDir, 'Test'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Augment & resize data
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augImdsValidation = augmentedImageDatastore(inputSize, imdsValidation);
augImdsTest = augmentedImageDatastore(inputSize, imdsTest);
% Network architecture

% Training options

% instead of training, load pre-trained model
load bestNet_large_conv_BN_ReLU_poolx2_FC.mat

% Test the network
% Get true labels
trueTrainLabels = imdsTrain.Labels;
trueValidationLabels = imdsValidation.Labels;
trueTestLabels = imdsTest.Labels; 

% Get predictions
predTrainLabels = classify(net, augImdsTrain);
predValidationLabels = classify(net, augImdsValidation);
predTestLabels = classify(net, augImdsTest);

% Compute accuracy
trainAccuracy = mean(predTrainLabels == trueTrainLabels);
validationAccuracy = mean(predValidationLabels == trueValidationLabels);
testAccuracy = mean(predTestLabels == trueTestLabels);

% Display results
disp(['Training Accuracy: ' num2str(round(trainAccuracy * 100, 2)) '%']);
disp(['Validation Accuracy: ' num2str(round(validationAccuracy * 100, 2)) '%']);
disp(['Test Accuracy: ' num2str(round(testAccuracy * 100, 2)) '%']);

% Accuracy comparison
figure
set(gcf,'Units','centimeters');
set(gcf,'Position',[0 0 12 8]);

values = [trainAccuracy, validationAccuracy, testAccuracy] * 100;
acc_comp = bar([trainAccuracy, validationAccuracy, testAccuracy] * 100);

ylabel('Accuracy [%]');
ylim([0 100]);

ax = gca;
ax.XTickLabel = {'Train','Validation','Test'};
ax.FontName = 'Times';
ax.FontSize = 20;

% add values above the bars
x = acc_comp.XEndPoints;
y = acc_comp.YEndPoints;
labels = arrayfun(@(v) sprintf('%.2f', v), values, 'UniformOutput', false);
text(x, y, labels, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontName','Times', ...
    'Color','w', ...
    'FontSize',16);


grid on;
exportgraphics(gcf,'acc-comp-largeNet.png');
% Confusion matrix
confMat = confusionmat(trueTestLabels, predTestLabels);
classNames = categories(trueTestLabels);
% Clear any existing chart output
clf;
% Override class display names for readability
customClassNames = ["LEAK" "NO LEAK"]

% Extract true positives, false positives and false negatives
% confMat(row, col)
TP = confMat(1,1);
FN = confMat(1,2);
FP = confMat(2,1);
TN = confMat(2,2);

% Create conf chart
figure
set(gcf,'Units','centimeters');
% set(gcf,'Position',[0 0 12 12]);
set(gcf,'Position',[0 0 16 12]);


% cm = confusionchart(confMat, customClassNames, 'Normalization', 'column-normalized');
cm = confusionchart(confMat, customClassNames);

cm.FontName = 'Times';
cm.FontSize = 20;
cm.Position = [0.2 0.2 0.65 0.65];
exportgraphics(gcf,'confMat-largeNet.png');
% Precision & recall and F1 score
precision = TP / (TP + FP + eps);
recall = TP / (TP + FN + eps);
f1_score = 2 * (precision * recall) / (precision + recall + eps);

% Display results
disp(['Precision: ' num2str(round(precision * 100, 2)) '%']);
disp(['Recall: ' num2str(round(recall * 100, 2)) '%']);
disp(['F1 score: ' num2str(round(f1_score * 100, 2)) '%']);
% Mismatched images
% Extract false negatives
isFalseNegative = (trueTestLabels == 'LEAK') & (predTestLabels == 'NO_LEAK');
falseNegativeFiles = imdsTest.Files(isFalseNegative);
disp('False negative files:');
disp(falseNegativeFiles);

% Extract false positives
isFalsePositive = (trueTestLabels == 'NO_LEAK') & (predTestLabels == 'LEAK');
falsePositiveFiles = imdsTest.Files(isFalsePositive);
disp('False positive files:');
disp(falsePositiveFiles);
