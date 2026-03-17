%% =========================================================================
%  TUBERCULOSIS DETECTION FROM CHEST X-RAY IMAGES USING DEEP LEARNING
%  =========================================================================
%  Student ID    : 3004795
%  Module        : CN7023 - Artificial Intelligence & Machine Vision
%  Module Leader : Dr Shaheen Khatoon
%  Date          : March 2026
%
%  For this coursework I chose to work on TB detection from chest X-rays
%  because tuberculosis is still a big problem worldwide and catching it
%  early from X-ray images can really help. I picked Option 3 (Image
%  Processing with Deep Learning) from the brief.
%
%  What I did:
%    Experiment 1 - I built a custom CNN from scratch (a shallow network)
%    Experiment 2 - I used GoogLeNet with transfer learning (inception modules)
%    Experiment 3 - I used ResNet-18 with transfer learning (skip connections)
%
%  The reason I tried three different models is to compare a simple network
%  against two pretrained ones and see which one works best for this
%  medical imaging task. I wanted to understand if transfer learning
%  actually makes a difference compared to training from scratch.
%  =========================================================================

%% Housekeeping
close all;
clear;
clc;

% All models and figures save to the same folder as this script
savePath = fileparts(which(mfilename));
if isempty(savePath), savePath = pwd; end

%% =========================================================================
%  PART 1: LOADING THE DATASET
%  =========================================================================
%  I found a TB chest X-ray dataset on Kaggle with 4,200 images - 3,500
%  Normal and 700 Tuberculosis. The dataset is imbalanced (many more
%  Normal images than TB), which is actually realistic because in real
%  life most X-rays are normal. This makes the task harder, so data
%  augmentation and proper evaluation metrics like F1-score become even
%  more important. The images are organised in two folders so
%  imageDatastore can pick up the labels automatically from the folder names.
%  Source: kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
%  =========================================================================

datasetPath = 'CourseWork/TB_Chest_Radiography_Database';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

totalImages = numel(imds.Files);
fprintf('Total images loaded: %d\n', totalImages);

labelCount = countEachLabel(imds);
disp('Number of images per class:');
disp(labelCount);

%% I plotted a bar chart to see the class imbalance
fprintf('\n--- Figure 1: Class Distribution ---\n\n');
figure('Position', [100, 100, 600, 450]);
b = bar(labelCount.Label, labelCount.Count, 'EdgeColor', 'k', 'FaceColor', 'flat');
b.CData(1,:) = [0.2 0.7 0.4];
b.CData(2,:) = [0.85 0.25 0.25];
title('Dataset Class Distribution', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Class', 'FontSize', 13);
ylabel('Number of Images', 'FontSize', 13);
ylim([0 max(labelCount.Count) + 500]);
text(1, labelCount.Count(1) + 80, num2str(labelCount.Count(1)), ...
    'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
text(2, labelCount.Count(2) + 80, num2str(labelCount.Count(2)), ...
    'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
saveas(gcf, 'class_distribution.png');

%% I displayed some sample X-rays to understand what the images look like
fprintf('\n--- Figure 2: Sample X-Ray Images ---\n\n');
figure('Position', [100, 100, 1200, 500]);
sgtitle('Sample Chest X-Ray Images', 'FontSize', 16, 'FontWeight', 'bold');

normalIdx = find(imds.Labels == 'Normal');
tbIdx     = find(imds.Labels == 'Tuberculosis');

for i = 1:5
    subplot(2, 5, i);
    img = readimage(imds, normalIdx(i));
    imshow(img);
    title(['Normal #' num2str(i)], 'FontSize', 10);
end

for i = 1:5
    subplot(2, 5, i + 5);
    img = readimage(imds, tbIdx(i));
    imshow(img);
    title(['Tuberculosis #' num2str(i)], 'FontSize', 10);
end
saveas(gcf, 'sample_images.png');

%% I checked the original image size to know what I'm starting with
img = readimage(imds, 1);
fprintf('Original image size: %d x %d x %d\n', size(img,1), size(img,2), size(img,3));

%% =========================================================================
%  PART 2: PREPROCESSING & DATA AUGMENTATION
%  =========================================================================
%  Before feeding images into any model, I needed to do a few things:
%    - Resize all images to 224x224 because GoogLeNet and ResNet-18 expect that
%    - Convert grayscale X-rays to 3-channel RGB since pretrained models
%      were originally trained on colour images
%    - Apply data augmentation on the training set to help prevent
%      overfitting (I learned about this in the Week 6 lab)
%  =========================================================================

imageSize = [224 224 3];

%% Splitting into train / validation / test (70/15/15)
%  I split the data into three parts: 70% for training, 15% for validation,
%  and 15% for testing. The validation set helps me monitor overfitting
%  during training, and the test set is kept completely separate so I can
%  get a fair result at the end.

trainRatio = 0.70;
valRatio   = 0.15;
testRatio  = 0.15;

[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, ...
    trainRatio, valRatio, testRatio, 'randomized');

fprintf('\nDataset Split:\n');
fprintf('  Training:   %d images (%.0f%%)\n', numel(imdsTrain.Files), trainRatio*100);
fprintf('  Validation: %d images (%.0f%%)\n', numel(imdsValidation.Files), valRatio*100);
fprintf('  Test:       %d images (%.0f%%)\n', numel(imdsTest.Files), testRatio*100);

trainCount = countEachLabel(imdsTrain);
valCount   = countEachLabel(imdsValidation);
testCount  = countEachLabel(imdsTest);
fprintf('\nTraining set per class (before balancing):\n'); disp(trainCount);
fprintf('Validation set per class:\n');   disp(valCount);
fprintf('Test set per class:\n');         disp(testCount);

%% Balancing the training set by oversampling the minority class (TB)
%  The dataset has 3,500 Normal but only 700 TB images, so after splitting
%  the training set is heavily imbalanced. If I don't fix this, the model
%  might just predict "Normal" for everything and still get ~83% accuracy
%  while completely missing TB cases. To solve this, I duplicated the TB
%  training images until both classes have roughly the same number.

tbFiles     = imdsTrain.Files(imdsTrain.Labels == 'Tuberculosis');
normalFiles = imdsTrain.Files(imdsTrain.Labels == 'Normal');
numNormal   = numel(normalFiles);
numTB       = numel(tbFiles);

% Repeat TB images to match the number of Normal images
repeatFactor  = ceil(numNormal / numTB);
tbOversampled = repmat(tbFiles, repeatFactor, 1);
tbOversampled = tbOversampled(1:numNormal);

% Build a new balanced training datastore
balancedFiles  = [normalFiles; tbOversampled];
balancedLabels = [repmat(categorical({'Normal'}), numNormal, 1); ...
                  repmat(categorical({'Tuberculosis'}), numNormal, 1)];

imdsTrain = imageDatastore(balancedFiles);
imdsTrain.Labels = balancedLabels;

fprintf('After balancing - Training set per class:\n');
disp(countEachLabel(imdsTrain));
fprintf('Total training images: %d\n', numel(imdsTrain.Files));

%% Data augmentation for training
%  I applied data augmentation because in real life X-rays can come at
%  slightly different angles and positions. By randomly rotating, shifting
%  and flipping the training images each epoch, the model doesn't just
%  memorise specific images - it actually learns the patterns.

trainAugmenter = imageDataAugmenter( ...
    'RandRotation',     [-15 15], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXReflection',  true, ...
    'RandScale',        [0.9 1.1]);

augTrainDS = augmentedImageDatastore(imageSize, imdsTrain, ...
    'DataAugmentation', trainAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

% For validation and test I only resize and convert to RGB - no augmentation
% because I want to evaluate on the original images, not modified ones
augValDS = augmentedImageDatastore(imageSize, imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

augTestDS = augmentedImageDatastore(imageSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

fprintf('Augmented training datastore ready: %d images\n', augTrainDS.NumObservations);

%% =========================================================================
%  PART 3: CUSTOM CNN ARCHITECTURE (Experiment 1 - "Shallow" Network)
%  =========================================================================
%  For my first experiment I built a custom CNN from scratch. I based the
%  structure on what I learned in the Week 6 lab but made it bigger with
%  4 convolutional blocks. The filters double each time (16 -> 32 -> 64
%  -> 128) so the early layers learn simple things like edges and the
%  deeper layers can pick up more complex patterns like lung shapes and
%  TB abnormalities. I added dropout (50%) in the fully connected layer
%  to reduce overfitting, which is a common problem with medical datasets.
%  =========================================================================

layers = [
    imageInputLayer(imageSize, 'Name', 'input')

    % Block 1 - picks up basic edges and textures
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

    % Block 2 - starts recognising simple shapes
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    % Block 3 - picks up lung regions and structures
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')

    % Block 4 - detects TB-specific patterns (cavities, infiltrates)
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')

    % Classification head
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'dropout1')
    fullyConnectedLayer(2, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

analyzeNetwork(layers, 'TargetUsage', 'trainNetwork');

%% =========================================================================
%  PART 4: TRAINING THE CUSTOM CNN
%  =========================================================================
%  I used Adam as the optimiser because it adapts the learning rate
%  automatically during training, which usually gives better results when
%  training a CNN from scratch. I set ValidationPatience to 5 so training
%  stops automatically if accuracy stops improving, which saves time and
%  prevents overfitting.
%  =========================================================================

iterationsPerEpoch = floor(numel(imdsTrain.Files) / 32);

optionsCustom = trainingOptions('adam', ...
    'InitialLearnRate',     0.001, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.5, ...
    'LearnRateDropPeriod',  5, ...
    'MaxEpochs',            20, ...
    'MiniBatchSize',        32, ...
    'Shuffle',              'every-epoch', ...
    'ValidationData',       augValDS, ...
    'ValidationFrequency',  30, ...
    'ValidationPatience',   5, ...
    'Verbose',              true, ...
    'Plots',                'training-progress');

fprintf('\n============================================\n');
fprintf('  TRAINING: Custom CNN (Experiment 1)\n');
fprintf('============================================\n');

[netCustom, infoCustom] = trainNetwork(augTrainDS, layers, optionsCustom);

fprintf('Custom CNN training complete.\n');

YPredVal_Custom = classify(netCustom, augValDS);
YVal = imdsValidation.Labels;
customValAcc = sum(YPredVal_Custom == YVal) / numel(YVal) * 100;
fprintf('Custom CNN Validation Accuracy: %.2f%%\n', customValAcc);

%% =========================================================================
%  PART 5: TRANSFER LEARNING WITH GOOGLENET (Experiment 2)
%  =========================================================================
%  For my second experiment I used GoogLeNet, which won the ImageNet
%  competition in 2014 with only 5 million parameters. I chose GoogLeNet
%  because it introduced the "inception module" - instead of just stacking
%  convolutions one after another like VGG, it runs multiple filter sizes
%  in parallel and combines them. This makes it both fast and accurate.
%  I learned about this architecture in the Week 7 lecture.
%
%  GoogLeNet also uses global average pooling instead of big FC layers,
%  which is why it has so few parameters compared to VGG16 (5M vs 140M).
%  This makes it much faster to train, especially on CPU.
%
%  What I did: I took the pretrained GoogLeNet, removed the last 3 layers
%  (loss3-classifier, prob, output) which were for 1000 ImageNet classes,
%  and replaced them with my own layers for 2-class TB classification.
%  =========================================================================

netGoogle = googlenet;
lgraphGoogle = layerGraph(netGoogle);

% I removed the last 3 layers that were for ImageNet's 1000 classes
lgraphGoogle = removeLayers(lgraphGoogle, {'loss3-classifier', 'prob', 'output'});

% I added my own layers: FC(2) + Softmax + Classification for TB vs Normal
newLayersGoogle = [
    fullyConnectedLayer(2, 'Name', 'fc_tb_google', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_tb_google')
    classificationLayer('Name', 'output_tb_google')
];

lgraphGoogle = addLayers(lgraphGoogle, newLayersGoogle);
lgraphGoogle = connectLayers(lgraphGoogle, 'pool5-drop_7x7_s1', 'fc_tb_google');

analyzeNetwork(lgraphGoogle, 'TargetUsage', 'trainNetwork');

%% Training options for GoogLeNet
%  I used a smaller learning rate (0.0001) compared to my custom CNN
%  because the pretrained weights are already good - if I use a big
%  learning rate it would destroy what GoogLeNet already learned.

optionsGoogle = trainingOptions('sgdm', ...
    'InitialLearnRate',     0.0001, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.5, ...
    'LearnRateDropPeriod',  5, ...
    'MaxEpochs',            15, ...
    'MiniBatchSize',        32, ...
    'Shuffle',              'every-epoch', ...
    'ValidationData',       augValDS, ...
    'ValidationFrequency',  30, ...
    'ValidationPatience',   5, ...
    'Verbose',              true, ...
    'Plots',                'training-progress');

fprintf('\n============================================\n');
fprintf('  TRAINING: GoogLeNet Transfer Learning (Experiment 2)\n');
fprintf('============================================\n');

[netGoogLeNet, infoGoogLeNet] = trainNetwork(augTrainDS, lgraphGoogle, optionsGoogle);

fprintf('GoogLeNet training complete.\n');

% Save immediately so I don't lose this model if MATLAB crashes later
save(fullfile(savePath, 'trained_googlenet.mat'), 'netGoogLeNet', 'infoGoogLeNet');
fprintf('GoogLeNet saved to: %s\n', fullfile(savePath, 'trained_googlenet.mat'));

YPredVal_Google = classify(netGoogLeNet, augValDS);
googleValAcc = sum(YPredVal_Google == YVal) / numel(YVal) * 100;
fprintf('GoogLeNet Validation Accuracy: %.2f%%\n', googleValAcc);

%% =========================================================================
%  PART 6: TRANSFER LEARNING WITH RESNET-18 (Experiment 3)
%  =========================================================================
%  For my third experiment I chose ResNet-18. The reason I picked this
%  one is because it has "skip connections" (residual connections) which
%  solve the vanishing gradient problem. I learned about this in the
%  Week 7 lecture.
%
%  ResNet-18 has 18 layers and ~11M parameters. It's one of the most
%  popular architectures in medical imaging research so I thought it
%  would be a good fit for TB detection.
%
%  I did the same thing as with GoogLeNet: removed the last 3 layers
%  and added my own classification layers for Normal vs Tuberculosis.
%  =========================================================================

netResNet = resnet18;
lgraphRes = layerGraph(netResNet);

% I removed the last 3 layers (fc1000, prob, ClassificationLayer_predictions)
lgraphRes = removeLayers(lgraphRes, {'fc1000', 'prob', 'ClassificationLayer_predictions'});

newLayersRes = [
    fullyConnectedLayer(2, 'Name', 'fc_tb_res', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_tb_res')
    classificationLayer('Name', 'output_tb_res')
];

lgraphRes = addLayers(lgraphRes, newLayersRes);
lgraphRes = connectLayers(lgraphRes, 'pool5', 'fc_tb_res');

analyzeNetwork(lgraphRes, 'TargetUsage', 'trainNetwork');

%% Training options for ResNet-18
%  I used the same training strategy as GoogLeNet - low learning rate to keep
%  the pretrained features intact, and my new layers learn 10x faster.

optionsRes = trainingOptions('sgdm', ...
    'InitialLearnRate',     0.0001, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.5, ...
    'LearnRateDropPeriod',  5, ...
    'MaxEpochs',            15, ...
    'MiniBatchSize',        32, ...
    'Shuffle',              'every-epoch', ...
    'ValidationData',       augValDS, ...
    'ValidationFrequency',  30, ...
    'ValidationPatience',   5, ...
    'Verbose',              true, ...
    'Plots',                'training-progress');

fprintf('\n============================================\n');
fprintf('  TRAINING: ResNet-18 Transfer Learning (Experiment 3)\n');
fprintf('============================================\n');

[netResNet18, infoResNet] = trainNetwork(augTrainDS, lgraphRes, optionsRes);

fprintf('ResNet-18 training complete.\n');

YPredVal_Res = classify(netResNet18, augValDS);
resValAcc = sum(YPredVal_Res == YVal) / numel(YVal) * 100;
fprintf('ResNet-18 Validation Accuracy: %.2f%%\n', resValAcc);

%% =========================================================================
%  PART 7: EVALUATING ALL THREE MODELS ON THE TEST SET
%  =========================================================================
%  Now I test all three models on the test set which hasn't been used at
%  all during training. This is the only fair way to see how each model
%  would perform on completely new X-rays it has never seen before.
%  =========================================================================

fprintf('\n============================================\n');
fprintf('  TEST SET EVALUATION\n');
fprintf('============================================\n');

[YPredCustom, scoresCustom] = classify(netCustom, augTestDS);
[YPredGoogle, scoresGoogle] = classify(netGoogLeNet, augTestDS);
[YPredRes,    scoresRes]    = classify(netResNet18, augTestDS);
YTest = imdsTest.Labels;

customTestAcc = sum(YPredCustom == YTest) / numel(YTest) * 100;
googleTestAcc = sum(YPredGoogle == YTest) / numel(YTest) * 100;
resTestAcc    = sum(YPredRes == YTest) / numel(YTest) * 100;

fprintf('\n  Custom CNN  Test Accuracy: %.2f%%\n', customTestAcc);
fprintf('  GoogLeNet   Test Accuracy: %.2f%%\n', googleTestAcc);
fprintf('  ResNet-18   Test Accuracy: %.2f%%\n', resTestAcc);

%% =========================================================================
%  PART 8: ACCURACY CURVES
%  =========================================================================
%  The coursework requires accuracy curves showing epochs vs accuracy.
%  MATLAB saves accuracy per iteration not per epoch, so I had to average
%  each epoch's iterations to get clean per-epoch values. I plotted
%  training accuracy, validation accuracy, and the final test accuracy
%  as a reference line on each graph.
%  =========================================================================

% Converting iteration-level data to epoch-level for all three models

% Custom CNN
numItersCustom  = numel(infoCustom.TrainingAccuracy);
numEpochsCustom = ceil(numItersCustom / iterationsPerEpoch);

trainAccEp_Custom = zeros(1, numEpochsCustom);
valAccEp_Custom   = zeros(1, numEpochsCustom);
trainLossEp_Custom = zeros(1, numEpochsCustom);
valLossEp_Custom   = zeros(1, numEpochsCustom);

for ep = 1:numEpochsCustom
    s = (ep-1)*iterationsPerEpoch + 1;
    e = min(ep*iterationsPerEpoch, numItersCustom);
    trainAccEp_Custom(ep)  = mean(infoCustom.TrainingAccuracy(s:e));
    trainLossEp_Custom(ep) = mean(infoCustom.TrainingLoss(s:e));
    vAcc = infoCustom.ValidationAccuracy(s:e);
    vAcc = vAcc(~isnan(vAcc));
    if ~isempty(vAcc), valAccEp_Custom(ep) = vAcc(end);
    elseif ep > 1,     valAccEp_Custom(ep) = valAccEp_Custom(ep-1); end
    vLoss = infoCustom.ValidationLoss(s:e);
    vLoss = vLoss(~isnan(vLoss));
    if ~isempty(vLoss), valLossEp_Custom(ep) = vLoss(end);
    elseif ep > 1,      valLossEp_Custom(ep) = valLossEp_Custom(ep-1); end
end

% GoogLeNet
numItersGoogle  = numel(infoGoogLeNet.TrainingAccuracy);
numEpochsGoogle = ceil(numItersGoogle / iterationsPerEpoch);

trainAccEp_Google = zeros(1, numEpochsGoogle);
valAccEp_Google   = zeros(1, numEpochsGoogle);
trainLossEp_Google = zeros(1, numEpochsGoogle);
valLossEp_Google   = zeros(1, numEpochsGoogle);

for ep = 1:numEpochsGoogle
    s = (ep-1)*iterationsPerEpoch + 1;
    e = min(ep*iterationsPerEpoch, numItersGoogle);
    trainAccEp_Google(ep)  = mean(infoGoogLeNet.TrainingAccuracy(s:e));
    trainLossEp_Google(ep) = mean(infoGoogLeNet.TrainingLoss(s:e));
    vAcc = infoGoogLeNet.ValidationAccuracy(s:e);
    vAcc = vAcc(~isnan(vAcc));
    if ~isempty(vAcc), valAccEp_Google(ep) = vAcc(end);
    elseif ep > 1,     valAccEp_Google(ep) = valAccEp_Google(ep-1); end
    vLoss = infoGoogLeNet.ValidationLoss(s:e);
    vLoss = vLoss(~isnan(vLoss));
    if ~isempty(vLoss), valLossEp_Google(ep) = vLoss(end);
    elseif ep > 1,      valLossEp_Google(ep) = valLossEp_Google(ep-1); end
end

% ResNet-18
numItersRes  = numel(infoResNet.TrainingAccuracy);
numEpochsRes = ceil(numItersRes / iterationsPerEpoch);

trainAccEp_Res = zeros(1, numEpochsRes);
valAccEp_Res   = zeros(1, numEpochsRes);
trainLossEp_Res = zeros(1, numEpochsRes);
valLossEp_Res   = zeros(1, numEpochsRes);

for ep = 1:numEpochsRes
    s = (ep-1)*iterationsPerEpoch + 1;
    e = min(ep*iterationsPerEpoch, numItersRes);
    trainAccEp_Res(ep)  = mean(infoResNet.TrainingAccuracy(s:e));
    trainLossEp_Res(ep) = mean(infoResNet.TrainingLoss(s:e));
    vAcc = infoResNet.ValidationAccuracy(s:e);
    vAcc = vAcc(~isnan(vAcc));
    if ~isempty(vAcc), valAccEp_Res(ep) = vAcc(end);
    elseif ep > 1,     valAccEp_Res(ep) = valAccEp_Res(ep-1); end
    vLoss = infoResNet.ValidationLoss(s:e);
    vLoss = vLoss(~isnan(vLoss));
    if ~isempty(vLoss), valLossEp_Res(ep) = vLoss(end);
    elseif ep > 1,      valLossEp_Res(ep) = valLossEp_Res(ep-1); end
end

%% Accuracy curve - Custom CNN
fprintf('\n--- Figure 3: Custom CNN - Accuracy Over Epochs ---\n\n');
figure('Position', [100, 100, 800, 550]);
plot(1:numEpochsCustom, trainAccEp_Custom, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:numEpochsCustom, valAccEp_Custom, 'r--s', 'LineWidth', 2, 'MarkerSize', 6);
yline(customTestAcc, 'g-.', sprintf('Test: %.1f%%', customTestAcc), ...
    'LineWidth', 2, 'FontSize', 11, 'LabelHorizontalAlignment', 'left');
hold off;
title('Custom CNN - Accuracy Curve', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Number of Epochs', 'FontSize', 13);
ylabel('Accuracy (%)', 'FontSize', 13);
legend('Training', 'Validation', 'Test Accuracy', 'Location', 'southeast', 'FontSize', 12);
grid on; ylim([40 105]); xlim([1 numEpochsCustom]);
saveas(gcf, 'custom_cnn_accuracy_curve.png');

%% Accuracy curve - GoogLeNet
fprintf('\n--- Figure 4: GoogLeNet - Accuracy Over Epochs ---\n\n');
figure('Position', [100, 100, 800, 550]);
plot(1:numEpochsGoogle, trainAccEp_Google, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:numEpochsGoogle, valAccEp_Google, 'r--s', 'LineWidth', 2, 'MarkerSize', 6);
yline(googleTestAcc, 'g-.', sprintf('Test: %.1f%%', googleTestAcc), ...
    'LineWidth', 2, 'FontSize', 11, 'LabelHorizontalAlignment', 'left');
hold off;
title('GoogLeNet - Accuracy Curve', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Number of Epochs', 'FontSize', 13);
ylabel('Accuracy (%)', 'FontSize', 13);
legend('Training', 'Validation', 'Test Accuracy', 'Location', 'southeast', 'FontSize', 12);
grid on; ylim([40 105]); xlim([1 numEpochsGoogle]);
saveas(gcf, 'googlenet_accuracy_curve.png');

%% Accuracy curve - ResNet-18
fprintf('\n--- Figure 5: ResNet-18 - Accuracy Over Epochs ---\n\n');
figure('Position', [100, 100, 800, 550]);
plot(1:numEpochsRes, trainAccEp_Res, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:numEpochsRes, valAccEp_Res, 'r--s', 'LineWidth', 2, 'MarkerSize', 6);
yline(resTestAcc, 'g-.', sprintf('Test: %.1f%%', resTestAcc), ...
    'LineWidth', 2, 'FontSize', 11, 'LabelHorizontalAlignment', 'left');
hold off;
title('ResNet-18 - Accuracy Curve', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Number of Epochs', 'FontSize', 13);
ylabel('Accuracy (%)', 'FontSize', 13);
legend('Training', 'Validation', 'Test Accuracy', 'Location', 'southeast', 'FontSize', 12);
grid on; ylim([40 105]); xlim([1 numEpochsRes]);
saveas(gcf, 'resnet18_accuracy_curve.png');

%% =========================================================================
%  PART 9: LOSS CURVES
%  =========================================================================
%  I also plotted loss curves because they help spot overfitting. If the
%  training loss keeps going down but the validation loss starts going up,
%  that means the model is memorising the training data instead of
%  actually learning useful patterns.
%  =========================================================================

fprintf('\n--- Figure 6: Training and Validation Loss ---\n\n');
figure('Position', [50, 100, 1500, 500]);
sgtitle('Training and Validation Loss', 'FontSize', 16, 'FontWeight', 'bold');

subplot(1, 3, 1);
plot(1:numEpochsCustom, trainLossEp_Custom, '-o', 'Color', [0.00 0.45 0.74], ...
    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.00 0.45 0.74]);
hold on;
plot(1:numEpochsCustom, valLossEp_Custom, '--s', 'Color', [0.00 0.25 0.50], ...
    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.00 0.25 0.50]);
hold off;
title('Custom CNN', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Epochs', 'FontSize', 12); ylabel('Loss', 'FontSize', 12);
legend('Train', 'Val', 'Location', 'northeast', 'FontSize', 9);
grid on;

subplot(1, 3, 2);
plot(1:numEpochsGoogle, trainLossEp_Google, '-o', 'Color', [0.85 0.15 0.15], ...
    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.85 0.15 0.15]);
hold on;
plot(1:numEpochsGoogle, valLossEp_Google, '--s', 'Color', [0.55 0.05 0.05], ...
    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.55 0.05 0.05]);
hold off;
title('GoogLeNet', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Epochs', 'FontSize', 12); ylabel('Loss', 'FontSize', 12);
legend('Train', 'Val', 'Location', 'northeast', 'FontSize', 9);
grid on;

subplot(1, 3, 3);
plot(1:numEpochsRes, trainLossEp_Res, '-o', 'Color', [0.13 0.65 0.30], ...
    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.13 0.65 0.30]);
hold on;
plot(1:numEpochsRes, valLossEp_Res, '--s', 'Color', [0.05 0.40 0.15], ...
    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.05 0.40 0.15]);
hold off;
title('ResNet-18', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Epochs', 'FontSize', 12); ylabel('Loss', 'FontSize', 12);
legend('Train', 'Val', 'Location', 'northeast', 'FontSize', 9);
grid on;

saveas(gcf, 'loss_curves.png');

%% =========================================================================
%  PART 10: CONFUSION MATRICES
%  =========================================================================
%  I created confusion matrices for each model to see exactly what the
%  model gets right and what it gets wrong. The diagonal shows correct
%  predictions, and the off-diagonal shows mistakes. I included row and
%  column summaries to show recall and precision per class.
%  =========================================================================

% Custom CNN confusion matrix
fprintf('\n--- Figure 7: Confusion Matrix - Custom CNN ---\n\n');
figure('Position', [100, 100, 600, 550]);
confMatCustom = confusionmat(YTest, YPredCustom);
confusionchart(confMatCustom, categories(YTest), ...
    'Title', sprintf('Custom CNN (Accuracy: %.2f%%)', customTestAcc), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
saveas(gcf, 'confusion_matrix_custom_cnn.png');

% GoogLeNet confusion matrix
fprintf('\n--- Figure 8: Confusion Matrix - GoogLeNet ---\n\n');
figure('Position', [100, 100, 600, 550]);
confMatGoogle = confusionmat(YTest, YPredGoogle);
confusionchart(confMatGoogle, categories(YTest), ...
    'Title', sprintf('GoogLeNet (Accuracy: %.2f%%)', googleTestAcc), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
saveas(gcf, 'confusion_matrix_googlenet.png');

% ResNet-18 confusion matrix
fprintf('\n--- Figure 9: Confusion Matrix - ResNet-18 ---\n\n');
figure('Position', [100, 100, 600, 550]);
confMatRes = confusionmat(YTest, YPredRes);
confusionchart(confMatRes, categories(YTest), ...
    'Title', sprintf('ResNet-18 (Accuracy: %.2f%%)', resTestAcc), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
saveas(gcf, 'confusion_matrix_resnet18.png');

%% =========================================================================
%  PART 10B: ROC CURVES
%  =========================================================================
%  ROC curves show the trade-off between true positive rate (catching TB)
%  and false positive rate (false alarms) at different thresholds. AUC
%  (Area Under Curve) closer to 1.0 means the model is better at
%  separating the two classes. This is especially important in medical
%  imaging because missing a TB case is much worse than a false alarm.
%  =========================================================================

tbIdx = find(categories(YTest) == "Tuberculosis");
YTestBinary = (YTest == 'Tuberculosis');

fprintf('\n--- Figure 10: ROC Curves ---\n\n');
figure('Position', [50, 100, 1600, 500]);
sgtitle('ROC Curves - TB Detection', 'FontSize', 16, 'FontWeight', 'bold');

% Custom CNN ROC
subplot(1, 3, 1);
[Xroc_C, Yroc_C, ~, AUC_C] = perfcurve(YTestBinary, scoresCustom(:,tbIdx), true);
plot(Xroc_C, Yroc_C, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
hold off;
title(sprintf('Custom CNN\nAUC = %.4f', AUC_C), 'FontSize', 11, 'FontWeight', 'bold');
xlabel('False Positive Rate', 'FontSize', 10);
ylabel('True Positive Rate', 'FontSize', 10);
grid on;

% GoogLeNet ROC
subplot(1, 3, 2);
[Xroc_G, Yroc_G, ~, AUC_G] = perfcurve(YTestBinary, scoresGoogle(:,tbIdx), true);
plot(Xroc_G, Yroc_G, 'r-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
hold off;
title(sprintf('GoogLeNet\nAUC = %.4f', AUC_G), 'FontSize', 11, 'FontWeight', 'bold');
xlabel('False Positive Rate', 'FontSize', 10);
ylabel('True Positive Rate', 'FontSize', 10);
grid on;

% ResNet-18 ROC
subplot(1, 3, 3);
[Xroc_R, Yroc_R, ~, AUC_R] = perfcurve(YTestBinary, scoresRes(:,tbIdx), true);
plot(Xroc_R, Yroc_R, 'g-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
hold off;
title(sprintf('ResNet-18\nAUC = %.4f', AUC_R), 'FontSize', 11, 'FontWeight', 'bold');
xlabel('False Positive Rate', 'FontSize', 10);
ylabel('True Positive Rate', 'FontSize', 10);
grid on;

saveas(gcf, 'roc_curves.png');

fprintf('\nROC AUC Values:\n');
fprintf('  Custom CNN:  %.4f\n', AUC_C);
fprintf('  GoogLeNet:   %.4f\n', AUC_G);
fprintf('  ResNet-18:   %.4f\n', AUC_R);

%% I also plotted all three ROC curves on one figure for easy comparison
fprintf('\n--- Figure 11: ROC Curve Comparison ---\n\n');
figure('Position', [100, 100, 700, 550]);
plot(Xroc_C, Yroc_C, 'b-', 'LineWidth', 2); hold on;
plot(Xroc_G, Yroc_G, 'r-', 'LineWidth', 2);
plot(Xroc_R, Yroc_R, 'g-', 'LineWidth', 2);
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
hold off;
title('ROC Curve Comparison', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('False Positive Rate', 'FontSize', 13);
ylabel('True Positive Rate', 'FontSize', 13);
legend(sprintf('Custom CNN (AUC=%.4f)', AUC_C), ...
       sprintf('GoogLeNet (AUC=%.4f)', AUC_G), ...
       sprintf('ResNet-18 (AUC=%.4f)', AUC_R), ...
       'Random', 'Location', 'southeast', 'FontSize', 11);
grid on;
saveas(gcf, 'roc_comparison.png');

%% =========================================================================
%  PART 11: PRECISION, RECALL, F1-SCORE
%  =========================================================================
%  I calculated precision, recall, and F1-score because accuracy alone
%  can be misleading in medical imaging. For example, if the model says
%  everyone is "Normal" it could still get high accuracy but miss all TB
%  cases. Precision tells me how many of the predicted TB cases are
%  actually TB, recall tells me how many real TB cases I caught, and
%  F1-score balances both. I used the same approach from the Week 6 lab.
%  =========================================================================

classNames = categories(YTest);
numClasses = numel(classNames);

% --- Custom CNN metrics ---
fprintf('\n============================================\n');
fprintf('  METRICS - Custom CNN\n');
fprintf('============================================\n');

precCustom = zeros(numClasses, 1);
recCustom  = zeros(numClasses, 1);
f1Custom   = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confMatCustom(i,i);
    FP = sum(confMatCustom(:,i)) - TP;
    FN = sum(confMatCustom(i,:)) - TP;
    precCustom(i) = TP / (TP + FP + eps);
    recCustom(i)  = TP / (TP + FN + eps);
    f1Custom(i)   = 2*(precCustom(i)*recCustom(i)) / (precCustom(i)+recCustom(i)+eps);
end

disp(table(classNames, precCustom, recCustom, f1Custom, ...
    'VariableNames', {'Class','Precision','Recall','F1Score'}));

TP_c = sum(diag(confMatCustom));
FP_c = sum(sum(confMatCustom,1)) - TP_c;
FN_c = sum(sum(confMatCustom,2)) - TP_c;
oPrec_C = TP_c/(TP_c+FP_c+eps);
oRec_C  = TP_c/(TP_c+FN_c+eps);
oF1_C   = 2*(oPrec_C*oRec_C)/(oPrec_C+oRec_C+eps);

fprintf('Overall  Precision: %.4f  Recall: %.4f  F1: %.4f\n', oPrec_C, oRec_C, oF1_C);

% --- GoogLeNet metrics ---
fprintf('\n============================================\n');
fprintf('  METRICS - GoogLeNet\n');
fprintf('============================================\n');

precGoogle = zeros(numClasses, 1);
recGoogle  = zeros(numClasses, 1);
f1Google   = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confMatGoogle(i,i);
    FP = sum(confMatGoogle(:,i)) - TP;
    FN = sum(confMatGoogle(i,:)) - TP;
    precGoogle(i) = TP / (TP + FP + eps);
    recGoogle(i)  = TP / (TP + FN + eps);
    f1Google(i)   = 2*(precGoogle(i)*recGoogle(i)) / (precGoogle(i)+recGoogle(i)+eps);
end

disp(table(classNames, precGoogle, recGoogle, f1Google, ...
    'VariableNames', {'Class','Precision','Recall','F1Score'}));

TP_v = sum(diag(confMatGoogle));
FP_v = sum(sum(confMatGoogle,1)) - TP_v;
FN_v = sum(sum(confMatGoogle,2)) - TP_v;
oPrec_V = TP_v/(TP_v+FP_v+eps);
oRec_V  = TP_v/(TP_v+FN_v+eps);
oF1_V   = 2*(oPrec_V*oRec_V)/(oPrec_V+oRec_V+eps);

fprintf('Overall  Precision: %.4f  Recall: %.4f  F1: %.4f\n', oPrec_V, oRec_V, oF1_V);

% --- ResNet-18 metrics ---
fprintf('\n============================================\n');
fprintf('  METRICS - ResNet-18\n');
fprintf('============================================\n');

precRes = zeros(numClasses, 1);
recRes  = zeros(numClasses, 1);
f1Res   = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confMatRes(i,i);
    FP = sum(confMatRes(:,i)) - TP;
    FN = sum(confMatRes(i,:)) - TP;
    precRes(i) = TP / (TP + FP + eps);
    recRes(i)  = TP / (TP + FN + eps);
    f1Res(i)   = 2*(precRes(i)*recRes(i)) / (precRes(i)+recRes(i)+eps);
end

disp(table(classNames, precRes, recRes, f1Res, ...
    'VariableNames', {'Class','Precision','Recall','F1Score'}));

TP_r = sum(diag(confMatRes));
FP_r = sum(sum(confMatRes,1)) - TP_r;
FN_r = sum(sum(confMatRes,2)) - TP_r;
oPrec_R = TP_r/(TP_r+FP_r+eps);
oRec_R  = TP_r/(TP_r+FN_r+eps);
oF1_R   = 2*(oPrec_R*oRec_R)/(oPrec_R+oRec_R+eps);

fprintf('Overall  Precision: %.4f  Recall: %.4f  F1: %.4f\n', oPrec_R, oRec_R, oF1_R);

%% =========================================================================
%  PART 12: THREE-WAY MODEL COMPARISON
%  =========================================================================

fprintf('\n============================================\n');
fprintf('  MODEL COMPARISON SUMMARY\n');
fprintf('============================================\n');

fprintf('\n  %-25s %-15s %-15s %-15s\n', 'Metric', 'Custom CNN', 'GoogLeNet', 'ResNet-18');
fprintf('  %-25s %-15s %-15s %-15s\n', '-------------------------', '---------------', '---------------', '---------------');
fprintf('  %-25s %-15.2f %-15.2f %-15.2f\n', 'Validation Accuracy (%)', customValAcc, googleValAcc, resValAcc);
fprintf('  %-25s %-15.2f %-15.2f %-15.2f\n', 'Test Accuracy (%)',       customTestAcc, googleTestAcc, resTestAcc);
fprintf('  %-25s %-15.4f %-15.4f %-15.4f\n', 'Overall Precision',       oPrec_C, oPrec_V, oPrec_R);
fprintf('  %-25s %-15.4f %-15.4f %-15.4f\n', 'Overall Recall',          oRec_C, oRec_V, oRec_R);
fprintf('  %-25s %-15.4f %-15.4f %-15.4f\n', 'Overall F1-Score',        oF1_C, oF1_V, oF1_R);

%% One grouped bar chart to compare all three models side by side
fprintf('\n--- Figure 12: Model Performance Comparison ---\n\n');
metricNames  = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
customScores = [customTestAcc, oPrec_C*100, oRec_C*100, oF1_C*100];
googleScores = [googleTestAcc, oPrec_V*100, oRec_V*100, oF1_V*100];
resScores    = [resTestAcc,    oPrec_R*100, oRec_R*100, oF1_R*100];

barData = [customScores; googleScores; resScores]';
X = categorical(metricNames);
X = reordercats(X, metricNames);

figure('Position', [50, 100, 900, 550]);
b = bar(X, barData, 'grouped');
b(1).FaceColor = [0.00 0.45 0.74]; b(1).EdgeColor = 'none';
b(2).FaceColor = [0.85 0.15 0.15]; b(2).EdgeColor = 'none';
b(3).FaceColor = [0.13 0.65 0.30]; b(3).EdgeColor = 'none';

title('Model Performance Comparison', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Score (%)', 'FontSize', 13);
legend('Custom CNN', 'GoogLeNet', 'ResNet-18', 'Location', 'southwest', 'FontSize', 11);
ylim([93 101]);
set(gca, 'YTick', 93:1:101, 'FontSize', 11);
grid on;

for k = 1:numel(b)
    xtips  = b(k).XEndPoints;
    ytips  = b(k).YEndPoints;
    labels = string(round(b(k).YData, 1)) + "%";
    text(xtips, ytips + 0.1, labels, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 8, 'FontWeight', 'bold');
end

saveas(gcf, 'model_comparison.png');

%% =========================================================================
%  PART 13: SAMPLE PREDICTIONS
%  =========================================================================
%  Finally I wanted to visually check how the best model (GoogLeNet)
%  performs on actual X-ray images. I picked 8 random test images and
%  displayed the true label vs predicted label. Green means correct,
%  red means the model got it wrong.
%  =========================================================================

fprintf('\n--- Figure 13: Sample Predictions (GoogLeNet) ---\n\n');
figure('Position', [100, 100, 1200, 700]);
sgtitle('Sample Test Predictions (GoogLeNet - Best Model)', ...
    'FontSize', 16, 'FontWeight', 'bold');

rng(42);
sampleIdx = randperm(numel(YTest), 8);

for i = 1:8
    subplot(2, 4, i);
    img = readimage(imdsTest, sampleIdx(i));
    imshow(img);

    trueLabel = string(YTest(sampleIdx(i)));
    predLabel = string(YPredGoogle(sampleIdx(i)));

    if trueLabel == predLabel
        titleColor = [0 0.6 0];
    else
        titleColor = [0.8 0 0];
    end

    title(sprintf('True: %s | Pred: %s', trueLabel, predLabel), ...
        'FontSize', 10, 'Color', titleColor, 'FontWeight', 'bold');
end

saveas(gcf, 'sample_predictions.png');

%% =========================================================================
%  PART 14: SAVING THE TRAINED MODELS
%  =========================================================================
%  I saved all three trained models so I don't have to retrain them every
%  time I want to use or test them again.

save(fullfile(savePath, 'trained_custom_cnn.mat'), 'netCustom', 'infoCustom');
save(fullfile(savePath, 'trained_googlenet.mat'), 'netGoogLeNet', 'infoGoogLeNet');
save(fullfile(savePath, 'trained_resnet18.mat'), 'netResNet18', 'infoResNet');
fprintf('\nAll three models saved to .mat files.\n');

%% Final summary
fprintf('\n============================================\n');
fprintf('  ALL DONE\n');
fprintf('============================================\n');
fprintf('  Custom CNN  - Val: %.2f%%  |  Test: %.2f%%\n', customValAcc, customTestAcc);
fprintf('  GoogLeNet   - Val: %.2f%%  |  Test: %.2f%%\n', googleValAcc, googleTestAcc);
fprintf('  ResNet-18   - Val: %.2f%%  |  Test: %.2f%%\n', resValAcc, resTestAcc);
fprintf('============================================\n');
fprintf('  Figures saved:\n');
fprintf('    class_distribution.png\n');
fprintf('    sample_images.png\n');
fprintf('    custom_cnn_accuracy_curve.png\n');
fprintf('    googlenet_accuracy_curve.png\n');
fprintf('    resnet18_accuracy_curve.png\n');
fprintf('    loss_curves.png\n');
fprintf('    confusion_matrix_custom_cnn.png\n');
fprintf('    confusion_matrix_googlenet.png\n');
fprintf('    confusion_matrix_resnet18.png\n');
fprintf('    roc_curves.png\n');
fprintf('    roc_comparison.png\n');
fprintf('    model_comparison.png\n');
fprintf('    sample_predictions.png\n');
fprintf('============================================\n');
