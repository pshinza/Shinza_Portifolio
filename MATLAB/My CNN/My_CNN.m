%% My CNN

close all
clear all

%% Load Data
digitDatasetPath = 'C:\Users\pedro.shinzato\Desktop\imagens processadas\RGB2';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

figure;
perm = randperm(3000,6);
for i = 1:6
    subplot(2,3,i);
    imshow(imds.Files{perm(i)});
end

labelCount = countEachLabel(imds);

numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


%% Network Architeture
layers = [
    imageInputLayer([224 224 3])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

%% Training
options = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',15, ...
    'ValidationPatience',Inf,...
    'Shuffle','every-epoch',...
    'Verbose',false, ...    
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);  
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)