%% This code retrains Googlenet to classify 3 types of imagens
% monkeys , dogs and cats.

close all
clear all

%% Dataset
digitDatasetPath = 'C:\Users\pedro.shinzato\Desktop\imagens processadas\RGB_G';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

numTrainFiles = 113;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Net
net = googlenet;
lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

%% Layers
% To retrain a neural network, we only remove the last layes e create
% new ones that fit to our specific problem

lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
%% Congelando Camadas iniciais
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);

%% Data Adjustments

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% Network training

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',5, ...
    'Verbose',false ,...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%% Display classification

idx = randperm(numel(imdsValidation.Files),20);
figure
for i = 1:20
    subplot(4,5,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
