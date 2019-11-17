% CONVOLUTIONAL NETWORKS

[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(4);

inputSize = [ 32 32 3];
nClasses = 10;

layers1 = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,20,'Padding',1,'Stride',1)
	reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(nClasses)
    softmaxLayer
    classificationLayer];

options1 = trainingOptions('sgdm', ...
           'Momentum',0.9, ...
           'MaxEpochs',120, ...
           'Plots','Training-Progress', ...
           'MiniBatchSize', 8192, ...
           'InitialLearnRate',0.001, ...
           'Shuffle','every-epoch', ...
           'ValidationPatience', 3, ...
           'ValidationFrequency', 30,...
            'ValidationData', {xValid,tValid});
          
net1 = trainNetwork(xTrain,tTrain,layers1,options1);

test_error1 = 1-mean(net1.classify(xTest) == tTest);
valid_error1 = 1-mean(net1.classify(xValid) == tValid);
train_error1 = 1-mean(net1.classify(xTrain) == tTrain);

layers2 = [
    imageInputLayer(inputSize)
    convolution2dLayer(3,20,'Padding',1,'Stride',1)
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,30,'Padding',1,'Stride',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,50,'Padding',1,'Stride',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(nClasses)
    softmaxLayer
    classificationLayer];

options2 = trainingOptions('sgdm', ...
           'Momentum',0.9, ...
           'MaxEpochs',120, ...
           'Plots','Training-Progress', ...
           'MiniBatchSize', 8192, ...
           'InitialLearnRate',0.001, ...
           'Shuffle','every-epoch', ...
           'ValidationPatience', 3, ...
           'ValidationFrequency', 30,...
            'ValidationData', {xValid,tValid});
        
net2 = trainNetwork(xTrain,tTrain,layers2,options2);

test_error2 = 1-mean(net2.classify(xTest) == tTest);
valid_error2 = 1-mean(net2.classify(xValid) == tValid);
train_error2 = 1-mean(net2.classify(xTrain) == tTrain);
