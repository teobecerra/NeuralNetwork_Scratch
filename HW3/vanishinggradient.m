
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(2);
xTrainMean = mean(xTrain,2);
xTrainNew = xTrain - xTrainMean;
xValidNew = xValid - xTrainMean;
nPatterns = size(xTrain,2);
learnRate = 0.01;
batchSize = 100;
experiments = 100; %100

[layers, weights, thresholds, errors, uLayers] = selectNetwork(6);
L = size(layers,2);
    
energyLevels = zeros(1,40000);
energyPoints = zeros(1,100);

dataPoints = cell(1,L);
for i = 2:L
    dataPoints{i} = zeros(1,100);
end

for i = 1: (400 * experiments)       
    randArr = randi(40000,batchSize,1);
    miniBatchTrain = xTrainNew(:,randArr);
    miniBatchTarget = tTrain(:,randArr);

    tUpdate = cell(1,L);
    for t = 2:L
        tUpdate{t} = zeros(size(layers{t},1),1);
    end

    wUpdate = cell(1,L);
    for w = 2:L
        wUpdate{w} = zeros(size(layers{w-1},1),size(layers{w},1));
    end
    
    energy = 0;

    %TRAIN
    for index = 1:batchSize
        layers{1} = miniBatchTrain(:, index);
        for l = 2:L
            layers{l} = sigmoid(weights{l}' * layers{l-1} - thresholds{l});
        end

        %CALC DIFF
        diff = miniBatchTarget(:,index) - layers{L};

        energy = energy + sum(diff.*diff);
         
        %CALC ERROR
        errors{L} = (layers{L}.*(1-layers{L})).*diff;
        for e = (L-1):-1:2
            errors{e} = weights{e+1} * errors{e+1}.* (layers{e}.*(1-layers{e}));       
        end

        %CALC THRESH & WEIGHT UPDATE (BATCH STYLE)
        for t = 2:L
            tUpdate{t} = tUpdate{t} + errors{t};
            wUpdate{t} = wUpdate{t} + layers{t-1} * errors{t}';
        end  
    end
    
    energyLevels(i) = energy / 2;
   %UPDATE WEIGHTS/THRESH/uLAYERS (BATCH STYLE)
   for w = 2:L
       uLayers{w}(1, i) = mean(abs(tUpdate{w}));         %threshold or errors(tUpdate)?
       weights{w} = weights{w} + learnRate * wUpdate{w};
       thresholds{w} = thresholds{w} - learnRate * tUpdate{w} ;
   end
end

for j = 2:L
    for i = 1:100
        min = (i-1) * (nPatterns/batchSize) +1;
        max = i * (nPatterns/batchSize) -1;
        dataPoints{j}(1,i) = mean(uLayers{j}(1,min:max));
    end
end

for i = 1:100
    min = (i-1) * (nPatterns/batchSize) +1;
    max = i * (nPatterns/batchSize) -1;
    energyPoints(i) = sum(energyLevels(1,min:max));
end

semilogy(1:100, dataPoints{2},1:100, dataPoints{3}...
    ,1:100, dataPoints{4},1:100, dataPoints{5} ... 
    ,1:100, dataPoints{6})
legend({'Layer 1', 'Layer 2','Layer 3','Layer 4','Output Layer'}, 'Location','northeast');
ylabel({'Log(U)'});
xlabel({'Epochs'});
title({'Learning Speed by Layer'});

hold on
plot(energyPoints, 'LineWidth',1);
legend({'Energy Level'}, 'Location','southeast');
ylabel({'Energy(H)'});
xlabel({'Epoch'});
title({'Energy function'});
hold off
    
function [layers, weights, thresholds, errors,uLayers] = selectNetwork(n)
    if(n==1)
       [layers, weights, thresholds, errors, uLayers] = generateNetwork(2, [3072,20]);
    end
    if(n==2)
      [layers, weights, thresholds, errors,uLayers] = generateNetwork(3, [3072,20,20]);
    end
    if(n==3)
      [layers, weights, thresholds, errors,uLayers] = generateNetwork(3,[3072,20,10]);
    end
    if(n==4)
     [layers, weights, thresholds, errors,uLayers] = generateNetwork(4,[3072,50,50,10]);
    end
    if(n==6)
      [layers,weights,thresholds,errors,uLayers] = generateNetwork(6,[3072,20,20,20,20,10]);
    end
    return
end

function [layers, weights, thresholds,errors,uLayers] = generateNetwork(nLayers, nNeurons)
    layers = cell(1,nLayers);
    
    for l = 1:nLayers
        layers{l} = zeros(nNeurons(l),1);
    end
    
    thresholds = cell(1,nLayers);
    errors = cell(1,nLayers);
    weights = cell(1,nLayers);
    uLayers = cell(1,nLayers);
    
    for t = 2:nLayers
        thresholds{t} = zeros(nNeurons(t),1);
        errors{t} = zeros(nNeurons(t),1);
        weights{t} = initWeights(nNeurons(t-1),nNeurons(t));
        uLayers{t} = zeros(1,40000); %experiments * nPatterns/batchSize
    end

    return   
end
 
function [weights] = initWeights(row,col)
    gg = 1/sqrt(row);
    weights = gg *randn(row,col);
end
    
function [g] = sigmoid(b)
    g =  1 ./ (1 + exp(-b));
end