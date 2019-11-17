[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(1);
xTrainMean = mean(xTrain,2);
xTrainNew = xTrain - xTrainMean;
xValidNew = xValid - xTrainMean;
xTestNew = xTest - xTrainMean;
network = cell(1,4);
netResults = cell(1,4);

for n = 1:4 %4 %Main 
    results = zeros(1,4);
    network{n} = cell(4,20); %tError, vError, weights, thresh
    
    [layers, weights, thresholds, errors] = selectNetwork(n);
    
    [network{n}{1}, network{n}{2},network{n}{3},network{n}{4}] = trainNetwork(xTrainNew, tTrain, xValidNew, tValid, layers, weights, thresholds, errors);  
    
    %ASSIGN WEIGHTS & TRESHS FROM BEST VALID-ERROR
    
    [~,index] = min(network{n}{2});
    results(1) = index;
    results(2) = network{n}{1}(index);  %Ctrain
    results(3) = network{n}{2}(index);  %Cvalid
    weights = network{n}{3}{index};
    thresholds = network{n}{4}{index};
    
    %USE WEIGTHS AND THRESH AGAINST TEST SET
    results(4) = testNetwork(layers,weights,thresholds,xTestNew,tTest,size(layers,2));
    netResults{n} = results;
end

semilogy(1:20, network{1}{1}(1,:),1:20, network{1}{2}(1,:)...
    ,1:20, network{2}{1}(1,:),1:20, network{2}{2}(1,:) ... 
    ,1:20, network{3}{1}(1,:),1:20, network{3}{2}(1,:) ...
    ,1:20, network{4}{1}(1,:),1:20, network{4}{2}(1,:),'black', 'LineWidth',1);
legend({'N1 Train', 'N1 Valid','N2 Train','N2 Valid','N3 Train','N3 Valid','N4 Train','N4 Valid'}, 'Location','northeast');
ylabel({'Error rate'});
xlabel({'Epochs'});
title({'Classification Errors of Networks'});


function [layers, weights, thresholds, errors] = selectNetwork(n)
    if(n==1)
       [layers, weights, thresholds, errors] = generateNetwork(2, [3072,10]);
    end
    if(n==2)
      [layers, weights, thresholds, errors] = generateNetwork(3, [3072,10,10]);
    end
    if(n==3)
      [layers, weights, thresholds, errors] = generateNetwork(3,[3072,50,10]);
    end
    if(n==4)
     [layers, weights, thresholds, errors] = generateNetwork(4,[3072,50,50,10]);
    end
    return
end

function [layers, weights, thresholds,errors] = generateNetwork(nLayers, nNeurons)
    layers = cell(1,nLayers);
    
    for l = 1:nLayers
        layers{l} = zeros(nNeurons(l),1);
    end
    
    thresholds = cell(1,nLayers);
    errors = cell(1,nLayers);
    weights = cell(1,nLayers);
    
    for t = 2:nLayers
        thresholds{t} = zeros(nNeurons(t),1);
        errors{t} = zeros(nNeurons(t),1);
        weights{t} = initWeights(nNeurons(t-1),nNeurons(t));
    end

    return
end

function [trainError, validError, W, T] = trainNetwork(xTrainNew, tTrain, xValidNew, tValid, layers, weights, thresholds, errors)
    L = size(layers,2);
    learnRate = 0.1;
    batchSize = 100;
    trainError = zeros(1,20);
    validError = zeros(1,20);
    
    for epoch = 1:20      %20
        batchIndex = 1;
        perm = randperm(40000);
        
        for i = 1:400    %TRAINING   
            randArr = perm(1, batchIndex:batchIndex+99);
            batchIndex = batchIndex + 100;
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

            %TRAIN
            for index = 1:batchSize
                layers{1} = miniBatchTrain(:, index);
                for l = 2:L
                    layers{l} = sigmoid(weights{l}' * layers{l-1} - thresholds{l});
                end

                %CALC ERROR
                errors{L} = (layers{L}.*(1-layers{L})).*(miniBatchTarget(:,index)-layers{L});
                for e = (L-1):-1:2
                    errors{e} = weights{e+1} * errors{e+1}.* (layers{e}.*(1-layers{e}));       
                end

                %CALC THRESH & WEIGHT UPDATE (BATCH STYLE)
                for t = 2:L
                    tUpdate{t} = tUpdate{t} + errors{t};
                    wUpdate{t} = wUpdate{t} + layers{t-1} * errors{t}';
                end  
            end

           %UPDATE WEIGHTS (BATCH STYLE)
           for w = 2:L
               weights{w} = weights{w} + learnRate * wUpdate{w};
               thresholds{w} = thresholds{w} - learnRate * tUpdate{w};
           end
        end
        
        trainError(epoch) = testNetwork(layers, weights, thresholds, xTrainNew, tTrain,L);
        validError(epoch) = testNetwork(layers, weights, thresholds, xValidNew, tValid,L);
        W{epoch} = weights;
        T{epoch} = thresholds;
    end
    return          
end

function [error] = testNetwork(layers, weights, thresholds, xData, tData,L)
    error = 0;
    dataSize = size(xData,2);
    for index = 1:dataSize
       layers{1} = xData(:, index); 
       for l = 2:L
           layers{l} = sigmoid(weights{l}'* layers{l-1} - thresholds{l});
       end
       t = 1;
       while((tData(t,index) ~=1))
           t = t+1;
       end
       [~,ind] = max(layers{L});
       if (t~=ind)
           error = error + 1;
       end
    end
    error = error / dataSize;
    return
end
 
function [weights] = initWeights(row,col)
    gg = 1/sqrt(row);
    weights = gg *randn(row,col);
end
    
function [g] = sigmoid(b)
    g =  1 ./ (1 + exp(-b));
end