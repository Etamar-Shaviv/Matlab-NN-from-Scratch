function [params, J] = trainModel (hyperParams,X,Y,maxTime,maxIter,printNum)

J=[];minJ = inf;
m = size(Y,2);
numMiniBatches = ceil(m/hyperParams.miniBatchSize);
[W b] = initLearnableParamaetrs(hyperParams);
[VdW Vdb SdW Sdb] = initOptimizerParams(hyperParams);

stop = false;
e=0;
tic
while ~stop
    e=e+1;
    order = randperm(m);
    XfullBatch = X(:,order);
    YfullBatch = Y(:,order);
    
    for n = 1:numMiniBatches
        first = (n-1)*hyperParams.miniBatchSize + 1;
        last = min(n*hyperParams.miniBatchSize,m);
        
        if last - first <= 0.5 * hyperParams.miniBatchSize, continue; end
        Xbatch = XfullBatch(:,first:last);
        Ybatch = YfullBatch(:,first:last);
    
        [A,Z,other] = feedForward(Xbatch,W,b,hyperParams);
        J(end+1) = computeCost(A{end},Ybatch,W,hyperParams);
        grads = backProp(A,Z,Ybatch,W,b,Xbatch,hyperParams);
    %    res = gradChecking(grads,W,b,X,Y,hyperParams);
        [W b VdW Vdb] = updateParams(W,b,grads,hyperParams,e,VdW,Vdb,SdW,Sdb);
        
        useMin=false;
        if useMin
            if J(end) <= minJ
                minCostW = W;
                minCostb = b;
                minJ = J(end);
            end
        end
        if toc > maxTime, stop=true; break; end
        
    end
    
%     printNum=25;
    if mod(e,printNum)==0    
        display(sprintf('epoch: %d - Cost: %f',e,J(end)));
    end
%     printNum=25;
%     if mod(e,printNum)==0
%         trainAcc(e/printNum) = calAccuracy(W,b,X,Y,hyperParams);
%         crossValAcc(e/printNum) = calAccuracy(W,b,crossValX,crossValY,hyperParams);
%         testAcc(e/printNum) = calAccuracy(W,b,testX,testY,hyperParams);
%         display(sprintf('epoch: %d - Cost: %f - trainAcc: %f, CrossAcc: %f, TestAcc: %f - toc: %f',e,J(e),trainAcc(e/printNum),crossValAcc(e/printNum),testAcc(e/printNum),toc));         
%     end

    if e > maxIter, stop = true; end
end
if useMin
    params.W = minCostW;
    params.b = minCostb;
else
    params.W = W;
    params.b = b;
end