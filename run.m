clear all
%%
load('Data\Train_Sorted_EqualDistobutionPerDigit.mat');
samples = randperm(length(Train_Sorted_EqualDistobutionPerDigit.Y),54200);
samples = 1:54200;
X = Train_Sorted_EqualDistobutionPerDigit.X(:,samples);
Y = convert2OneHot(Train_Sorted_EqualDistobutionPerDigit.Y(samples),10);
%%
load('Data\Test_Sorted_EqualDistobutionPerDigit.mat');
samplesPerDigit = length(Test_Sorted_EqualDistobutionPerDigit.Y)/10;
crossValX = []; crossValY =[];
testX     = []; testY     =[];
for c = 1:2:20
    samples = (c-1)*floor(samplesPerDigit/2)+1:(c-1)*floor(samplesPerDigit/2)+floor(samplesPerDigit/2);
    crossValX = [crossValX Test_Sorted_EqualDistobutionPerDigit.X(:,samples)];
    crossValY = [crossValY convert2OneHot(Test_Sorted_EqualDistobutionPerDigit.Y(samples),10)];
    
    samples = (c)*ceil(samplesPerDigit/2)+1:(c)*ceil(samplesPerDigit/2)+ceil(samplesPerDigit/2);
    testX = [testX Test_Sorted_EqualDistobutionPerDigit.X(:,samples)];
    testY = [testY convert2OneHot(Test_Sorted_EqualDistobutionPerDigit.Y(samples),10)];
end
%%
randHyperSpace;
for r =1:numRandom
    display(sprintf('run number %d',r));
    hyperParams.LayerDims = [784 200 50 20 10];
    hyperParams.activationLayerType = {{'leaky_relu',0.01},{'relu'},{'tanh'},{'softmax'}};
    hyperParams.L2reg.Lambda = lambda(r);%0.001;
    hyperParams.LearningRate = alpha(r);%0.255;%3e-6;%0.00001 ;%0.25/4;
    hyperParams.Momentum.Beta = beta(r);%0.9;
    hyperParams.RMSprop.Enable = false;
    hyperParams.RMSprop.Beta = 0.999;
    hyperParams.miniBatchSize  = mbs(r);%256;
    %%
    [p, j] = trainModel(hyperParams,X,Y,100,500,5);
    params{r} = p;J{r} = j;
    %%
    trainAcc(r) = calAccuracy(params{r}.W,params{r}.b,X,Y,hyperParams);
    crossValAcc(r) = calAccuracy(params{r}.W,params{r}.b,crossValX,crossValY,hyperParams);
    testAcc(r) = calAccuracy(params{r}.W,params{r}.b,testX,testY,hyperParams);
end