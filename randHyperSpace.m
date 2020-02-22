%%
numRandom =50;

r = -(1.5*rand(numRandom,1)+1.5); %10^-6 : 1 % 10^-5 : 10^-1
alpha = 10.^r;

r = -(3.8*rand(numRandom,1)+0.8);
beta = 1 - 10.^r;

r = -(4*rand(numRandom,1)+1);
lambda = 10.^r;

trainSetSize = 54200;
lowerLim = 4;
upperLim = log2(trainSetSize)-2;
r = ((upperLim - lowerLim)*rand(numRandom,1)+lowerLim);
mbs = floor(2.^r);

