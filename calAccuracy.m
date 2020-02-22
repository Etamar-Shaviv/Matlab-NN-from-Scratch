function classCorrectAcc = calAccuracy(W,b,X,Y,hyperParams)
[A,Z,other] = feedForward(X,W,b,hyperParams);
Yhat = A{end} == repmat(max(A{end}),10,1);
correct = Yhat==Y;
classCorrectAcc = length(find(sum(correct)==10))/size(correct,2);
end