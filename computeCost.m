function [J] = computeCost(Yhat,Y,W,hyperParams)

regulizer = 0;
for l=1:length(hyperParams.LayerDims)-1
    regulizer = regulizer + 0.5 * hyperParams.L2reg.Lambda * sum(sum(W{l}.^2));
end

m=size(Y,2);
switch hyperParams.activationLayerType{end}{1}   
    case 'softmax'
        J = 1/m * (-sum(sum(Y.*log(Yhat))) + regulizer);
%         J = 1/m * (-diag(Yhat*log(Y')) + regulizer);
% if ~isempty(find(isnan(log(Yhat)),1,'first'))
%     find(isnan(log(Yhat)),1,'first')
% end
    otherwise
        error('Need compute cost implementation for last Layer other than softmax activation')
end