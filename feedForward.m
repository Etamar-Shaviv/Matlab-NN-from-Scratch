function [A,Z,other] = feedForward(X,W,b,hyperParams)
other = [];

for l = 1:length(hyperParams.LayerDims)-1 
    if l==1, Atmp = X; else Atmp = A{l-1}; end
    m = size(Atmp,2);
    Z{l} = W{l}*Atmp + repmat(b{l},[1,m]);
    %Add batch norm here
%     mu = 1/m*sum(Z{l},2);
%     sig2 = 1/m*sum((Z{l}-mu).^2,2);
%     
%     Zn{l} = (Z{l}-mu)./sqrt((sig2-eps));
%     Zt{l} = Gamma.*Zt{l} + Beta;
%     
    switch hyperParams.activationLayerType{l}{1}
        case 'relu'
            A{l} = max(0,Z{l});
        case 'leaky_relu'
            leakyParam = hyperParams.activationLayerType{l}{2};
            A{l} = max(leakyParam*Z{l},Z{l});
        case 'tanh'
            A{l} = (exp(Z{l})-exp(-Z{l}))./(exp(Z{l})+exp(-Z{l}));
        case 'softmax'
            tmp = exp(Z{l});
            A{l} = tmp./repmat(sum(tmp),[size(Z{l},1),1]);
        otherwise
            A{l} = Z{l};
    end
%     if ~isempty(find(isnan(A{l}),1,'first'))
%         [r c] =find(isnan(A{l}),1,'first')
%         break
%     end
end


end