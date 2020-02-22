function grads = backProp(A,Z,Y,W,b,X,hyperParams)

m = size(Y,2);

%Starting with last layer ('Softmax')
if strcmp(hyperParams.activationLayerType{end}{1},'softmax')==0 
    error('Current implementaion supports only softmax as output layer');
else
    l=length(hyperParams.LayerDims)-1;
    dZ{l} = A{l} - Y;
    dW{l} = 1/m * dZ{l} * A{l-1}';
    db{l} = 1/m * sum(dZ{l},2);
end

%Other layers
for l=length(hyperParams.LayerDims)-2:-1:1
    switch hyperParams.activationLayerType{l}{1}   
    case 'leaky_relu'
        gTag = ones(size(Z{l})) * hyperParams.activationLayerType{l}{2};
        gTag(Z{l} > 0) = 1;
    case 'relu'
        gTag = zeros(size(Z{l}));
        gTag(Z{l} > 0) = 1;
    case 'tanh'
        gTag = 1 - A{l}.^2;
        
    otherwise
        error('Need implementation Layer back prop gTag')
    end
    dZ{l} = W{l+1}' * dZ{l+1} .* gTag;
    if l==1, Atmp = X; else Atmp = A{l-1}; end
    lambda = hyperParams.L2reg.Lambda;
    dW{l} = 1/m * dZ{l} * Atmp' + lambda/m * W{l};
    db{l} = 1/m * sum(dZ{l},2);
    
end
grads.dW = dW;
grads.db = db;


end