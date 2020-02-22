function [W b] = initLearnableParamaetrs(hyperParams)


for l = 1:length(hyperParams.LayerDims)-1 %do not iclude input/output
    %Normal distrabution [-1 1]
    W{l} = rand(hyperParams.LayerDims(l+1),hyperParams.LayerDims(l))*2 - 1;
    %Change values to move mean & variance: xavier method
    W{l} = W{l} * sqrt(2/(hyperParams.LayerDims(l) + hyperParams.LayerDims(l+1)));
    
    
    b{l} = zeros(hyperParams.LayerDims(l+1),1);  
end

end