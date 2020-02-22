function [VdW Vdb  SdW Sdb] = initOptimizerParams(hyperParams)
for l = 1:length(hyperParams.LayerDims)-1 %do not iclude input/output
    VdW{l} = zeros(hyperParams.LayerDims(l+1),hyperParams.LayerDims(l));
    Vdb{l} = zeros(hyperParams.LayerDims(l+1),1);  
    SdW{l} = zeros(hyperParams.LayerDims(l+1),hyperParams.LayerDims(l));
    Sdb{l} = zeros(hyperParams.LayerDims(l+1),1);     
end