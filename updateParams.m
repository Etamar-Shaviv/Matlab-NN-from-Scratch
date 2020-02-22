function [W b VdW Vdb] = updateParams(W,b,grads,hyperParams,epoch,VdW,Vdb,SdW,Sdb)

for l = 1:length(hyperParams.LayerDims)-1 
    
    %Momentum
    VdW{l} = hyperParams.Momentum.Beta * VdW{l} + (1-hyperParams.Momentum.Beta) * grads.dW{l};
    Vdb{l} = hyperParams.Momentum.Beta * Vdb{l} + (1-hyperParams.Momentum.Beta) * grads.db{l};
    %Momentum bias correction
    VdW_ = VdW{l}/(1-hyperParams.Momentum.Beta^epoch);
    Vdb_ = Vdb{l}/(1-hyperParams.Momentum.Beta^epoch);
%     %RMSprop
    if hyperParams.RMSprop.Enable
        SdW{l} = hyperParams.RMSprop.Beta * SdW{l} + (1-hyperParams.RMSprop.Beta) * grads.dW{l}.^2;
        Sdb{l} = hyperParams.RMSprop.Beta * Sdb{l} + (1-hyperParams.RMSprop.Beta) * grads.db{l}.^2;
        %RMSprop bias correction
        SdW_ = SdW{l}/(1-hyperParams.RMSprop.Beta^epoch);
        Sdb_ = Sdb{l}/(1-hyperParams.RMSprop.Beta^epoch);
        %update rule
        W{l} = W{l} - hyperParams.LearningRate * VdW_./(sqrt(SdW_) + 1e-8);
        b{l} = b{l} - hyperParams.LearningRate * Vdb_./(sqrt(Sdb_) + 1e-8);
    else  
        W{l} = W{l} - hyperParams.LearningRate * VdW_;
        b{l} = b{l} - hyperParams.LearningRate * Vdb_;    
    end
end


end
