function delta = gradChecking(grads,W,b,X,Y,hyperParams)
%roll wieghts and grads to single vector)
epsilon = 1e-7;

for l = 1:length(hyperParams.LayerDims)-1 
    dW{l} = ones(size(grads.dW{l}));
    for j = 1:size(W{l},1)
        for k = 1:size(W{l},2)
            Wminus = W;         
            Wminus{l}(j,k) = W{l}(j,k)-epsilon;
            [A,Z,other] = feedForward(X,Wminus,b,hyperParams);
            Jminus = computeCost(A{end},Y,Wminus,hyperParams);
            
            Wplus = W;  
            Wplus{l}(j,k)  = W{l}(j,k)+epsilon;
            [A,Z,other] = feedForward(X,Wplus,b,hyperParams);
            Jplus = computeCost(A{end},Y,Wplus,hyperParams);
            
            dW{l}(j,k) = (Jplus - Jminus)/(2*epsilon);
        end
    end
    delta{l} = grads.dW{l} - dW{l};
end

G = [grads.dW{1}(:); grads.dW{2}(:)]
GA = [dW{1}(:); dW{2}(:)]
norm(G-GA)/(norm(G)+norm(GA))
dd=dW{l}-grads.dW{l};
d=dd(1,1:k-1);
norm(d)
norm(dW{l}(1,1:k-1))
norm(d)/(norm(dW{l}(1,1:k-1)) + norm(grads.dW{l}(1,1:k-1)))
figure;plot(d)
end