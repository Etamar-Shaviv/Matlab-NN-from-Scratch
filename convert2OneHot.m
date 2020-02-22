function Y = convert2OneHot(labels,C)
%assuming Classes run from 0 to C-1
if nargin < 2 || isempty(C), C = max(labels); end
     
m=length(labels);
Y = zeros(C,m);
for k=1:m
    if labels(k)==0 
        Y(C,k)=1;
    else
        Y(labels(k),k)=1;
    end
end
    