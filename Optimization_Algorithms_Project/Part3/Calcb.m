function [b,fnm] = Calcb(A,D,yk,k,p_vector,lambda)


for p = 1:(size(p_vector,2))  
        n = p_vector(1,p);
        m = p_vector(2,p);
        y_n = yk((n-1)*k+1:(n)*k);
        y_m = yk((m-1)*k+1:(m)*k);
        fnm(p,:) = (norm(y_m-y_n)-D(m,n));
        b(p,:) = A(p,:)*yk-fnm(p,:);
end

vector = yk.*sqrt(lambda);

b = [b;vector];
end