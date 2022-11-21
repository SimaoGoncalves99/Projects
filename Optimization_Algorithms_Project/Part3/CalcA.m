function [A,derivatives] = CalcA(funcs_no,dim,yk,k,p_vector,lambda)

partials = zeros(funcs_no,dim);

pos = dim/k;
deriv_map = zeros(2,pos);
a = 1;
for j = 1:pos
    deriv_map(1,a:a+(k-1)) = j;
    deriv_map(2,a:a+(k-1)) = linspace(1,k,k);
    a = a+k;
end

for p = 1:size(p_vector,2)  
    
    n = p_vector(1,p);
    m = p_vector(2,p);
    y_n = yk((n-1)*k+1:(n)*k);
    y_m = yk((m-1)*k+1:(m)*k);
    
        %Derivative computation
        for derivative_count = 1:dim
            
            i = deriv_map(1,derivative_count);
            j = deriv_map(2,derivative_count);
            
                if(i == n)
                    partials(p,derivative_count) = -(yk((m-1)*k+j)-yk((n-1)*k+j))/(norm(y_m-y_n));
                end
                if (i == m)
                    partials(p,derivative_count) = (yk((m-1)*k+j)-yk((n-1)*k+j))/(norm(y_m-y_n));
                end    
        end
end

A = [partials;sqrt(lambda)*eye(dim)];
derivatives = partials;
end

            
            
        
    