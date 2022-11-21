function fnmNEW = CalcfNEW(ykNEW,p_vector,k,D)

for p = 1:(size(p_vector,2))  
        n = p_vector(1,p);
        m = p_vector(2,p);
        y_n = ykNEW((n-1)*k+1:(n)*k);
        y_m = ykNEW((m-1)*k+1:(m)*k);
        f(p,:) = (norm(y_m-y_n)-D(m,n));
end
fnmNEW = f;
