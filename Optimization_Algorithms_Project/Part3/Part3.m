%% Task 1
X = load('data_opt.csv');

%Compute the distance matrix
max_distance = 0;
max_coordinates = zeros(1,2);
D = zeros(size(X,2),size(X,2));
count = 0;
for i =1:size(X,1)-1
    for j = i+1:size(X,1)
        count =count+1;
        distance = norm(X(i,:)-X(j,:));
        if distance > max_distance %Save the largest distance 
            max_distance = distance;
            max_coordinates = [i,j];
        end
        D(i,j) = distance;
        D(j,i) = D(i,j);
    end
end
%Print D23 and D45
fprintf('D23 = %.4f\nD45 = %.4f\n',D(2,3),D(4,5));

%% Task 3 k = 2
X = load('data_opt.csv');
y_2 = load('yinit2.csv');
y_3 = load('yinit3.csv');
funcsmn_no = 0;
p_vector = zeros(2,19900);
j = 1;
clear g y_store;

K = [2,3];
lambda_o=1;

%m,n to P mapping
for n = 1:size(X,1)-1
    for m = n+1:size(X,1)
        funcsmn_no = funcsmn_no + 1;
        p_vector(:,funcsmn_no) = [n;m];
    end
end

k = K(1);
epsilon = k*10^-2;
dim = k*size(X,1);
lambda = lambda_o;
iteration = 1;
ykactual = y_2;

while(1) 
    [A,derivatives] = CalcA(funcsmn_no,dim,ykactual,k,p_vector,lambda); %Calculate matrix A and the derivatives
    [b,fnm] = Calcb(A,D,ykactual,k,p_vector,lambda); %Calculate matrix B and functions fnm
    
    g(:,iteration) = (2*fnm'*derivatives)';
    y_store(:,iteration) = ykactual; 
    
    if(norm(g(:,iteration))<epsilon)
        break;
    end
    
    yknew = A\b;
    
    fACTUAL = sum(fnm.^2);
    fNEW = sum(CalcfNEW(yknew,p_vector,k,D).^2);
    
    if iteration == 1
        cost_function(iteration) = fACTUAL;
    else
        cost_function(iteration) = fNEW;
    end
    
    if(fNEW<fACTUAL)
        ykactual = yknew;
        fACTUAL = fNEW;
        lambda = 0.7*lambda;    
    else
        lambda = 2*lambda;
    end
    iteration = iteration + 1;
end

cost_function(iteration)=sum(CalcfNEW(yknew,p_vector,k,D).^2);
%Plots for k=2


figure(1)
for i = 1:iteration
    scatter(y_store(1:2:end,i), y_store(2:2:end, i), '.');
end
title('Dimensionality reduction output for k=2');
grid on;

figure(2)
semilogy(0:iteration-1,cost_function, 'Marker', '.', 'MarkerSize', 12);
grid on;
title('Cost Function k=2');

figure(3)
semilogy(0:iteration-1,vecnorm(g,2));
grid on;
title('Norm of the gradient along iterations k=2');

%% Task 3 k = 3
X = load('data_opt.csv');
y_3 = load('yinit3.csv');
funcsmn_no = 0;
p_vector = zeros(2,19900);
j = 1;
K = [2,3];
lambda_o=1;

%m,n to P mapping
for n = 1:size(X,1)-1
    for m = n+1:size(X,1)
        funcsmn_no = funcsmn_no + 1;
        p_vector(:,funcsmn_no) = [n;m];
    end
end

k = K(2);
epsilon = k*10^-2;
dim = k*size(X,1);
lambda = lambda_o;
iteration = 1;
ykactual = y_3;
clear g y_store;

while(1) 
    [A,derivatives] = CalcA(funcsmn_no,dim,ykactual,k,p_vector,lambda); %Calculate matrix A and the derivatives
    [b,fnm] = Calcb(A,D,ykactual,k,p_vector,lambda); %Calculate matrix B and functions fnm
    
    g(:,iteration) = (2*fnm'*derivatives)';
    y_store(:,iteration) = ykactual; 
    
    if(norm(g(:,iteration))<epsilon)
        break;
    end
    
    yknew = A\b;
    
    fACTUAL = sum(fnm.^2);
    fNEW = sum(CalcfNEW(yknew,p_vector,k,D).^2);
    
    if iteration == 1
        cost_function(iteration) = fACTUAL;
    else
        cost_function(iteration) = fNEW;
    end
    
    if(fNEW<fACTUAL)
        ykactual = yknew;
        fACTUAL = fNEW;
        lambda = 0.7*lambda;    
    else
        lambda = 2*lambda;
    end
    fprintf('iteration = %.4f\n',iteration);
    iteration = iteration + 1;
end

cost_function(iteration)=sum(CalcfNEW(yknew,p_vector,k,D).^2);
%Plots for k=3


figure(1)
scatter3(y_store(1:3:end,i), y_store(2:3:end, i), y_store(3:3:end), '.');

title('Dimensionality reduction output for k=3');
grid on;

figure(2)
semilogy(0:iteration-1,cost_function, 'Marker', '.', 'MarkerSize', 12);
grid on;
title('Cost Function k=3');

figure(3)
semilogy(0:iteration-1,vecnorm(g,2));
grid on;
title('Norm of the gradient along iterations k=3');

    



