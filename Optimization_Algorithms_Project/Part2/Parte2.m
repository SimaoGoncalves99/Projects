%% Gradient method

%% Task 2
load('data1.mat');
figure_counter = 1;
epsilon = 10^-6; %Stopping criterion
alpha = 1; %Backtracking routine parameter
gama = 10^-4; %Backtracking routine parameter
beta = 1/2; %Backtracking routine parameter
n = size(X,1); %Number of features
K = size(X,2); %Number of data points
A=[X(:,:); -1*ones(1, K)]; %Matrix A
x=[-1*ones(n,1); 0]; %Vector x
step = alpha; %Step size
% Gradient Descent algorithm 
i = 0;
while(1)
        
    grad_phi = (1/K)*(((exp(A'*x))./(1+exp(A'*x)))-Y');
    grad_f = A*grad_phi;
    g(:,i+1) = grad_f; 
    norms(i+1) = norm(g(:,i+1));
    
    if norms(i+1) < epsilon
        break;
    end
    
    d(:,i+1) = -g(:,i+1);
    loop = 1;
   
    while(loop == 1)
        fnext = sum((1/K)*(log(1+exp(A'*(x+step*d(:,i+1))))+Y'.*(-A'*(x+step*d(:,i+1)))));
        fatual = sum((1/K)*(log(1+exp(A'*x))+Y'.*(-A'*x)));
        if  fnext < fatual + gama*grad_f'*(step*d(:,i+1))
            break;
        else
            step = beta*step;
            if step < 0
                disp('Error, step cannot be a negative or null value');
                break;
            else
                continue
            end
        end 
    end
    x = x+step*d(:,i+1);
    step = alpha;
    i = i+1; 
end
    

figure(figure_counter)
hold on;
for j = 1:150
    if Y(j) == 1 
        scatter(X(1,j),X(2,j),'blue','filled') %Dataset
    else
        scatter(X(1,j),X(2,j),'red','filled')
    end
end

x1 = linspace(min(X(1,:)),max(X(1,:)),150);
plot(x1,(x(3,1)-x(1,1)*x1)/(x(2,1)), '--'); %Hyperplane
title('Dataset and Hyperplane (Data1.mat)');
legend('y = 1','y = 0');
hold off;
figure_counter = figure_counter + 1;

figure(figure_counter);
semilogy(norms);
grid on;
title('Norm of the gradient along iterations (Data1.mat)');
legend('||\nablaf(s_k,r_k)||');
figure_counter = figure_counter + 1;

%% Task 3

clear;
close all;

load('data2.mat');
figure_counter = 1;
epsilon = 10^-6; %Stopping criterion
alpha = 1; %Backtracking routine parameter
gama = 10^-4; %Backtracking routine parameter
beta = 1/2; %Backtracking routine parameter
n = size(X,1); %Number of features
K = size(X,2); %Number of data points
A=[X(:,:); -1*ones(1, K)]; %Matrix A
x=[-1*ones(n,1); 0]; %Vector x
step = alpha; %Step size
% Gradient Descent algorithm 
i = 0;
while(1)
        
    grad_phi = (1/K)*(((exp(A'*x))./(1+exp(A'*x)))-Y');
    grad_f = A*grad_phi;
    g(:,i+1) = grad_f; 
    norms(i+1) = norm(g(:,i+1));
    
    if norms(i+1) < epsilon
        break;
    end
    
    d(:,i+1) = -g(:,i+1);
    loop = 1;
   
    while(loop == 1)
        fnext = sum((1/K)*(log(1+exp(A'*(x+step*d(:,i+1))))+Y'.*(-A'*(x+step*d(:,i+1)))));
        fatual = sum((1/K)*(log(1+exp(A'*x))+Y'.*(-A'*x)));
        if  fnext < fatual + gama*grad_f'*(step*d(:,i+1))
            break;
        else
            step = beta*step;
            if step < 0
                disp('Error, step cannot be a negative or null value');
                break;
            else
                continue
            end
        end 
    end
    x = x+step*d(:,i+1);
    step = alpha;
    i = i+1; 
end
    

figure(figure_counter)
hold on;
for j = 1:150
    if Y(j) == 1 
        scatter(X(1,j),X(2,j),'blue','filled') %Dataset
    else
        scatter(X(1,j),X(2,j),'red','filled')
    end
end

x1 = linspace(min(X(1,:)),max(X(1,:)),150);
plot(x1,(x(3,1)-x(1,1)*x1)/(x(2,1)), '--'); %Hyperplane
title('Dataset and Hyperplane (Data2.mat)');
legend('y = 1','y = 0');
hold off;
figure_counter = figure_counter + 1;

figure(figure_counter);
semilogy(norms);
grid on;
title('Norm of the gradient along iterations(Data2.mat)');
legend('||\nablaf(s_k,r_k)||');
figure_counter = figure_counter + 1;

%% Task 4
%Data 3

clear;
close all;

load('data3.mat');
epsilon = 10^-6; %Stopping criterion
alpha = 1; %Backtracking routine parameter
gama = 10^-4; %Backtracking routine parameter
beta = 1/2; %Backtracking routine parameter
n = size(X,1); %Number of features
K = size(X,2); %Number of data points
A = [X(:,:); -1*ones(1, K)]; %Matrix A
x = [-1*ones(n,1); 0]; %Vector x
step = alpha; %Step size
% Gradient Descent algorithm 
i = 0;
while(1)
        
    grad_phi = (1/K)*(((exp(A'*x))./(1+exp(A'*x)))-Y');
    grad_f = A*grad_phi;
    g(:,i+1) = grad_f; 
    norms(i+1) = norm(g(:,i+1));
    
    if norms(i+1) < epsilon
        break;
    end
    
    d(:,i+1) = -g(:,i+1);
    loop = 1;
   
    while(loop == 1)
        fnext = sum((1/K)*(log(1+exp(A'*(x+step*d(:,i+1))))+Y'.*(-A'*(x+step*d(:,i+1)))));
        fatual = sum((1/K)*(log(1+exp(A'*x))+Y'.*(-A'*x)));
        if  fnext < fatual + gama*grad_f'*(step*d(:,i+1))
            break;
        else
            step = beta*step;
            if step < 0
                disp('Error, step cannot be a negative or null value');
                break;
            else
                continue
            end
        end 
    end
    x = x+step*d(:,i+1);
    step = alpha;
    i = i+1; 
end
figure(1);
semilogy(norms);
grid on;
title('Norm of the gradient along iterations(Data3.mat)');
legend('||\nablaf(s_k,r_k)||');

%% Data 4
clear;
close all;

load('data4.mat');
epsilon = 10^-6; %Stopping criterion
alpha = 1; %Backtracking routine parameter
gama = 10^-4; %Backtracking routine parameter
beta = 1/2; %Backtracking routine parameter
n = size(X,1); %Number of features
K = size(X,2); %Number of data points
A = [X(:,:); -1*ones(1, K)]; %Matrix A
x = [-1*ones(n,1); 0]; %Vector x
step = alpha; %Step size
% Gradient Descent algorithm 
i = 0;
while(1)
        
    grad_phi = (1/K)*(((exp(A'*x))./(1+exp(A'*x)))-Y');
    grad_f = A*grad_phi;
    g(:,i+1) = grad_f; 
    norms(i+1) = norm(g(:,i+1));
    
    if norms(i+1) < epsilon
        break;
    end
    
    d(:,i+1) = -g(:,i+1);
    loop = 1;
   
    while(loop == 1)
        fnext = sum((1/K)*(log(1+exp(A'*(x+step*d(:,i+1))))+Y'.*(-A'*(x+step*d(:,i+1)))));
        fatual = sum((1/K)*(log(1+exp(A'*x))+Y'.*(-A'*x)));
        if  fnext < fatual + gama*grad_f'*(step*d(:,i+1))
            break;
        else
            step = beta*step;
            if step < 0
                disp('Error, step cannot be a negative or null value');
                break;
            else
                continue
            end
        end 
    end
    x = x+step*d(:,i+1);
    step = alpha;
    i = i+1; 
end
figure(1);
semilogy(norms);
grid on;
title('Norm of the gradient along iterations(Data4.mat)');
legend('||\nablaf(s_k,r_k)||');

%% Task 6

%% Data 1
clear;
close all;

load('data1.mat');
epsilon = 10^-6; %Stopping criterion
alpha = 1; %Backtracking routine parameter
gama = 10^-4; %Backtracking routine parameter
beta = 1/2; %Backtracking routine parameter
n = size(X,1); %Number of features
K = size(X,2); %Number of data points
A = [X(:,:); -1*ones(1, K)]; %Matrix A
x = [-1*ones(n,1); 0]; %Vector x
step = alpha; %Step size

i = 0;

while(1)
    grad_phi = (1/K)*(((exp(A'*x))./(1+exp(A'*x)))-Y');
    grad_f = A*grad_phi;
    g(:,i+1) = grad_f;
    norms(i+1) = norm(g(:,i+1));
    
    if(norms(i+1)<epsilon)
        break;
    end
    D = diag((exp(A'*x)./((1+(exp(A'*x))).^2)));
    d(:,i+1) = -(A*D*(A')./K)\g(:,i+1);
    
    loop = 1;
    while(loop == 1)
        fnext = sum((1/K)*(log(1+exp(A'*(x+step*d(:,i+1))))+Y'.*(-A'*(x+step*d(:,i+1)))));
        fatual = sum((1/K)*(log(1+exp(A'*x))+Y'.*(-A'*x)));
        if  fnext < fatual + gama*grad_f'*(step*d(:,i+1))
            break;
        else
            step = beta*step;
        end 
    end
    x = x+step*d(:,i+1);
    store(i+1) = step;
    step = alpha;
    i = i+1;
end

figure(1);
semilogy(norms);
grid on;
title('Newton: Norm of the gradient along iterations(Data1.mat)');
legend('||\nablaf(s_k,r_k)||');

%Plot the stepsizes along iterations
figure(2)
stem(store);
xlabel('k');
title('\alpha_k Newton method');

%% Data 2
clear;
close all;

load('data2.mat');
epsilon = 10^-6; %Stopping criterion
alpha = 1; %Backtracking routine parameter
gama = 10^-4; %Backtracking routine parameter
beta = 1/2; %Backtracking routine parameter
n = size(X,1); %Number of features
K = size(X,2); %Number of data points
A = [X(:,:); -1*ones(1, K)]; %Matrix A
x = [-1*ones(n,1); 0]; %Vector x
step = alpha; %Step size

i = 0;

while(1)
    grad_phi = (1/K)*(((exp(A'*x))./(1+exp(A'*x)))-Y');
    grad_f = A*grad_phi;
    g(:,i+1) = grad_f;
    norms(i+1) = norm(g(:,i+1));
    
    if(norms(i+1)<epsilon)
        break;
    end
    D = diag((exp(A'*x)./((1+(exp(A'*x))).^2)));
    d(:,i+1) = -(A*D*(A')./K)\g(:,i+1);
    
    loop = 1;
    while(loop == 1)
        fnext = sum((1/K)*(log(1+exp(A'*(x+step*d(:,i+1))))+Y'.*(-A'*(x+step*d(:,i+1)))));
        fatual = sum((1/K)*(log(1+exp(A'*x))+Y'.*(-A'*x)));
        if  fnext < fatual + gama*grad_f'*(step*d(:,i+1))
            break;
        else
            step = beta*step;
        end 
    end
    x = x+step*d(:,i+1);
    store(i+1) = step;
    step = alpha;
    i = i+1;
end

figure(1);
semilogy(norms);
grid on;
title('Newton: Norm of the gradient along iterations(Data2.mat)');
legend('||\nablaf(s_k,r_k)||');

%Plot the stepsizes along iterations
figure(2)
stem(store);
xlabel('k');
title('\alpha_k Newton method');

%% Data 3
clear;
close all;

load('data3.mat');
epsilon = 10^-6; %Stopping criterion
alpha = 1; %Backtracking routine parameter
gama = 10^-4; %Backtracking routine parameter
beta = 1/2; %Backtracking routine parameter
n = size(X,1); %Number of features
K = size(X,2); %Number of data points
A = [X(:,:); -1*ones(1, K)]; %Matrix A
x = [-1*ones(n,1); 0]; %Vector x
step = alpha; %Step size

i = 0;

while(1)
    grad_phi = (1/K)*(((exp(A'*x))./(1+exp(A'*x)))-Y');
    grad_f = A*grad_phi;
    g(:,i+1) = grad_f;
    norms(i+1) = norm(g(:,i+1));
    
    if(norms(i+1)<epsilon)
        break;
    end
    D = diag((exp(A'*x)./((1+(exp(A'*x))).^2)));
    d(:,i+1) = -(A*D*(A')./K)\g(:,i+1);
    
    loop = 1;
    while(loop == 1)
        fnext = sum((1/K)*(log(1+exp(A'*(x+step*d(:,i+1))))+Y'.*(-A'*(x+step*d(:,i+1)))));
        fatual = sum((1/K)*(log(1+exp(A'*x))+Y'.*(-A'*x)));
        if  fnext < fatual + gama*grad_f'*(step*d(:,i+1))
            break;
        else
            step = beta*step;
        end 
    end
    x = x+step*d(:,i+1);
    store(i+1) = step;
    step = alpha;
    i = i+1;
end

figure(1);
semilogy(norms);
grid on;
title('Newton: Norm of the gradient along iterations(Data3.mat)');
legend('||\nablaf(s_k,r_k)||');

%Plot the stepsizes along iterations
figure(2)
stem(store);
xlabel('k');
title('\alpha_k Newton method');

%% Data 4
clear;
close all;

load('data4.mat');
epsilon = 10^-6; %Stopping criterion
alpha = 1; %Backtracking routine parameter
gama = 10^-4; %Backtracking routine parameter
beta = 1/2; %Backtracking routine parameter
n = size(X,1); %Number of features
K = size(X,2); %Number of data points
A = [X(:,:); -1*ones(1, K)]; %Matrix A
x = [-1*ones(n,1); 0]; %Vector x
step = alpha; %Step size

i = 0;

while(1)
    grad_phi = (1/K)*(((exp(A'*x))./(1+exp(A'*x)))-Y');
    grad_f = A*grad_phi;
    g(:,i+1) = grad_f;
    norms(i+1) = norm(g(:,i+1));
    
    if(norms(i+1)<epsilon)
        break;
    end
    D = diag((exp(A'*x)./((1+(exp(A'*x))).^2)));
    d(:,i+1) = -(A*D*(A')./K)\g(:,i+1);
    
    loop = 1;
    while(loop == 1)
        fnext = sum((1/K)*(log(1+exp(A'*(x+step*d(:,i+1))))+Y'.*(-A'*(x+step*d(:,i+1)))));
        fatual = sum((1/K)*(log(1+exp(A'*x))+Y'.*(-A'*x)));
        if  fnext < fatual + gama*grad_f'*(step*d(:,i+1))
            break;
        else
            step = beta*step;
        end 
    end
    x = x+step*d(:,i+1);
    store(i+1) = step;
    step = alpha;
    i = i+1;
end

figure(1);
semilogy(norms);
grid on;
title('Newton: Norm of the gradient along iterations(Data4.mat)');
legend('||\nablaf(s_k,r_k)||');

%Plot the stepsizes along iterations
figure(2)
stem(store);
xlabel('k');
title('\alpha_k Newton method');



