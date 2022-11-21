%% Task 1

% Constants
clear all;
close all;

T = 80; %Total time
K = 6; %Number of waypoints
p_initial = [0;5]; %Initial position
p_final = [15;-15]; %Final position
v_initial = zeros(2,1); %Initial velocity
v_final = v_initial; %Final velocity
taus = [10 25 30 40 50 60]; %Time instant at each waypoint
U_max = 100; %Maximum force that can be apllied
figure_counter = 1;
iteration_counter = 1;
n_changes = linspace(0,0,7);
mean_dev = n_changes;
w = [10,20,30,30,20,10;10,10,10,0,0,-10]; %Waypoints coordinates
E = [1,0,0,0;0,1,0,0]; %Matrix that retrieves p(t) from x(t)
A = [1,0,0.1,0;0,1,0,0.1;0,0,0.9,0;0,0,0,0.9]; %Matrix that depends on physical constants
B = [0,0;0,0;0.1,0;0,0.1];%Matrix that depends on physical constants

lambda = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3]; % Regularizer parameter
% Solve the optimazation problem 
for iteration_counter = 1:7

    cvx_begin quiet
        variables x(4,T+1) u(2,T)
        expression p(2,6) %Declare p as a variable of cvx type
        
        f_2 = 0;
        f_1 = 0;
                
        for i = 2:T
            f_2 = f_2 + square_pos(norm(u(:,i)- u(:,i-1),2));
        end
        
        
         for i = 1:K
            p(:,i) = E*x(:,taus(i)+1); %Store the positions
            f_1 = f_1 + square_pos(norm(p(:,i)-w(:,i),2)); %Compute the distance penalization
         end
        
        minimize(f_1+lambda(iteration_counter)*f_2);
        
        %Subject to
        x(:,1) == [p_initial(1);p_initial(2);v_initial(1);v_initial(2)];
        x(:,T+1) == [p_final(1);p_final(2);v_final(1);v_final(2)];
        for i=1:T
            norm(u(:,i)) <=  U_max;
            x(:,i+1) == A*x(:,i)+B*u(:,i);
        end
     cvx_end;
     %Plot of the trajectories and the waypoints
     figure(figure_counter)
     hold on;
     plot(x(1,:), x(2,:), 'bo', 'MarkerSize', 3);%Trajectory
     plot(w(1,:), w(2,:),'rs', 'MarkerSize', 12);%Waypoints
     plot(p(1,:), p(2,:),'mo', 'MarkerSize', 10)%Robot position at waypoint times
     legend('Trajectory','Waypoints','Positions at \tau_k');
     title(['Trajectory for \lambda = ',num2str(lambda(iteration_counter))]);
     grid on;
     hold off;
     figure_counter = figure_counter + 1;
     
    %Plot of the applied force (u(t))
    figure(figure_counter)
    hold on;
    plot(u(1,:));%X Control input over time
    plot(u(2,:));%Y Control input over time
    title(['u(t) for \lambda = ',num2str(lambda(iteration_counter))]);
    legend('u1','u2');
    hold off;
    figure_counter = figure_counter + 1;
    
    %Number of times u(t) changes
    n_changes(iteration_counter) = 0;
    
    for i=2:T
        if norm(u(:,i)-u(:,i-1),2)>10^-4
            n_changes(iteration_counter) = n_changes(iteration_counter) + 1;
        end
    end
    
    %Mean deviation from the waypoints
    mean_dev(iteration_counter) = 0;
    sum = 0;
    
     for i = 1:K
            sum = sum + (norm(p(:,i)- w(:,i),2));
     end
     mean_dev(iteration_counter) = (1/K)*sum;
end
%% Task 2

close all;

%load('constants.mat'); %Loads the constants from task 1

% Solve the optimazation problem 
for iteration_counter = 1:7

    cvx_begin quiet
        variables x(4,T+1) u(2,T)
        expression p(2,6) %Declare p as a variable of cvx type
        
        f_2 = 0;
        f_1 = 0;
                
        for i = 2:T
            f_2 = f_2 + (norm(u(:,i)- u(:,i-1),2));
        end
        
        
         for i = 1:K
            p(:,i) = E*x(:,taus(i)+1); %Store the positions
            f_1 = f_1 + square_pos(norm(p(:,i)-w(:,i),2)); %Compute the distance penalization
         end
        
        minimize(f_1+lambda(iteration_counter)*f_2);
        
        %Subject to
        x(:,1) == [p_initial(1);p_initial(2);v_initial(1);v_initial(2)];
        x(:,T+1) == [p_final(1);p_final(2);v_final(1);v_final(2)];
        for i=1:T
            norm(u(:,i)) <=  U_max;
            x(:,i+1) == A*x(:,i)+B*u(:,i);
        end
     cvx_end;
     %Plot of the trajectories and the waypoints
     figure(figure_counter)
     hold on;
     plot(x(1,:), x(2,:), 'bo', 'MarkerSize', 3);%Trajectory
     plot(w(1,:), w(2,:),'rs', 'MarkerSize', 12);%Waypoints
     plot(p(1,:), p(2,:),'mo', 'MarkerSize', 10)%Robot position at waypoint times
     legend('Trajectory','Waypoints','Positions at \tau_k');
     title(['Trajectory for \lambda = ',num2str(lambda(iteration_counter))]);
     grid on;
     hold off;
     figure_counter = figure_counter + 1;
     
    %Plot of the applied force (u(t))
    figure(figure_counter)
    hold on;
    plot(u(1,:));%X Control input over time
    plot(u(2,:));%Y Control input over time
    title(['u(t) for \lambda = ',num2str(lambda(iteration_counter))]);
    legend('u1','u2');
    hold off;
    figure_counter = figure_counter + 1;
    
    %Number of times u(t) changes
    n_changes(iteration_counter) = 0;
    
    for i=2:T
        if norm(u(:,i)-u(:,i-1),2)>10^-4
            n_changes(iteration_counter) = n_changes(iteration_counter) + 1;
        end
    end
    
    %Mean deviation from the waypoints
    mean_dev(iteration_counter) = 0;
    sum = 0;
    
     for i = 1:K
            sum = sum + (norm(p(:,i)- w(:,i),2));
     end
     mean_dev(iteration_counter) = (1/K)*sum;
end
%% Task 3

close all;

% load('constants.mat'); %Loads the constants from task 1

% Solve the optimazation problem 
for iteration_counter = 1:7

    cvx_begin quiet
        variables x(4,T+1) u(2,T)
        expression p(2,6) %Declare p as a variable of cvx type
        
        f_2 = 0;
        f_1 = 0;
                
        for i = 2:T
            f_2 = f_2 + (norm(u(:,i)- u(:,i-1),1));
        end
        
        
         for i = 1:K
            p(:,i) = E*x(:,taus(i)+1); %Store the positions
            f_1 = f_1 + square_pos(norm(p(:,i)-w(:,i),2)); %Compute the distance penalization
         end
        
        minimize(f_1+lambda(iteration_counter)*f_2);
        
        %Subject to
        x(:,1) == [p_initial(1);p_initial(2);v_initial(1);v_initial(2)];
        x(:,T+1) == [p_final(1);p_final(2);v_final(1);v_final(2)];
        for i=1:T
            norm(u(:,i)) <=  U_max;
            x(:,i+1) == A*x(:,i)+B*u(:,i);
        end
     cvx_end;
     %Plot of the trajectories and the waypoints
     figure(figure_counter)
     hold on;
     plot(x(1,:), x(2,:), 'bo', 'MarkerSize', 3);%Trajectory
     plot(w(1,:), w(2,:),'rs', 'MarkerSize', 12);%Waypoints
     plot(p(1,:), p(2,:),'mo', 'MarkerSize', 10)%Robot position at waypoint times
     legend('Trajectory','Waypoints','Positions at \tau_k');
     title(['Trajectory for \lambda = ',num2str(lambda(iteration_counter))]);
     grid on;
     hold off;
     figure_counter = figure_counter + 1;
     
    %Plot of the applied force (u(t))
    figure(figure_counter)
    hold on;
    plot(u(1,:));%X Control input over time
    plot(u(2,:));%Y Control input over time
    title(['u(t) for \lambda = ',num2str(lambda(iteration_counter))]);
    legend('u1','u2');
    hold off;
    figure_counter = figure_counter + 1;
    
    %Number of times u(t) changes
    n_changes(iteration_counter) = 0;
    
    for i=2:T
        if norm(u(:,i)-u(:,i-1),2)>10^-4
            n_changes(iteration_counter) = n_changes(iteration_counter) + 1;
        end
    end
    
    %Mean deviation from the waypoints
    mean_dev(iteration_counter) = 0;
    sum = 0;
    
     for i = 1:K
            sum = sum + (norm(p(:,i)- w(:,i),2));
     end
     mean_dev(iteration_counter) = (1/K)*sum;
end 

%% Task 5

clear all;
close all;

%Values initialization
c = [0.6332 -0.0054 2.3322 4.4526 6.1752;-3.2012 -1.7104 -0.7620 3.1001 4.2391];
R = [2.2727,0.7281,1.3851,1.8191,1.0895];
t = [0,1,1.5,3,4.5];
x_critical = [6;10];
t_critical = 8;
K = 5;

% Solve the optimazation problem 

cvx_begin quiet
variables p_0(2,1) v(2,1)
expression P(2,5)
        
minimize(norm((p_0+t_critical*v)-x_critical));
        
%Subject to
for k = 1:K 
    P(:,k) = p_0+t(k)*v;
    norm(p_0+t(k)*v-c(:,k)) <= R(k);
end
cvx_end;

p_critical = p_0+t_critical*v;
figure

hold on
for i=1:K
    drawCircle(c(1,i), c(2,i), R(i));
    text(c(1,i)-0.1,c(2,i)+0.1,num2str(i));
    plot(p_0(1)+v(1)*t(i),p_0(2)+v(2)*t(i),'rs', 'MarkerSize', 8, 'LineWidth', 2);
end

plot(p_critical(1),p_critical(2),'rs', 'MarkerSize', 8, 'LineWidth', 2);

plot(x_critical(1), x_critical(2), '.k', 'MarkerSize', 30);
text(x_critical(1)-0.2, x_critical(2)-0.6,texlabel('x^*'));
axis equal
grid on

%% Task 6

clear all;
close all;

%Values initialization
c = [0.6332 -0.0054 2.3322 4.4526 6.1752;-3.2012 -1.7104 -0.7620 3.1001 4.2391];
R = [2.2727,0.7281,1.3851,1.8191,1.0895];
t = [0,1,1.5,3,4.5];
x_critical = [6;10];
t_critical = 8;
K = 5;

% Solve the optimazation problem 

cvx_begin quiet
variables p_0(2,1) v(2,1)
expressions P(2,5) p_critical(2,1)
p_critical = p_0+t_critical*v

minimize(p_critical(1,1));     

%Subject to
for k = 1:K 
    P(:,k) = p_0+t(k)*v;
    norm(p_0+t(k)*v-c(:,k)) <= R(k);
end
x_minimo = p_critical(1,1);
cvx_end;

cvx_begin quiet
variables p_0(2,1) v(2,1)
expressions P(2,5) p_critical(2,1)
p_critical = p_0+t_critical*v

minimize(p_critical(2,1));     

%Subject to
for k = 1:K 
    P(:,k) = p_0+t(k)*v;
    norm(p_0+t(k)*v-c(:,k)) <= R(k);
end
cvx_end;

y_minimo = p_critical(2,1);

cvx_begin quiet
variables p_0(2,1) v(2,1)
expressions P(2,5) p_critical(2,1)
p_critical = p_0+t_critical*v

maximize(p_critical(1,1));     

%Subject to
for k = 1:K 
    P(:,k) = p_0+t(k)*v;
    norm(p_0+t(k)*v-c(:,k)) <= R(k);
end
cvx_end;

x_maximo = p_critical(1,1);

cvx_begin quiet
variables p_0(2,1) v(2,1)
expressions P(2,5) p_critical(2,1)
p_critical = p_0+t_critical*v

maximize(p_critical(2,1));     

%Subject to
for k = 1:K 
    P(:,k) = p_0+t(k)*v;
    norm(p_0+t(k)*v-c(:,k)) <= R(k);
end
cvx_end;

y_maximo = p_critical(2,1);

figure;
hold on;
line([x_minimo,x_minimo],[y_minimo,y_maximo]);
line([x_minimo,x_maximo],[y_maximo,y_maximo]);
line([x_maximo,x_maximo],[y_maximo,y_minimo]);
line([x_maximo,x_minimo],[y_minimo,y_minimo]);

for i=1:K
    drawCircle(c(1,i), c(2,i), R(i));
    text(c(1,i)-0.1,c(2,i)+0.1,num2str(i));
end

plot(x_critical(1), x_critical(2), '.k', 'MarkerSize', 30);
text(x_critical(1)-0.2, x_critical(2)-0.6,texlabel('x^*'));
axis equal
grid on





