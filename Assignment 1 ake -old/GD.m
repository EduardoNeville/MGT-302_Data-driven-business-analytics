clear all; %#ok<CLALL> 
close all;

load("weighttrain.mat");

for i = 1:5
    x(:,i) = ( x(:,i) - mean(x(:,i)) ) / std(x(:,i)); %#ok<SAGROW> 
end
N = 100;
alphas = logspace(-4,2,N);

for i = 1:N

    alpha = alphas(i);

    Theta = zeros(6,1)';
    Theta_old = Theta;
    Theta = Theta - alpha * cost_grad(x,y,Theta);
    Iterration = 1;

    while not(all( abs(Theta-Theta_old ) < 1e-7 )) 
        
        Theta_old = Theta;
        Theta = Theta - alpha * cost_grad(x,y,Theta);
        Iterration = Iterration +1;
        
        if any(isnan(Theta))
            disp(num2str(alpha) +' : NAN');
            break;
        end
        if Iterration > 1e6
            break;
        end
    end
    costs(i) = cost(x,y,Theta);
    alphas(i) = alpha;
    Iterrations(i) = Iterration;
    disp(alpha);
end
%%
fs = 25; lw = 2; ms = 15;

figure('Name','alpha vs cost ');
loglog(alphas,costs,'r-',LineWidth=lw);
grid on;
xlabel('$\alpha$',FontSize=fs,Interpreter='latex');
ylabel('Cost',FontSize=fs,Interpreter='latex');

figure('Name','alpha vs Iterrations ');
loglog(alphas,Iterrations,'r-',LineWidth=lw);
grid on;
xlabel('$\alpha$',FontSize=fs,Interpreter='latex');
ylabel('Iterrations',FontSize=fs,Interpreter='latex');


%h = @(X,T) T(1) + sum(X.*T(1,2:6),2);
%cost = @(X,Y,T) (1/2*length(Y)) * sum( (h(X,T)-Y).^2);
%cost_gradient = @(X,Y,T) (1/length(Y)) * sum( [zeros(200,1)+1,X].*(h(X,T)-Y) );


function result = h(X,T)
    result = T(1) + T(2)*X(1) + T(3)*X(2) + T(4)*X(3) + T(5)*X(4) + T(6)*X(5);
end

function sum_ = cost(X,Y,T)
    sum_ = 0;
    N = length(Y);
    for i = 1:N
        sum_ = sum_ + (h(X(i,:),T) - Y(i,1))^2;
    end
    sum_ = sum_/(2*N);
end

function sum_ = cost_grad(X,Y,T)
    sum_ = 0;
    N = length(Y);
    for i = 1:N
        sum_ = sum_ + [1,X(i,:)]*(h(X(i,:),T)-Y(i,1));
    end
    sum_ = sum_/(N);
end

