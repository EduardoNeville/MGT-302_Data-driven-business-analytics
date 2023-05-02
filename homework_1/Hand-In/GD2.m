clear all; %#ok<CLALL> 
close all;

load("weighttrain.mat");

for i = 1:5
    x(:,i) = ( x(:,i) - mean(x(:,i)) ) / std(x(:,i)); %#ok<SAGROW> 
end
N = 40;
alpha = 0.01; %Optimal alpha
Iterrations = 10:10:400;
for i = 1:N

    Theta = zeros(6,1)';
    Theta_old = Theta;
    Theta = Theta - alpha * cost_grad(x,y,Theta);
    Iterration = 1;

    while true 
        
        Theta_old = Theta;
        Theta = Theta - alpha * cost_grad(x,y,Theta);
        Iterration = Iterration +1;
        
        if any(isnan(Theta))
            disp(num2str(alpha) +" : NAN");
            break;
        end
        if Iterration > Iterrations(i)
            break;
        end
    end
    costs(i) = cost(x,y,Theta);
    disp(Iterrations(i));
end
%%
fs = 25; lw = 2; ms = 15;

figure('Name','alpha vs cost ');
plot(Iterrations,costs,'r-',LineWidth=lw);
grid on;
xlabel('\# of Iterrations',FontSize=fs,Interpreter='latex');
ylabel('Cost',FontSize=fs,Interpreter='latex');
ylim([min(costs),max(costs)]);

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

