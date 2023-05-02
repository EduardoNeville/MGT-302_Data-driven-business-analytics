clear all; %#ok<CLALL> 
close all;

load("Q1-Dataset/weighttrain.mat");

for i = 1:5
    x(:,i) = ( x(:,i) - mean(x(:,i)) ) / std(x(:,i)); %#ok<SAGROW> 
end

alpha = 0.01; %Optimal alpha
IterrationMax = 4000;


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
    if Iterration > IterrationMax
        break;
    end
end
%Unable 
clear x y
load("Q1-Dataset/weighttest.mat");

for i = 1:5
    x(:,i) = ( x(:,i) - mean(x(:,i)) ) / std(x(:,i)); %#ok<SAGROW> 
end

disp(cost(x,y,Theta));


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

