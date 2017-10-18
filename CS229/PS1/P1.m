X_ = readtable('logistic_x.txt');
Y = csvread('logistic_y.txt');
% Y = table2array(Y_);

[m, n] = size(X_);
n = n + 1;
X = ones(m, n);
X(:,1:2) = table2array(X_);


theta = zeros(3,1);

grad = 1000.0;
eps = 1e-5;

jac = Jacob(theta, X, Y); 
H = Hess(theta, X, Y); 

its = 0;
while grad > eps
    theta = theta - (jac/H)';
    jac = Jacob(theta, X, Y); 
    H = Hess(theta, X, Y);
    grad = norm(jac);
    its = its + 1;
end

loss = Jlog(theta, X, Y);
x1 = 0:0.1:8;
x2 = ((0.5 - theta(3)) - theta(1)*x1)/theta(2);

figure 
hold on
for i = 1:m
    if Y(i) == 1
        plot(X(i,1), X(i,2), 'or')
    else
        plot(X(i,1), X(i,2), 'xb')
    end
end

plot(x1, x2)

title('Logistic Regression on Dataset')
xlabel('X1')
ylabel('X2')

function loss = Jlog(theta, X, Y)

[m, ~] = size(X);

loss = 0.0;
for i = 1:m
    h = (1 + exp(-Y(i)*theta'*X(i,:)'))^-1;
    loss = loss + log(h); 
end
loss = -1/m*loss; 
end

function jac = Jacob(theta, X, Y)
    [m, n] = size(X);
    jac = zeros(1, n);
    for i = 1:m
       num = Y(i)*X(i,:)*exp(-Y(i)*theta'*X(i,:)');
       den = 1 + exp(-Y(i)*theta'*X(i,:)');
       jac = jac + num./den;
    end
    jac = -1/m*jac;
end

function H = Hess(theta, X, Y)
    [m, n] = size(X);
    H = zeros(n, n);
    for i = 1:m
       num = -exp(-Y(i)*theta'*X(i,:)');
       den = (1 + exp(-Y(i)*theta'*X(i,:)'))^2;
       mat = X(i,:)'*X(i,:);
       H = H + Y(i)^2*mat*num./den;
    end
    H = -1/m*H;
end