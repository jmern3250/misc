load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);

%% b i 

close all 
theta_bi = (lambdas'*lambdas)\(lambdas'*train_qso(1,:)');
qpred = theta_bi*lambdas;
figure
hold on 
plot(lambdas, train_qso(1,:)');
plot(lambdas, qpred, 'LineWidth', 3)
title('Non-weighted linear regression')
xlabel('Wavelength (angstrom)')
ylabel('Flux')
legend('Data', 'Linear Fit')

%% b ii 

close all 
[m, n] = size(train_qso); 
tau = 5;

theta_b2 = zeros(n,1);
for i = 1:n
    w = exp(-(lambdas - lambdas(i)).^2/(2*tau^2));
    W = diag(w);
    theta_b2(i) = (lambdas'*W*lambdas)\(lambdas'*W*train_qso(1,:)');
end

qpred = theta_b2.*lambdas;
figure
hold on 
plot(lambdas, train_qso(1,:)');
plot(lambdas, qpred, 'LineWidth', 3)
title('Non-weighted linear regression')
xlabel('Wavelength (angstrom)')
ylabel('Flux')
legend('Data', 'Linear Fit')

%% b iii 
close all 
[m, n] = size(train_qso); 
tau = 1;

theta_b2 = zeros(n,1);
for i = 1:n
    w = exp(-(lambdas - lambdas(i)).^2/(2*tau^2));
    W = diag(w);
    theta_b2(i) = (lambdas'*W*lambdas)\(lambdas'*W*train_qso(1,:)');
end

qpred = theta_b2.*lambdas;
figure
hold on 
plot(lambdas, train_qso(1,:)');
plot(lambdas, qpred, 'LineWidth', 3)
title('Non-weighted linear regression')
xlabel('Wavelength (angstrom)')
ylabel('Flux')

tau = 10;

theta_b2 = zeros(n,1);
for i = 1:n
    w = exp(-(lambdas - lambdas(i)).^2/(2*tau^2));
    W = diag(w);
    theta_b2(i) = (lambdas'*W*lambdas)\(lambdas'*W*train_qso(1,:)');
end

qpred = theta_b2.*lambdas;
plot(lambdas, qpred, 'LineWidth', 3)

tau = 100;

theta_b2 = zeros(n,1);
for i = 1:n
    w = exp(-(lambdas - lambdas(i)).^2/(2*tau^2));
    W = diag(w);
    theta_b2(i) = (lambdas'*W*lambdas)\(lambdas'*W*train_qso(1,:)');
end

qpred = theta_b2.*lambdas;
plot(lambdas, qpred, 'LineWidth', 3)

tau = 1000;

theta_b2 = zeros(n,1);
for i = 1:n
    w = exp(-(lambdas - lambdas(i)).^2/(2*tau^2));
    W = diag(w);
    theta_b2(i) = (lambdas'*W*lambdas)\(lambdas'*W*train_qso(1,:)');
end

qpred = theta_b2.*lambdas;
plot(lambdas, qpred, 'LineWidth', 3)
legend('Data', 'Tau=1', 'Tau=10', 'Tau=100', 'Tau=1000')

%% 5c i
close all 

[m, n] = size(train_qso);
[mt, ~] = size(test_qso);
tau = 5;
train_smth = zeros(m,n);
test_smth = zeros(mt,n);

for i = 1:m
    theta = zeros(n,1);
    for j = 1:n
        w = exp(-(lambdas - lambdas(j)).^2/(2*tau^2));
        W = diag(w);
        theta(j) = (lambdas'*W*lambdas)\(lambdas'*W*train_qso(i,:)');
    end
    train_smth(i,:) = theta.*lambdas;
end

for i = 1:mt
    theta = zeros(n,1);
    for j = 1:n
        w = exp(-(lambdas - lambdas(j)).^2/(2*tau^2));
        W = diag(w);
        theta(j) = (lambdas'*W*lambdas)\(lambdas'*W*test_qso(i,:)');
    end
    test_smth(i,:) = theta.*lambdas;
end

%% 5c ii 

error = zeros(m, 1);
idx_l = 50;
idx_r = 151;

for i = 1:m
    f = train_smth(i, :);
    fest = fleft(f, train_smth, 3, idx_r, idx_l);
    error(i) = f_dist(fest, f(1:idx_l));
end

avg_error_train = mean(error);

%% 5c iii 

error = zeros(mt, 1);
idx_l = 50;
idx_r = 151;

for i = 1:mt
    f = test_smth(i, :);
    fest = fleft(f, train_smth, 3, idx_r, idx_l);
    error(i) = f_dist(fest, f(1:idx_l));
end

avg_error_test = mean(error);

f = test_smth(1, :);
figure 
hold on
plot(lambdas, test_smth(1,:))
plot(lambdas(1:idx_l), fleft(f, train_smth, 3, idx_r, idx_l), 'o')

f = test_smth(6, :);
figure 
hold on
plot(lambdas, test_smth(6,:))
plot(lambdas(1:idx_l), fleft(f, train_smth, 3, idx_r, idx_l), 'o')

%% Fxns

function f_est = fleft(f, F, k, idx_r, idx_l) 
    [Fnbr, Dnbr, h] = fnbrs(F, k, f, idx_r);
    num = zeros(1, idx_l); 
    den = 0.0; 
    for i = 1:k
        num = num + ker(Dnbr(i)/h)*Fnbr(i,1:idx_l);
        den = den + ker(Dnbr(i)/h);
    end
    f_est = num/den; 
end

function [Fnbr, Dnbr, h] = fnbrs(F, k, f, idx_r)
    [m, n] = size(F); 
    dist = zeros(m-idx_r,1); 
    for i = idx_r:m
        dist(i) = f_dist(f(idx_r:n), F(i,idx_r:n));
    end
    h = max(dist); 
    [Ds, Is] = sort(dist);
    Idxr = Is(2:k+1) + idx_r - 1;
    Fnbr = F(Idxr,:); 
    Dnbr = Ds(2:k+1); 
end

function x = ker(t)
    x = max(1-t, 0);
end


function dist = f_dist(f1, f2)
    dist = 0;
    [~, n] = size(f1);
    for i = 1:n
        dist = dist + (f1(i) - f2(i))^2;
    end
end