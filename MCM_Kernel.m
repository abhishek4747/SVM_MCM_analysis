function [ yPred, exitflag ] = MCM_Kernel_New(xTrain, yTrain, xTest, yTest, kernel, beta, C)

%%% Define Kernel
if (strcmp(kernel,'RBF') == 1)
    Kernel = @(x,y) exp(-beta * norm(x-y)^2);
elseif (strcmp(kernel,'Linear') == 1)
    Kernel = @(x,y) ( (x*y'));
else
    fprintf(2,'KERNEL INPUT NOT VALID \n\n\n\n\n\n\n\n\n\n\n');
    return;
end


N = size(xTrain,1);
D = size(xTrain,2);


%solve linear program
%   [    lambda,         b,          q,         h]
X = [randn(N,1);randn(1,1); randn(N,1);randn(1,1);];    
f = [zeros(N,1);zeros(1,1);C*ones(N,1);         1;];

LM = zeros(N,N);
    for i=1:N
        for j=1:N
            LM(i,j) = yTrain(i) * Kernel(xTrain(i,:),xTrain(j,:));
        end
    end

LX = zeros(D,N);
    for i=1:D
        for j=1:N
            LX(i,j) = xTrain(j,i);
        end
    end

    
    
%   [lambda,               b,          q,              h]
A = [ LM        ,     yTrain,   eye(N,N),   -1*ones(N,1);
     -LM        ,    -yTrain,  -eye(N,N),     zeros(N,1);];

B = [zeros(N,1);-1*ones(N,1);];


Aeq = [];
Beq = [];

%    [        lambda,      b,             q,     h]
lb = [-inf*ones(N,1);   -inf;    zeros(N,1);     0;];
ub = [ inf*ones(N,1);    inf; inf*ones(N,1);   inf;];

options=optimset('display','final', 'Largescale', 'off', 'Simplex', 'on');

[X, fval, exitflag]  = linprog(f,A,B,Aeq,Beq, lb,ub, [], options);
fprintf('Exitflag : %d \n', exitflag)

lambda = X(1:N,:);
boffset = X(N + 1,:);
q = X(N+1 + 1:N+1 + N,:);
h = X(2*N+1 + 1,:);
nsv = size(find(lambda ~= 0), 1);

yt1 = yTrain;


for i = 1:N
    sumj = boffset;
    for j = 1:N
        sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTrain(i, :));
    end
    yt1(i) = sumj;
end

ntest = size(xTest, 1);
yPred = zeros(ntest, 1);
for i = 1:ntest
    sumj = boffset;
    for j = 1:N
        sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTest(i, :));
    end
    yPred(i) = sumj;
end

yPred = sign(yPred);

% testAcc = sum(yPred.*yTest>0)/size(yTest,1) * 100;
% fprintf(2, 'Training set accuracy: %f Test set accuracy: %f   nSV: %d Training Set size: %d  C =%d  beta=%d \n', trainAcc, testAcc, nsv, size(lambda, 1),C,beta);
% %fprintf(2, '--------------------------------------------\n');
% ncpu = 100;
% %function [ trainAcc, testAcc, nSV, ntrain, ntest, ncpu ] = kmcm1(xTrain, yTrain, xTest, yTest, beta, C)

end
