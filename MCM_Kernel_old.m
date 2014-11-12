function [ yPred ] = MCM_Kernel(xTrain, yTrain, xTest, yTest, kernel, beta, C)

%%% Define Kernel
if (strcmp(kernel,'RBF') == 1)
    Kernel = @(x,y,beta) exp(-beta * norm(x-y)^2);
elseif (strcmp(kernel,'Linear') == 1)
    Kernel = @(x,y,beta) ( (x*y'));
else
    fprintf(2,'KERNEL INPUT NOT VALID \n\n\n\n\n\n\n\n\n\n\n');
    return;
end


N = size(xTrain,1);
%solve linear program
X = [randn(N,1);randn(1,1);randn(N,1);randn(1,1)];       %[lambda, b, q, g]
f = [zeros(N,1);zeros(1,1);0*ones(N,1);1]; %%%

nrows = N;
ncols = 2 * N + 2;
leqcons = 2 * nrows; % <= constraints
A = zeros(leqcons, ncols);
b = zeros(leqcons, 1);

for i = 1:nrows
    for j = 1:nrows

        A(i, j) = yTrain(i) * Kernel(xTrain(i, :), xTrain(j, :), beta);
       
    end
    
    A(i, nrows + 1) = yTrain(i); % b
    A(i, ncols) = -1; % g
    b(i,:) = 0;
end

offset = nrows;
for i = 1:nrows
    for j = 1: nrows
        A(offset + i, j) = -yTrain(i) * Kernel(xTrain(i, :), xTrain(j, :), beta);
    end
    A(offset + i, nrows + 1) = -yTrain(i); % b
    A(offset + i, nrows + 1 + i) = 0; %%%% % q_i
    A(offset + i, ncols) = 0; % g
    b(offset + i) = -1;
end




Aeq = [];
beq = [];

lb = [-inf*ones(N,1);-inf*ones(1,1);zeros(N,1);0]; % lambda, b, q, g
ub = [ inf*ones(N,1); inf*ones(1,1);inf*ones(N,1);inf];

options=optimset('display','final', 'Largescale', 'off', 'Simplex', 'on');

[X]  = linprog(f,A,b,Aeq,beq, lb,ub, [], options);
lambda = X(1:N,:);
boffset = X(N + 1,:);
q = X(N + 2:2*N + 1,:);
g = X(2 * N + 2,:);
nsv = size(find(lambda ~= 0), 1);

yt1 = yTrain;


for i = 1:N
    sumj = boffset;
    for j = 1:nrows
        sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTrain(i, :), beta);
    end
    yt1(i) = sumj;
end

ntest = size(xTest, 1);
yPred = zeros(ntest, 1);
for i = 1:ntest
    sumj = boffset;
    for j = 1:nrows
        sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTest(i, :), beta);
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
