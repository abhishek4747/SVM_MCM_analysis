function [mcm_plotted, accuracy, m_apos] = mcmtoy(label_matrix, instance_matrix, options, contour_level, kernel, beta, C)

%%% mcmtoy(label_matrix, instance_matrix, options, contour_level,beta,C)
% label_matrix: N by 1, has to be two-class
% instance_matrix: N by 2
% options: default '',
%
% contour_level: default [0 0], 
%                change to [-1 0 1] for showing the +/- 1 margin.
%
% svmtoy shows the two-class classification boundary of the 2-D data
% based on libsvm-mat-2.8

%%%%%%%%%%%
% Example:                                              beta,C                        
% mcmtoy(label_matrix,instance_matrix,'',[0 0],'RBF',   0.5 ,10);
% mcmtoy(label_matrix,instance_matrix,'',[0 0],'Linear',0.5 ,10);




if nargin <= 1
  instance_matrix = [];
elseif nargin == 2    
  options = ''
end

if nargin <= 3
  contour_level = [0 0];
end

N = size(label_matrix, 1);
if N <= 0
  fprintf(2, 'number of data should be positive\n');
  return;
end

if size(label_matrix, 2) ~= 1
  fprintf(2, 'the label matrix should have only one column\n');
  return;
end

if size(instance_matrix, 1) ~= N
  fprintf(2, ['the label and instance matrices should have the same ' ...
              'number of rows\n']);
  return;
end

if size(instance_matrix, 2) ~= 2
  fprintf(2, 'svmtoy only works for 2-D data\n');
  return;
end



minX = min(instance_matrix(:, 1));
maxX = max(instance_matrix(:, 1));
minY = min(instance_matrix(:, 2));
maxY = max(instance_matrix(:, 2));

gridX = (maxX - minX) ./ 100;
gridY = (maxY - minY) ./ 100;

minX = minX - 10 * gridX;
maxX = maxX + 10 * gridX;
minY = minY - 10 * gridY;
maxY = maxY + 10 * gridY;

[bigX, bigY] = meshgrid(minX:gridX:maxX, minY:gridY:maxY);


ntest=size(bigX, 1) * size(bigX, 2);
instance_test = [reshape(bigX, ntest, 1), reshape(bigY, ntest, 1)];
label_test = zeros(size(instance_test, 1), 1);

[Z, exitflag] = MCM_Kernel(instance_matrix, label_matrix, instance_test, label_test, kernel, beta, C);


[Z_train, eflag] = MCM_Kernel(instance_matrix, label_matrix, instance_matrix, label_matrix, kernel, beta, C);

neg_train = find(Z_train<0);
Z_train = ones(size(Z_train));
Z_train(neg_train) = -1;
accuracy  = int32(double(length(find(Z_train==label_matrix))/length(label_matrix)*100));




bigZ = reshape(Z, size(bigX, 1), size(bigX, 2));

tf1 = isequal(bigZ, -1*ones(size(bigX, 1), size(bigX, 2)));
tf2 = isequal(bigZ, ones(size(bigX, 1), size(bigX, 2)));


%clf;
hold on;
mcm_plotted = 0 ;
m_apos = 0;
if((exitflag==1) && not(tf1) && not(tf2))
  

  ispos = (label_matrix == label_matrix(1));
  pos = find(ispos);
  neg = find(~ispos);

  %plot(instance_matrix(pos, 1), instance_matrix(pos, 2), 'o');
  %plot(instance_matrix(neg, 1), instance_matrix(neg, 2), 'x');

  %save('Z.mat', 'bigZ');



  contour(bigX, bigY, bigZ, contour_level, 'g');

  %title(options);
  mcm_plotted = 1 ;
else
  if(tf2)
    m_apos=1;
  end
  
end
