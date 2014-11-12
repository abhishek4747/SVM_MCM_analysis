function [svm_plotted, accuracy, s_apos] = svmtoy(label_matrix, instance_matrix, options, contour_level)


%%% svmtoy(label_matrix, instance_matrix, options, contour_level)
% label_matrix: N by 1, has to be two-class
% instance_matrix: N by 2
% options: default '',
%
% contour_level: default [0 0], 
%                change to [-1 0 1] for showing the +/- 1 margin.
%
% svmtoy shows the two-class classification boundary of the 2-D data
% based on libsvm-mat-2.8

%%%%%%%%%%
% Run with options as '-t 0 -c 10' for linear kernel with C = 10, 
% with options as '-t 2 -g 100 -c 10' for rbf with gamma = 100, C = 10

% Example: svmtoy(label_matrix, instance_matrix, '-t 0 -c 1');
% will run a linear kernel with C = 1


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

mdl = svmtrain(label_matrix, instance_matrix, options); %%%%

nclass = mdl.nr_class;
svmtype = mdl.Parameters(1);



if nclass ~= 2 || svmtype >= 2
  fprintf(2, ['cannot plot the decision boundary for these ' ...
              'SVM problems\n']);
  return
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

%mdl.Parameters(1) = 3; % the trick to get the decision values
ntest=size(bigX, 1) * size(bigX, 2);
instance_test=[reshape(bigX, ntest, 1), reshape(bigY, ntest, 1)];
label_test = zeros(size(instance_test, 1), 1);

[Z] = svmpredict(label_test, instance_test, mdl);

[Z_train] = svmpredict(label_matrix, instance_matrix, mdl);

neg_train = find(Z_train<0);
Z_train = ones(size(Z_train));
Z_train(neg_train) = -1;
accuracy  = int32(double(length(find(Z_train==label_matrix))/length(label_matrix)*100));

bigZ = reshape(Z, size(bigX, 1), size(bigX, 2));


tf1 = bigZ > 0;
tf1 = sum(sum(tf1));
if(tf1==length(Z))
  tf1=1;
else
  tf1=0;
end
tf2 = bigZ < 0;
tf2 = sum(sum(tf2));
if(tf2==length(Z))
  tf2=1;
else
  tf2=0;
end









clf;
hold on;

svm_plotted=0;
s_apos=0;
fprintf('All negative : %d, All +ve : %d\n', tf1, tf2);

%disp(bigZ);


ispos = (label_matrix == label_matrix(1));
pos = find(ispos);
neg = find(~ispos);

plot(instance_matrix(pos, 1), instance_matrix(pos, 2), '+');
plot(instance_matrix(neg, 1), instance_matrix(neg, 2), ' o ');


if(not(tf1) && not(tf2))
  contour(bigX, bigY, bigZ, contour_level, 'r');

  svm_plotted=1;


else
  if(tf1)
    s_apos=1;
  end
end
%title(options);
