clear;
addpath('D:\SEMESTERS\Sem7\neural\project\libsvm-3.18\libsvm-3.18\windows');
%addpath('C:\Users\Abhishek\Dropbox\SEM VII\EEL781 Neural Networks\Project\libsvm-3.18\matlab');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%File reading for training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
try
    load('opti_gamma_C.mat');
    fprintf(strcat('Variables loaded.',32,num2str(0),32,'something something.\n'));
catch VOCAB
	file_train = 'train/monks-1.train';
	[train_X, train_y, train_m] = file_read(file_train);

	file_test = 'test/monks-1.test';
	[test_X, test_y, test_m] = file_read(file_test);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% gamma_arr = [0.001; 0.002; 0.003; 0.01; 0.02; 0.03; 0.1; 0.2; 0.3; 1; 2; 3; 10; 20; 30; 100; 200; 300];
	% C_arr = [0.001; 0.002; 0.003; 0.01; 0.02; 0.03; 0.1; 0.2; 0.3; 1; 2; 3; 10; 20; 30; 100; 200; 300];
	% accuracy_arr = zeros(length(gamma_arr), length(C_arr));

	gamma_arr = [0.001:0.001:0.01 0.01:0.01:0.1 0.1:0.1:1 1:1:10 10:10:100 100:100:1000];
	C_arr = [0.001:0.001:0.01 0.01:0.01:0.1 0.1:0.1:1 1:1:10 10:10:100 100:100:1000];
	accuracy_arr = zeros(length(gamma_arr), length(C_arr));

	for i =1:length(gamma_arr)
		for j=1:length(C_arr)

			gamma_svm = gamma_arr(i);
			C_svm = C_arr(j);
			options_svm = ['-t 2 -g ', num2str(gamma_svm),' -c ', num2str(C_svm)]


			model = svmtrain(train_y, train_X, options_svm);
			%[predicted_label, accuracy, sink] = svmpredict(test_y, test_X, model);

			%accuracy_arr(i, j) = accuracy(1);

			model.Parameters(1) = 3;
			[predicted_label] = svmpredict(test_y, test_X, model);
			neg_test = find(predicted_label<0);
			predicted_label = ones(size(predicted_label));
			predicted_label(neg_test) = -1;
			accuracy  = (double(length(find(predicted_label==test_y))/length(test_y)*100));
			accuracy_arr(i, j) = accuracy;

		end
	end


	index = find(accuracy_arr==max(max(accuracy_arr)));
	index = index(1);
	opti_C_svm = C_arr( ceil(index/length(gamma_arr)))
	opti_gamma_svm = gamma_arr( mod(index, length(gamma_arr)))

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MCM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



	gamma_arr = [0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100; 300];
	C_arr = [0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100; 300];
	accuracy_arr = zeros(length(gamma_arr), length(C_arr));


	for i =1:length(gamma_arr)
		for j=1:length(C_arr)

			gamma_mcm = gamma_arr(i);
			C_mcm = C_arr(j);

			options_mcm = ['-t 2 -g ', num2str(gamma_mcm),' -c ', num2str(C_mcm)]

			[predicted_label, exitflag] = MCM_Kernel(train_X, train_y, test_X, test_y, 'RBF', gamma_mcm, C_mcm);

			
			accuracy  = (double(length(find(predicted_label==test_y))/length(test_y)*100));
			accuracy_arr(i, j) = accuracy;

		end
	end


	index = find(accuracy_arr==max(max(accuracy_arr)));
	index = index(1);
	opti_C_mcm = C_arr(ceil(index/length(gamma_arr)))
	opti_gamma_mcm = gamma_arr( mod(index, length(gamma_arr)))


	save('opti_gamma_C.mat', 'opti_C_svm', 'opti_gamma_svm', 'opti_C_mcm', 'opti_gamma_mcm', 'test_X', 'test_y', 'train_X', 'train_y');
end





princ_mat = princomp(train_X);
princ_x = princ_mat(:,1);
princ_y = princ_mat(:,2);


train_X_2d = [train_X*princ_x train_X*princ_y];

figure(1);
ispos = (train_y == 1);
pos = find(ispos);
neg = find(~ispos);
hold on;
plot(train_X_2d(pos, 1), train_X_2d(pos, 2), '+');
plot(train_X_2d(neg, 1), train_X_2d(neg, 2), ' o ');

figure(2);
hold on;
plot(train_X_2d(pos, 1), train_X_2d(pos, 2), '+');
plot(train_X_2d(neg, 1), train_X_2d(neg, 2), ' o ');



% density = 1000;

% minX = min(train_X_2d(:, 1));
% maxX = max(train_X_2d(:, 1));
% minY = min(train_X_2d(:, 2));
% maxY = max(train_X_2d(:, 2));

% gridX = (maxX - minX) ./ density;
% gridY = (maxY - minY) ./ density;

% minX = minX - 10 * gridX;
% maxX = maxX + 10 * gridX;
% minY = minY - 10 * gridY;
% maxY = maxY + 10 * gridY;

% [bigX, bigY] = meshgrid(minX:gridX:maxX, minY:gridY:maxY);

% options_svm = ['-t 2 -g ', num2str( opti_gamma_svm),' -c ', num2str(opti_C_svm)]
% model = svmtrain(train_y, train_X, options_svm);
% model.Parameters(1) = 3;

% label_test = ones(size(bigX,1)*size(bigX,2),1);

% instance_test = ones(size(bigX,1)*size(bigX,2),6);
% i=0;
% for ii=1:length(bigX(1,:))
% 	for	jj=1:length(bigY(:,1))
% 		i = i + 1;
% 		instance_test(i,:) = bigX(1,ii)*princ_x + bigY(jj,1)*princ_y;
% 	end
% end
% pause

% [Z] = svmpredict(label_test, instance_test, model);
% % Z(find(Z>=0))=1;
% % Z(find(Z<0))=-1;
% bigZ = reshape(Z, size(bigX, 1), size(bigX, 2));

% figure(1);
% hold on;
% contour(bigX, bigY, bigZ, [-1 0 1], 'r');




density = 0.5;
aa1 = 1:density:3;
aa2 = 1:density:3;
aa3 = 1:density:2;
aa4 = 1:density:3;
aa5 = 1:density:4;
aa6 = 1:density:2;


test_X = zeros(length(aa1)*length(aa2)*length(aa3)*length(aa4)*length(aa5)*length(aa6), 6);
test_y = ones(size(test_X, 1), 1);
i=0;
for a1=aa1
	for a2=aa2
		for a3=aa3
			for a4=aa4
				for a5=aa5
					for a6=aa6
						i=i+1;
						test_X(i, :) = [a1 a2 a3 a4 a5 a6];
					end
				end
			end	
		end
	end	
end
clear aa1 aa2 aa3 aa4 aa5 aa6;

fprintf('loops done: i:%d',i);


options_svm = ['-t 2 -g ', num2str( opti_gamma_svm),' -c ', num2str(opti_C_svm)]
model = svmtrain(train_y, train_X, options_svm);
model.Parameters(1) = 3;
[predicted_label_svm] = svmpredict(test_y, test_X, model);
neg_test = find(predicted_label_svm<0);
predicted_label_svm = ones(size(predicted_label_svm));
predicted_label_svm(neg_test) = -1;	



[predicted_label_mcm, exitflag] = MCM_Kernel(train_X, train_y, test_X, test_y, 'RBF', opti_gamma_mcm, opti_C_mcm);






test_X_2d_x = test_X*princ_x;
test_X_2d_y = test_X*princ_y;

% contx = sort(test_X_2d_x);
% contx_mat = contx*ones(1,length(contx));

% conty = sort(test_X_2d_y);
% conty_mat = conty*ones(1,length(conty));
% conty_mat = conty';

% contz = zeros(size(contx_mat,1),size(contx_mat,2));
% for ii=1:length(predicted_label_svm)
% 	xcord = find(contx==test_X_2d_x(ii));
% 	ycord = find(conty==test_X_2d_y(ii));
% 	contz(xcord,ycord) = predicted_label_svm(ii);
% end


ispos = (predicted_label_svm == 1);
pos = find(ispos);
neg = find(~ispos);
figure(1)
hold on;
plot(test_X_2d_x(pos), test_X_2d_y(pos), 'ko', 'MarkerSize', 2);
plot(test_X_2d_x(neg), test_X_2d_y(neg), 'ro', 'MarkerSize',  2);

ispos = (predicted_label_mcm == 1);
pos = find(ispos);
neg = find(~ispos);
figure(2);
hold on;
plot(test_X_2d_x(pos), test_X_2d_y(pos), 'ko', 'MarkerSize', 2);
plot(test_X_2d_x(neg), test_X_2d_y(neg), 'ro', 'MarkerSize',  2);


% contour(contx_mat,conty_mat,contz,[-1 0 1]);





