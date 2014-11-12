clear;
% clc;
addpath('D:\SEMESTERS\Sem7\neural\project\libsvm-3.18\libsvm-3.18\windows');
%addpath('C:\Users\Abhishek\Dropbox\SEM VII\EEL781 Neural Networks\Project\libsvm-3.18\matlab');

try

	load('training.mat');
	fprintf('training.mat loaded\n');

catch LOAD

	z = sqrt(2/3)-sqrt(3/8);
	v1 = [-0.5 -1/sqrt(12) -z];
	v2 = [0.5 -1/sqrt(12) -z];
	v3 = [0 (1/sqrt(3)) -z];
	v4 = [0 0 sqrt(3.0/8)];

	tetra = [v1; v2; v3; v4];
	% v = [v1; v2; v3; v4; v1]

	% plot(v(:,1), v(:,2));
	% axis equal


	inner_a = 2;
	outer_a = 5;

	outer_tetra = tetra*outer_a;
	inner_tetra = tetra*inner_a;

	density = 0.5;


	x = min(outer_tetra(:,1))-1:density:max(outer_tetra(:,1))+1;
	y = min(outer_tetra(:,2))-1:density:max(outer_tetra(:,2))+1;
	z = min(outer_tetra(:,3))-1:density:max(outer_tetra(:,3))+1;


	train_X = zeros(length(x)*length(y)*length(z), 3);
	train_y = zeros(length(x)*length(y)*length(z), 1);
	train_m = length(x)*length(y)*length(z);

	fprintf('Making training data\n');
	ii=0;
	for i=x
		for j=y
			for k=z
				ii=ii+1;
				point = [i j k];
				train_X(ii,:) = point;
				is_in_inner = is_in(inner_tetra, point);
				is_in_outer = is_in(outer_tetra, point);

				if(is_in_outer && ~is_in_inner)
					train_y(ii)=-1;
				else	
					train_y(ii)=1;
				end
				
			end
		end
	end

	fprintf('Done\n');




	save('training.mat', 'train_X', 'train_y', 'train_m');
	fprintf('training.mat saved\n');




end


% neg = find(train_y==-1);
% neg = train_X(neg,:);
% scatter3(neg(:,1), neg(:,2), neg(:,3), 'r');




try
    load('opti_gamma_C.mat');
    fprintf(strcat('Variables loaded.',32,num2str(0),32,'something something.\n'));
catch VOCAB
	
	

	test_X = train_X;
	test_y = train_y;
	test_m = train_m;
	[test_X, test_y, test_m] = [train_X, train_y, train_m];

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

