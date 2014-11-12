clear;

figure;
label_matrix = load('data1_y.dat');
instance_matrix = load('data1_x.dat');
instance_matrix = feature_normalize(instance_matrix);
mcmtoy(label_matrix,instance_matrix,'',[0 0],'RBF',   5 ,10);
%pause();
%sprintf('Press Enter to continue 1')

%mcmtoy(label_matrix,instance_matrix,'',[0 0],'Linear',   0.5 ,10);

%%pause();
%%sprintf('Press Enter to continue;')
figure(2);

label_matrix = load('data2_y.dat');
instance_matrix = load('data2_x.dat');
instance_matrix = feature_normalize(instance_matrix);
mcmtoy(label_matrix,instance_matrix,'',[0 0],'RBF',  0.5, 10);

%pause();
%sprintf('Press Enter to continue 2')

%mcmtoy(label_matrix,instance_matrix,'',[0 0],'Linear',   10, 10);

%%pause();
%%sprintf('Press Enter to continue;')

figure(3);
load('data3.mat');

pos = find(y == 0); 
y(pos) = -1;


label_matrix = y;
instance_matrix = X;

instance_matrix = feature_normalize(instance_matrix);



mcmtoy(label_matrix,instance_matrix,'',[0 0],'RBF',  0.5, 10);

%pause();
%sprintf('Press Enter to continue 3')

%mcmtoy(label_matrix,instance_matrix,'',[0 0],'Linear',   10, 10);

%%pause();
%%sprintf('Press Enter to continue;')

figure(4);
load('data4.mat');

pos = find(y == 0); 
y(pos) = -1;


label_matrix = y;
instance_matrix = X;

instance_matrix = feature_normalize(instance_matrix);


%sprintf('This takes a lot of time')
mcmtoy(label_matrix,instance_matrix,'',[0 0],'RBF',  0.1, 1);
%pause();
%sprintf('Press Enter to continue 4')



figure(5);
load('data5.mat');

pos = find(y == 0); 
y(pos) = -1;


label_matrix = y;
instance_matrix = X;

instance_matrix = feature_normalize(instance_matrix);

mcmtoy(label_matrix,instance_matrix,'',[0 0],'RBF',  0.1, 1);








