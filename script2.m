clear;
addpath('D:\SEMESTERS\Sem7\neural\project\libsvm-3.18\libsvm-3.18\windows');

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);


n_points = 20;
range = 10;
X = range*rand(n_points, 2);
n_plots = 100;


psfile=[strtok('test','.') '.ps'];
pdffile=[strtok('test','.') '.pdf'];


C_mcm = 1;
C_svm = 1;

for i=1:n_plots

	fprintf('\n--------------------------------------- %d -------------------------------------------\n', i);

	ff = figure;
	set(gcf,'Visible','off') 
	random = rand(n_points, 1);
	pos = find(random>=0.5);
	Y = -1*ones(n_points, 1);
	Y(pos) = 1;
	s = ['-t 2 -g 0.5 -c ' int2str(C_svm)];
	[svm_plotted, s_accuracy, s_apos] = svmtoy(Y, X, s);
	fprintf('SVM done\n\n');
	hold on;
	[mcm_plotted, m_accuracy, m_apos] = mcmtoy(Y, X, '', [0 0], 'RBF', 0.5, C_mcm);
	fprintf('MCM done\n\n');
	svm_s = strcat('SVM C=', int2str(C_svm),' accuracy=',int2str(s_accuracy));
	mcm_s = strcat('MCM C=', int2str(C_mcm),' accuracy=', int2str(m_accuracy));
	fprintf('mcm_plotted = %d, svm_plotted = %d \n', mcm_plotted, svm_plotted );


	if(mcm_plotted==1 && svm_plotted==1)
		legend('Positive', 'Negative', svm_s, mcm_s);
	elseif(mcm_plotted==1)
		legend('Positive', 'Negative', mcm_s);
		if(s_apos==1)
			svm_s = strcat(svm_s,32,'All Positive');
		else
			svm_s = strcat(svm_s,32,'All Negative');
		end

	elseif (svm_plotted==1)
		legend('Positive', 'Negative', svm_s);

		if(m_apos==1)
			mcm_s = strcat(mcm_s,32,'All Positive');
		else
			mcm_s = strcat(mcm_s,32,'All Negative');
		end


	else
		legend('Positive', 'Negative');
		if(s_apos==1)
			svm_s = strcat(svm_s,32,'All Positive');
		else
			svm_s = strcat(svm_s,32,'All Negative');
		end

		if(m_apos==1)
			mcm_s = strcat(mcm_s,32,'All Positive');
		else
			mcm_s = strcat(mcm_s,32,'All Negative');
		end
	end

	title(strcat(svm_s, 10, mcm_s));
	if(i==1)
		print(['-f' num2str(ff)],psfile,'-dpsc2')
	else
		print(['-f' num2str(ff)],psfile,'-dpsc2','-append')	
	end
		
end


ps2pdf('psfile', psfile, 'pdffile', pdffile);
delete(psfile);








