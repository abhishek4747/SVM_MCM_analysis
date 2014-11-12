function [X, y, m] = file_read(file)

fid = fopen(file, 'r');
line = fgets(fid);
m=0;
while ischar(line)
    m = m+1;
    line = fgets(fid);
end
fclose(fid);

inp = zeros(m, 7);

fid = fopen('train/monks-1.train', 'r');
line = fgets(fid);
j=0;
while ischar(line)
    sread = strread(line,'%s');
    j=j+1;
    for i=1:length(sread)-1
    	inp(j, i) = str2num(sread{i});
    end
    line = fgets(fid);
end
fclose(fid);


X = inp(:, 2:end );
y = inp(:, 1);


zero = find(y==0);
y(zero) = -1;