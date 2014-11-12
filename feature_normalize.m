function X_norm = feature_normalize(X)

X_norm = X;

mu = mean(X);
sigma = std(X);

for i=1:size(X, 2)
	X_norm(:, i) = X_norm(:, i) - mu(i);
	X_norm(:, i) = X_norm(:, i) / sigma(i);
end;
