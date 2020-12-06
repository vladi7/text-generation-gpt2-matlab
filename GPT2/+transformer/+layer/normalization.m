function Z = normalization(X, g, b, opt)

if isequal(opt, 'layer')
normalizationDimension = 1;

epsilon = single(1e-5);

U = mean(X, normalizationDimension);
S = mean((X-U).^2, normalizationDimension);
X = (X-U) ./ sqrt(S + epsilon);
Z = g.*X + b;
end
if isequal(opt, 'batch')
normalizationDimension = 1;
epsilon = single(1e-5);

U = mean(X, normalizationDimension);
S = mean((X).^2, normalizationDimension);
X = (X-U) ./ sqrt(S + epsilon);
Z = g.*X + b;
end
if isequal(opt, 'powernorm')
normalizationDimension = 1;

U = mean(X, normalizationDimension);
S = mean((X-U).^2, normalizationDimension);
X = (X-U) ./ sqrt(S);
Z = g.*X + b;
end
end