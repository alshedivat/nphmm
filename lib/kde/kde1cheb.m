function vals = kde1cheb(X, kdeFuncH)
% A function so that the KDE estimate amenable to chebfun2.
  Xall = X(:);
  vals = kdeFuncH(Xall);
  vals = reshape(vals, size(X));
end

