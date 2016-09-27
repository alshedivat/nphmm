function vals = kde2cheb(X, Y, kdeFuncH)
% A function so that the KDE estimate amenable to chebfun2.
  Xall = X(:);
  Yall = Y(:);
  vals = kdeFuncH([Xall, Yall]);
  vals = reshape(vals, size(X));
end

