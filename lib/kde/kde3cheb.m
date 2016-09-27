function vals = kde3cheb(X, Y, Z, kdeFuncH)
% A function  so that the KDE is amenable to chebfun3.
  Xall = X(:);
  Yall = Y(:);
  Zall = Z(:);
  vals = kdeFuncH([Xall Yall Zall]);
  vals = reshape(vals, size(X));
end

