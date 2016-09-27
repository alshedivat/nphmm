function [M, z] = lognormalize(A, dim)
% LOGNORMALISE Make the entries of a (multidimensional) array sum to 0.
% [M, z] = lognormalize(A)
% where z is the lognormalizing constant.
%
% [M, z] = lognormalize(A, dim)
% If dim is specified, we lognormalize the specified dimension only,
% otherwise we lognormalize the whole array.

  if nargin < 2
    z = logsumexp(A(:));
  else
    z = logsumexp(A, dim);
  end
  M = bsxfun(@minus, A, z);

end
