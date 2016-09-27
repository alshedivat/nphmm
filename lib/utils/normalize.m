function [M, z] = normalize(A, dim)
% NORMALISE Make the entries of a (multidimensional) array sum to 1.
% [M, z] = normalize(A)
% where z is the normalizing constant.
%
% [M, z] = normalize(A, dim)
% If dim is specified, we normalize the specified dimension only,
% otherwise we normalize the whole array.

  if nargin < 2
    z = sum(A(:));
  else
    z = sum(A, dim);
  end
  s = z + (z == 0);
  M = bsxfun(@rdivide, A, s);

end
