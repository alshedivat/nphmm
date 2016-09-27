function [T, Z] = make_stochastic(T)
% MAKE_STOCHASTIC Ensure the argument is a stochastic matrix, i.e.,
%                    the sum over the last dimension is 1.
%
% [T, Z] = mk_stochastic(T)
%
% If T is a vector, it will sum to 1.
% If T is a matrix, each row will sum to 1.
% If T is a 3D array, then sum_k T(i,j,k) = 1 for all i, j.

% Set zeros to 1 before dividing.
% This is valid since S(j) = 0 iff T(i,j) = 0 for all j.

if (ndims(T) == 2) && (size(T, 1) == 1 || size(T, 2) == 1) % isvector
  [T, Z] = normalize(T);
elseif ndims(T) == 2 % matrix
  [T, Z] = normalize(T, 2);
else % multi-dimensional array
  ns = size(T);
  T = reshape(T, prod(ns(1:end-1)), ns(end));
  [T, Z] = normalize(T, 2);
  T = reshape(T, ns);
  Z = reshape(Z, ns(1:end-1));
end
