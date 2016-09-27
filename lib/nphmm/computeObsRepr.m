function [obsHMM, b1, binf, Bx, U, Bxc] = ...
  computeObsRepr(P1, P21, P321, m, obsBoundary, options)
  % Compute the observable representation for the nonparametric HMM
  % given the probabilities P1, P12 and P123.
  % Inputs:
  %   P1: A chebfun handle for the marginals.
  %   P21: A chebfun2 handle for the pairs.
  %   P321: A chebfun3 handle for the triples.
  %   m: The number of hidden states to use.
  %   obsBoundary: The 1D domain for the observations, [minVal, maxVal].
  %   options: A structure with options.
  % Outputs
  %   See the paper.

  if ~exist('options', 'var'); options = []; end;
  if ~isfield(options, 'verbose'); options.verbose = 1; end;
  if ~isfield(options, 'tol'); options.tol = 1e-4; end;

  % Make sure that P1, P21 and P321 are chebfun objects
  assert(isa(P1, 'chebfun'));
  assert(isa(P21, 'chebfun2'));
  assert(isa(P321, 'chebfun3'));

  % Compute the SVD
  [U, S, V] = svd(P21);
  Svals = diag(S);
  rankP21 = length(Svals);

  % Estimate the spectral gap if the true m is given
  spectralGap = 0.0;
  if ~isempty(m) && m < length(Svals)
    spectralGap = Svals(m) / Svals(m + 1);
  end

  % Determine the number of hidden states
  estM = sum(Svals / Svals(1) > options.tol);
  if isempty(m); m = estM; end;
  m = min(m, rankP21);

  if options.verbose
    printM = min(2 * m, rankP21);
    dispSingVals = round(Svals(1:printM) * 1e4) / 1e4;
    fprintf('Rank(P21) = %d, m = %d, SingVals: %s\n', rankP21, m, ...
            mat2str(dispSingVals));
  end

  % Truncate the SVD
  U = U(:, 1:m);
  S = diag(Svals(1:m));
  V = V(:, 1:m);

  % Re-normalize the pdf
  P21 = U * S * V';
  P21 = P21 ./ sum2(P21);

  % Note: chebfun2 internally uses rows as functions of the second argument
  %       columns as functions of the first argument. This is a bit unintuitive
  %       from the standard linear algebra perspective. Anyway, to get U as
  %       the _left_ singular Q-matrix, we need to transpose P21.
  [U, S, V] = svd(P21');
  Svals = diag(S);
  U = U(:, 1:m);
  if options.verbose
    dispSingVals = round(Svals * 1e4) / 1e4;
    fprintf('P21 (m=%d): SingVals: %s.\n', m, mat2str(dispSingVals));
  end

  % Compute b1
  b1 = U' * P1;

  % Compute binf
  binf = pinv(P21 * U) * P1;

  % Finally do Bx
  [Bx, Bxc] = computeBxCF(U, P21, P321);

  % Return an npObsHMM object
  obsHMM = npObsHMM(obsBoundary, b1, binf, Bx, U, P1, P21, P321, ...
                    estM, spectralGap);

end


function [Bx, Bxc] = computeBxCF(U, P21, P321)
  m = size(U, 2);

  % Compute L = U^T P321
  L = cell(m, 1);
  for j = 1:m
    L{j} = P321 * U(:, j); % -> F21 (i.e., chebfun2 with 2,1 arguments)
  end

  % Compute R = pinv(U^T P21)
  R = pinv(P21 * U)';

  % Compute Bx = L * R
  % Notes: 1) Cell array is a workaround for the dimensionality limitations
  %           imposed by Chebfun. We can't have a chebfun object with ndim > 2.
  %        2) Note that we need to transpose L to compute L * R, since chebfun2
  %           contracts along the first dimension on multiplication.
  Bxc = cell(m, 1);
  for j = 1:m
    Bxc{j} = L{j}' * R;
  end
  Bxc = cat(2, Bxc{:});

  Bx = @(x) reshape(Bxc(x), m, m)';
end
