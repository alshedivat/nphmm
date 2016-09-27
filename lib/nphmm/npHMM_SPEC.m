function estHMM = npHMM_SPEC(X, m, obsBoundary, options)
  % Estimates a nonparametric HMM using the data.
  %
  % TODO:
  % 1. If m not given, cross validate for m
  % 2. Rehash code in the case X has several longer sequences.

  if ~exist('options', 'var'); options = []; end;
  if ~isfield(options, 'verbose'); options.verbose = 0; end;
  if ~isfield(options, 'estimate_P1'); options.estimate_P1 = 1; end;
  if ~isfield(options, 'estimate_P21'); options.estimate_P21 = 1; end;

  if isa(X, 'double') & size(X, 2) == 3
    Xsingles = X(:, 1);
    Xpairs = X(:, [2 1]);
    Xtriples = X(:, [3 2 1]);
  elseif isa(X, 'double')
    % TODO
  elseif isa(X, 'cell')
    % TODO
  end

  % construct density estimates.
  if options.verbose
    fprintf('Estimating densities ... ');
    tic
  end

  kdeSmoothness = 'gauss';
  kdeParams.doBoundaryCorrection = false;
  [~, P321hat] = kdePickBW(Xtriples, kdeSmoothness, kdeParams);

  % Construct chebfun objects for P321
  eP321c = chebfun3(@(x, y, z) kde3cheb(x, y, z, P321hat), ...
                    repmat(obsBoundary, 1, 3), ...
                    'tech', 'minSamples', 10, 'fiberDim', 2);
  eP321c = eP321c / sum3(eP321c);

  % Either estimate P21 via KDE, or compute it from eP321c
  if options.estimate_P21
    [~, P21hat] = kdePickBW(Xpairs, kdeSmoothness, kdeParams);
    eP21c = chebfun2(@(x, y) kde2cheb(x, y, P21hat), ...
                     repmat(obsBoundary, 1, 2));
  else
    eP21c = sum(eP321c, 1);
  end
  eP21c = eP21c / sum2(eP21c);

  % Either estimate P1 via KDE, or compute it from eP21c
  if options.estimate_P1
    [~, P1hat] = kdePickBW(Xsingles, kdeSmoothness, kdeParams);
    eP1c = chebfun(@(x) kde1cheb(x, P1hat), obsBoundary);
  else
    % Note: chebfun2 swaps the axis internally, hence sum along the 2nd axis.
    eP1c = sum(eP21c, 2);
  end
  eP1c = eP1c / sum(eP1c);

  if options.verbose
    toc
  end

  % Construct the observable HMM
  estHMM = computeObsRepr(eP1c, eP21c, eP321c, m, obsBoundary, options);

end
