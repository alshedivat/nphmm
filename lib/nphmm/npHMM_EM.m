function [hmm, ell] = npHMM_EM(X, m, obsBoundary, params)
% npHMM_EM  Fits a nonparametric HMM using an EM-like iterative procedure.
%
% INPUTS:
%   hmm - an NPHMM object (should be already initialized).
%   X - the data matrix where every row is a sequence of observations.

  % Set the default params
  if ~exist('params','var'); params = []; end;
  if ~isfield(params,'verbose'); params.verbose = 1; end;
  if ~isfield(params,'maxIterEM'); params.maxIterEM = 10; end;
  if ~isfield(params,'tol'); params.tol = 1e-2; end;
  if ~isfield(params,'adj_Pi'); params.adj_Pi = 1; end;
  if ~isfield(params,'adj_T'); params.adj_T = 1; end;
  if ~isfield(params,'adj_O'); params.adj_O = 1; end;
  if ~isfield(params, 'sampleWeights')
    params.sampleWeights = true;
  end
  if ~isfield(params, 'doBoundaryCorrection')
    params.doBoundaryCorrection = false;
  end;
  if ~isfield(params,'opt')
    params.opt = [];
    params.opt.use_cheb = false;
  end

  smoothness = 'gauss';
  [numSamples, seqLength] = size(X);

  previous_loglik = -inf;
  loglik = 0;
  converged = 0;
  ell = [];

  % Initialize HMM
  hmm = npHMM(m, obsBoundary, [], [], [], params.opt);
  hmm.randInit();

  % Initialize the KDEs
  if params.verbose
    fprintf('Initialization ... ');
  end
  if params.adj_O
    weights = normalize(repmat(hmm.Pi.prob, 1, numSamples * seqLength), 2);
    [BWs, kdePdfs] = estKdePdfs(X, [], weights, smoothness, params);
    hmm.setO(kdePdfs);
  end

  if params.verbose
    fprintf('Done.\n');
    fprintf('Running EM:\n');
  end

  for iter = 1:params.maxIterEM
    % E-step
    [loglik, exp_trans, exp_visits1, weights] = comp_ess(hmm, X);

    % M-step
    if params.adj_Pi
      hmm.setPi(normalize(exp_visits1));
    end
    if params.adj_T
      hmm.setT(make_stochastic(exp_trans)');
    end
    if params.adj_O
      % BWs = [];
      [BWs, kdePdfs] = estKdePdfs(X, BWs, weights, smoothness, params);
      hmm.setO(kdePdfs);
    end

    if params.verbose
      fprintf('\titer: %d - loglik: %.4f\n', iter, loglik);
    end

    if abs(loglik - previous_loglik) < params.tol
      break;
    end

    previous_loglik = loglik;
    ell = [ell loglik];
  end

  % Re-fit KDE bandwidths
  [BWs, kdePdfs] = estKdePdfs(X, [], weights, smoothness, params);
  hmm.setO(kdePdfs);

  if params.verbose
    if iter < params.maxIterEM
      fprintf('Converged in %d EM iterations.\n', iter);
    else
      fprintf('Did not converge in %d EM iterations.\n', iter);
    end
  end
end


function [loglik, exp_num_trans, exp_num_visits1, weights] = comp_ess(hmm, X)
% COMP_ESS  Computes the expected sufficient statistics for the HMM.
%
% Outputs:
% exp_num_trans(i,j)   = sum_l sum_{t=2}^T Pr(Q(t-1) = i, Q(t) = j | Obs(l))
% exp_num_visits1(i)   = sum_l Pr(Q(1)=i | Obs(l))

  numSamples = length(X);
  exp_num_trans = zeros(hmm.m, hmm.m);
  exp_num_visits1 = zeros(hmm.m, 1);
  post_state_probs = [];
  loglik = 0;

  for ex = 1:numSamples
    % Do forward-backward pass
    [logA, logB, logG, logX, current_loglik] = hmm.fwdback(X(ex, :));
    loglik = loglik + current_loglik;

    % Compute expected sufficient statistics
    exp_num_trans = exp_num_trans + exp(logX);
    exp_num_visits1 = exp_num_visits1 + exp(logG(:, 1));
    post_state_probs = [post_state_probs exp(logG)];
  end

  % Normalize to get sample weights
  weights = normalize(post_state_probs, 2);

end


function [BWs, KDEs] = estKdePdfs(X, initBWs, weights, smoothness, params)
% ESTKDEPDFS  Estimate PDFs as with KDE on weighted samples.

  Y = reshape(X, [], 1);
  numStates = size(weights, 1);
  KDEs = cell(numStates, 1);

  if isempty(initBWs)
    BWs = cell(numStates, 1);
  else
    BWs = initBWs;
  end

  for s = 1:numStates
    params.sampleWeights = weights(s, :)';
    if ~isempty(initBWs)
      KDEs{s} = kdeGivenBW(Y, BWs{s}, smoothness, params);
    else
      [BWs{s}, KDEs{s}] = kdePickBW(Y, smoothness, params, []);
    end
  end
end
