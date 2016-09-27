classdef npHMM < handle
% Nonparametric HMM.
% The primary representation is unobservable (Pi, T, O).

  properties
    m;  % number of hidden states

    % Unobservable representation
    Pi; % initial probabilities of the states
    T;  % the stochastic matrix of transitions
    O;  % the nonparametric observation PDFs

    % Observable representation
    P1; P21; P321;
    b1;   % = (U' O) Pi
    binf; % = 1' \ (U' O)
    Bx;   % = (U' O) A_x \ (U' O)
    U;    % the top m singular vectors of P21.

    obsBoundary;
    opt;
  end % properties


  methods


    % Constructor
    function obj = npHMM(m, obsBoundary, Pi, T, O, opt)
      if nargin > 0
        obj.m = m;

        if exist('obsBoundary', 'var'); obj.obsBoundary = obsBoundary;
        else obj.obsBoundary = [-1, 1]; end;

        if exist('Pi', 'var') && ~isempty(Pi); obj.Pi = Pi;
        else obj.Pi = []; end;
        if exist('T', 'var') && ~isempty(T); obj.T = T;
        else obj.T = []; end;
        if exist('O', 'var') && ~isempty(O); obj.O = O;
        else obj.O = {}; end;

        if exist('opt', 'var') && ~isempty(opt); obj.opt = opt;
        else obj.opt = []; end;
        if ~isfield(obj.opt, 'use_cheb'); obj.opt.use_cheb = true; end;
      end
    end


    function setPi(obj, prob)
      obj.Pi.prob = prob;
      obj.Pi.logprob = log(prob);
      obj.Pi.dist = makedist('Multinomial', 'Probabilities', prob);
    end


    function setT(obj, T)
      numStates = size(T, 1);
      obj.T.prob = T;
      obj.T.logprob = log(T);
      obj.T.dist = cell(numStates, 1);
      for i = 1:numStates
        obj.T.dist{i} = makedist('Multinomial', 'Probabilities', T(:, i));
      end
    end


    function setO(obj, kdePdfs)
      obj.O.kdePdfs = kdePdfs;
      numStates = length(kdePdfs);

      if obj.opt.use_cheb
        obj.O.pdfs = cell(numStates, 1);
        for j = 1:numStates
          obj.O.pdfs{j} = chebfun(kdePdfs{j}, obj.obsBoundary);
          obj.O.pdfs{j} = obj.O.pdfs{j} / sum(obj.O.pdfs{j});
        end
      else
        obj.O.pdfs = obj.O.kdePdfs;
      end

      obj.O.logpdfs = cell(numStates, 1);
      for j = 1:numStates
        obj.O.logpdfs{j} = @(x) log(obj.O.pdfs{j}(x));
      end

    end


    function O = getChebO(obj)
      % Return emission PDFs as a cell array of chebfun objects.
      if obj.opt.use_cheb
        O = obj.O.pdfs;
      else
        numStates = length(obj.O.pdfs);
        O = cell(numStates, 1);
        for j = 1:numStates
          O{j} = chebfun(obj.O.pdfs{j}, obj.obsBoundary);
          O{j} = O{j} / sum(O{j});
        end
      end
    end


    function randInit(obj)
      obj.genInitialProbs();
      obj.genTransitionProbs();
      obj.genObservationPdfs();
    end


    function genInitialProbs(obj)
      numStates = obj.m;
      prob = rand(numStates, 1);
      prob = bsxfun(@rdivide, prob, sum(prob));
      prob = 0.2 * (ones(obj.m, 1) / obj.m) + 0.8 * prob;

      obj.setPi(prob);
    end


    function genTransitionProbs(obj)
      numStates = obj.m;
      prob = rand(numStates);
      prob = bsxfun(@rdivide, prob, sum(prob));

      % Create a banded matrix. Makes sure that T is decently conditioned and
      % also biases the transitions to give something interesting.
      I = eye(numStates);
      Tb = I + circshift(I, 1) + circshift(I, -1);
      Tb(1, end) = 0; Tb(end, 1) = 0;
      Tb = bsxfun(@rdivide, Tb, sum(Tb, 1));
      T = 0.5 * Tb + 0.5 * prob;

      obj.setT(T);
    end


    function genObservationPdfs(obj)
      numStates = obj.m;
      % [samplers, kdePdfs] = genNPpdfs(numStates, obj.obsBoundary);
      [samplers, kdePdfs] = genNPpdfs(numStates, obj.obsBoundary);
      obj.O.samplers = samplers;

      obj.setO(kdePdfs);
    end


    function [X, H] = sample(obj, numSequences, seqLength)
      X = zeros(numSequences, seqLength);
      H = zeros(numSequences, seqLength);
      for i = 1:numSequences
        % Sample an initial state and initial observation
        H(i, 1) = random(obj.Pi.dist, 1);
        X(i, 1) = obj.O.samplers{H(i, 1)}(1);
        for t = 2:seqLength
          % Make a transition
          H(i, t) = random(obj.T.dist{H(i, t - 1)}, 1);
          % Sample an observation
          X(i, t) = obj.O.samplers{H(i, t)}(1);
        end
      end
    end


    function [P1, P21, P321] = constructJointProbs(obj, verbose)
      % Constructs a P1, P21, and P321 as chebfun, chebfun2, and chebfun3
      % objects, respectively, in an incremental fashion.
      %
      % NOTE: the computations are done NOT in the log space, and this may,
      %       potentially, cause underflows. However, since we work with
      %       sequences no longer than triplets, this should work fine.
      if ~exist('verbose', 'var') || isempty(verbose)
        verbose = 1;
      end

      O = obj.getChebO();

      if verbose
        fprintf('Constructing Pr(X1) ... ');
        tic
      end

      % Construct Pr(X1, H1) = Pr(X1 | H1) Pr(H1)
      PrX1H1 = cell(obj.m, 1);
      for i = 1:obj.m
        PrX1H1{i} = obj.Pi.prob(i) * O{i};
      end

      % Construct Pr(X1) = sum_{j=1}^m Pr(X1, H1=j)
      P1 = 0;
      for j = 1:obj.m
        P1 = P1 + PrX1H1{j};
      end
      P1 = P1 / sum(P1);

      if verbose
        toc
        fprintf('Constructing Pr(X1, X2) ... ');
        tic
      end

      % Construct P(X1, H2) = sum_{j=1}^m Pr(X1, H1=j) Pr(H2 | H1=j)
      PrX1H2 = cell(obj.m, 1);
      for i = 1:obj.m
        PrX1H2{i} = 0;
        for j = 1:obj.m
          PrX1H2{i} = PrX1H2{i} + PrX1H1{j} * obj.T.prob(i, j);
        end
      end

      % Construct Pr(X1, X2, H2) = Pr(X2 | H2) P(X1, H2),
      PrX1X2H2 = cell(obj.m, 1);
      for i = 1:obj.m
        PrX1X2H2{i} = chebfun2( ...
            @(x, y) reshape(PrX1H2{i}(x(:)) .* O{i}(y(:)), size(x)), ...
            repmat(obj.obsBoundary, 1, 2));
      end

      % Construct Pr(X1, X2) = sum_{H2} Pr(X1, X2, H2)
      P12 = 0;
      for j = 1:obj.m
        P12 = P12 + PrX1X2H2{j};
      end
      P21 = P12' / sum2(P12);

      if verbose
        toc
        fprintf('Constructing Pr(X1, X2, X3) ... ');
        tic
      end

      % Construct P(X1, X2, H3) = sum_{j=1}^m Pr(X1, X2, H2=j) Pr(H3 | H2=j)
      PrX1X2H3 = cell(obj.m, 1);
      for i = 1:obj.m
        PrX1X2H3{i} = 0;
        for j = 1:obj.m
          PrX1X2H3{i} = PrX1X2H3{i} + PrX1X2H2{j} * obj.T.prob(i, j);
        end
      end

      % Construct Pr(X1, X2, X3, H3) = Pr(X3 | H3) Pr(X1, X2, H3)
      PrX1X2X3H3 = cell(obj.m, 1);
      for i = 1:obj.m
        PrX1X2X3H3{i} = chebfun3( ...
          @(x,y,z) reshape(PrX1X2H3{i}(x(:),y(:)) .* O{i}(z(:)), size(x)), ...
          repmat(obj.obsBoundary, 1, 3));
      end

      % Construct Pr(X1, X2, X3) = sum_{H3} Pr(X1, X2, X3, H3)
      P123 = 0;
      for j = 1:obj.m
        P123 = P123 + PrX1X2X3H3{j};
      end
      P321 = permute(P123, [3 2 1]) / sum3(P123);

      if verbose
        toc
      end
    end


    function A = Ax(obj, x)
      % Computes the matrix A(x) = T * diag( O(x) ).
      O = obj.getChebO(); O = cat(2, O{:});
      A = bsxfun(@times, obj.T.prob, O(x));
    end


    function [logA, logS, logX] = forward(obj, logO, scaled, initLogPr)
      % The forward pass of the forward-backward algorithm.

      if ~exist('initLogPr', 'var') || isempty(initLogPr)
        initLogPr = obj.Pi.logprob;
      end

      T = size(logO, 2);
      logA = zeros(obj.m, T);
      logS = zeros(1, T);

      if nargout > 2
        logX = zeros(obj.m, obj.m, T - 1);
      end

      % Compute logA at t = 1
      logA(:, 1) = logO(:, 1) + initLogPr;
      if scaled
        [logA(:, 1), logS(1)] = lognormalize(logA(:, 1));
      end

      for t = 2:T
        % Compute logA
        logA(:, t) = logO(:, t) + ...
            logsumexp(bsxfun(@plus, obj.T.logprob, logA(:, t - 1)'), 2);
        if scaled
          [logA(:, t), logS(t)] = lognormalize(logA(:, t));
        end

        if nargout > 2
          % Compute logX
          logX(:, :, t - 1) = ...
              lognormalize(bsxfun(@plus, logA(:, t - 1), logO(:, t)') + ...
                           obj.T.logprob');
        end
      end

      if nargout > 2
        % Sum probabilities over the sequence
        logX = logsumexp(logX, 3);
      end
    end


    function [logB] = backward(obj, logO, scaled)
      % The backward pass of the forward-backward algorithm.
      T = size(logO, 2);
      logB = zeros(obj.m, T);

      for t = T-1:-1:1
        % Compute logB
        b = logB(:, t + 1) + logO(:, t + 1);
        logB(:, t) = logsumexp(bsxfun(@plus, obj.T.logprob, b), 1)';
        if scaled
          logB(:, t) = lognormalize(logB(:, t));
        end
      end
    end


    function [logO] = compLogO(obj, X)
      % Pre-compute logO(j, t) = log p(X(t) | H_t=j).
      T = size(X, 2);
      logO = zeros(obj.m, T);
      for j = 1:obj.m
        logO(j, :) = obj.O.logpdfs{j}(X')';
      end
    end


    function [logA, logB, logG, logX, loglik] = fwdback(obj, X, params)
      % FWDBACK Compute the posterior probabilities in the HMM using the
      %         forwards backwards algorithm.
      %
      % OUTPUTS:
      % A(i, t) = p(H_t=i | X_{1:t}) (or p(H_t=i, X_{1:t}) if params.scaled=0)
      % B(i, t) = p(X_{t+1:T} | H_t=i) * p(X_{t+1:T} | X_{1:t}) (or
      %           p(X_{t+1:T} | H_t=i) if scaled=0)
      % G(i, t) = p(H_t=i | X_{1:T})
      % X(i, j) = sum_{t=}^{T-1} p(H_{t-1}=i, H_t=j | X_{1:T})
      % loglik  = log p(X_{1:T})

      % Parse params
      if ~exist('params', 'var'); params = []; end;
      if ~isfield(params,'scaled'); params.scaled = 1; end;

      % Do forward and backward passes
      logO = obj.compLogO(X);
      [logA, logS, logX] = obj.forward(logO, params.scaled);
      logB = obj.backward(logO, params.scaled);
      logG = lognormalize(logA + logB, 1);

      if params.scaled
        loglik = sum(logS);
      else
        loglik = logsumexp(logA(:, end));
      end
    end


    function loglik = computeLoglik(obj, X)
      % COMPUTELOGLIK  Compute log likelihood (a.k.a. log joint probability).
      numSequences = size(X, 1);

      loglik = zeros(numSequences, 1);
      for i = 1:numSequences
        logO = obj.compLogO(X(i, :));
        logA = obj.forward(logO, 0);
        loglik(i) = logsumexp(logA(:, end));
      end
    end


    function [logJointPr, finalLogPr] = computeLogJointProb(obj, X, initLogPr)
      % COMPUTELOGPROB  Compute log joint probability for each sequence.
      % logJointPr(i) = log P(X(i, 1:T))
      % finalLogPr(i, k) = log P(H_T=k, X(i, 1:T))
      [numSequences, T] = size(X);

      if ~exist('initLogPr', 'var') || isempty(initLogPr)
        initLogPr = obj.Pi.logprob;
      end
      if size(initLogPr, 2) == 1
        initLogPr = initLogPr';
      end
      if size(initLogPr, 1) == 1
        initLogPr = repmat(initLogPr, numSequences, 1);
      end

      logJointPr = zeros(numSequences, 1);
      if nargout > 1
        finalLogPr = zeros(numSequences, obj.m);
      end
      for i = 1:numSequences
        logO = obj.compLogO(X(i, :));
        q = initLogPr(i, :)';
        for t = 1:T
          q = logsumexp(bsxfun(@plus, obj.T.logprob, (logO(:, t) + q)'), 2);
        end
        logJointPr(i) = logsumexp(q);
        if nargout > 1
          finalLogPr(i, :) = q';
        end
      end
    end


    function logCondPr = computeLogCondProb(obj, targSeqs, condSeqs)
      % COMPUTELOGCONDPROB   Compute conditional probability of the targSeqs
      %                      given the condSeqs.
      [numSequences, condSeqSize] = size(condSeqs);
      targSeqSize = size(targSeqs, 2);

      [condLogJointPr, condFinalLogPr] = obj.computeLogJointProb(condSeqs);
      targetLogJointPs = obj.computeLogJointProb(targSeqs, condFinalLogPr);
      logCondPr = targetLogJointPs - condLogJointPr;
    end


    function predObsPdf = predObsPdf(obj, currState, numSteps)
      % Gets the predictive PDF of the observation after numSteps steps.
      % currState could be an integer indicating the current state or a vector
      % indicating the probabilities of each state.
      % The output predObsPdf is a chebfun object.
      if numel(currState) == 1
        p = zeros(obj.m, 1);
        p(currState) = 1;
        currState = p;
      end
      O = obj.getChebO();
      Tsteps = obj.T.prob^numSteps;
      stateDistro = Tsteps * currState;
      predObsPdf = cat(2, O{:}) * stateDistro;
    end


    function obsPdf = nextObsPdfGivenSeq(obj, condSeq)
      % NEXTOBSPDFGIVENSEQ  Create a chebfun handle for p(X_{t+1} | X_{1:t}).
      %                     Here, condSeq is X_{1:t}.
      [condLogJointPr, condFinalLogPr] = obj.computeLogJointProb(condSeq);

      % Create a closure that computes p(X_{t+1} | X_{1:t})
      function condProb = computeNextPdf(X)
        nextCondLogJointPr = obj.computeLogJointProb(X(:), condFinalLogPr);
        condProb = reshape(exp(nextCondLogJointPr - condLogJointPr), size(X));
      end

      obsPdf = chebfun(@(x) computeNextPdf(x), obj.obsBoundary, ...
                       'splitting', 'on');
      obsPdf = obsPdf ./ sum(obsPdf);
    end


  end % methods


end % classdef
