classdef npObsHMM < handle
% Nonparametric HMM.
% The primary representation is observable (b1, binf, Bx, U).

  properties
    m;    % number of hidden states
    estM; % estimated number of hidden states
    spectralGap; % spectral gap between sigma(m) and sigma(m+1)

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
  end % properties


  methods


    function obj = npObsHMM(obsBoundary, b1, binf, Bx, U, P1, P21, P321, ...
                            estM, spectralGap)
      if nargin > 0
        obj.b1 = b1;
        obj.binf = binf;
        obj.Bx = Bx;
        obj.U = U;
        obj.m = size(U, 2);
        obj.obsBoundary = obsBoundary;

        if exist('estM', 'var'); obj.estM = estM; end;
        if exist('spectralGap', 'var'); obj.spectralGap = spectralGap; end;

        obj.P1 = [];
        obj.P21 = [];
        obj.P321 = [];
        if exist('P1', 'var'), obj.P1 = P1; end
        if exist('P21', 'var'), obj.P21 = P21; end
        if exist('P321', 'var'), obj.P321 = P321; end
      end
    end


    function [logJointPr, finalLogBt] = computeLogJointProb(obj, X, initLogBt)
      % COMPUTELOGPROB  Compute log joint probability for each sequence using
      %                 the observable representation. Rows of X are sequences.
      %
      % NOTE: finalLogBt is generally a complex-valued vector.
      [numSequences, T] = size(X);

      if ~exist('initLogBt', 'var') || isempty(initLogBt)
        % Note: Elements of b1 is are not necessarily nonnegative.
        %       Hence, log(b1) is generally a complex-valued vector.
        initLogBt = log(obj.b1);
      end
      if size(initLogBt, 2) == 1
        initLogBt = initLogBt';
      end
      if size(initLogBt, 1) == 1
        initLogBt = repmat(initLogBt, numSequences, 1);
      end

      logJointPr = zeros(numSequences, 1);
      if nargout > 1
        finalLogBt = zeros(numSequences, obj.m);
      end
      for i = 1:numSequences
        q = initLogBt(i, :);
        for t = 1:T
          q = logsumexp(bsxfun(@plus, log(obj.Bx(X(i, t))), q), 2)';
        end
        logJointPr(i) = real(logsumexp(q' + log(obj.binf)));
        if nargout > 1
          finalLogBt(i, :) = q;
        end
      end
    end


    function logCondPr = computeLogCondProb(obj, targSeqs, condSeqs)
      % COMPUTELOGCONDPROB   Compute conditional probability of the targSeqs
      %                      given the condSeqs.
      [numSequences, condSeqSize] = size(condSeqs);
      targSeqSize = size(targSeqs, 2);

      [condLogJointPr, condFinalLogBt] = obj.computeLogJointProb(condSeqs);
      targetLogJointPr = obj.computeLogJointProb(targSeqs, condFinalLogBt);
      logCondPr = targetLogJointPr - condLogJointPr;
    end


    function obsPdf = nextObsPdfGivenSeq(obj, condSeq)
      % NEXTOBSPDFGIVENSEQ  Create a chebfun handle for p(X_{t+1} | X_{1:t}).
      %                     Here, condSeq is X_{1:t}.

      [condLogJointPr, condFinalLogBt] = obj.computeLogJointProb(condSeq);

      function condProb = computeNextPdf(X)
        nextLogJointPr = obj.computeLogJointProb(X(:), condFinalLogBt);
        condProb = reshape(exp(nextLogJointPr - condLogJointPr), size(X));
      end

      obsPdf = chebfun(@(x) computeNextPdf(x), obj.obsBoundary);
      obsPdf = obsPdf / sum(obsPdf);
    end


  end % end methods


end % end classdef

