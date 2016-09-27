function [optBW, kdeFuncH] = kdePickBW(X, smoothness, params, bwLogBounds)
% This picks a bandwidth for the KDE. We use k-fold cross validation in the
% range specified by bwLogBounds.
% If params.getKdeFuncH is True, then it also returns a function handle for the
% kde with the optimal bandiwidth.

  % prelims
  numData = size(X, 1);
  numDims = size(X, 2);
  USE_DIRECT = false;

  % Shuffle the data
  shuffleOrder = randperm(numData);
  X = X(shuffleOrder, :);

  % Obtain the Standard Deviation of X
  if numData == 1
    stdX = 1;
  else
    stdX = norm( std(X) );
  end

  % Set default parameter values
  if ~exist('smoothness', 'var') | isempty(smoothness)
    smoothness = 'gauss';
  end
  if ~exist('params', 'var')
    params = struct;
  end
  if ~isfield(params, 'doBoundaryCorrection')
    params.doBoundaryCorrection = false;
  end
  if ~isfield(params, 'numPartsKFCV')
    params.numPartsKFCV = 12;
  end
  if ~isfield(params, 'numAvgPartsKFCV')
    params.numAvgPartsKFCV = 2;
  end
  if ~isfield(params, 'numCandidates')
    params.numCandidates = 30;
  end
  if ~isfield(params, 'getKdeFuncH')
    params.getKdeFuncH = true;
  end
  if ~exist('bwLogBounds', 'var') || isempty(bwLogBounds)
    if isfield(params, 'bwLogBounds')
      bwLogBounds = params.bwLogBounds;
    else
      bwLogBounds = log( [0.01 0.5] * stdX * (numData/100)^(-1/(4+numDims)) );
%       bwLogBounds(2) = min(bwLogBounds(2), 1);
%       bwLogBounds(1) = max(bwLogBounds(1), 0.01);
    end
  end

  if USE_DIRECT
  % Use DiRect to Optimize over h
    diRectBounds = bwLogBounds;
    options.maxevals = params.numCandidates;
    kFoldFunc = @(t) kdeKFoldCV(t, X, smoothness, params);
    [~, maxPt, history] = diRectWrap(kFoldFunc, diRectBounds, options);
    optBW = exp(maxPt);
  else
  % Use just ordinary KFold CV
    bwCandidates = linspace(bwLogBounds(1), bwLogBounds(2), ...
                            params.numCandidates);
    bestLogLikl = -inf;
    optBW = 0.2 * numData ^(-4/(4+numDims));
    for candIter = 1:params.numCandidates
      currLogLikl = kdeKFoldCV(bwCandidates(candIter), X, smoothness, params);
      if currLogLikl > bestLogLikl
        bestLogLikl = currLogLikl;
        optBW = exp( bwCandidates(candIter) );
      end
    end
  end

  % smooth a little bit.
%   optBW = 2*optBW;
%   optBW = 20*stdX * numData^(-4/(4+numDims));

  % Return a function handle
  if params.getKdeFuncH
    kdeFuncH = kdeGivenBW(X, optBW, smoothness, params);
  else
    kdeFuncH = [];
  end
end



function avgLogLikl = kdeKFoldCV(logBW, X, smoothness, params)

  h = exp(logBW);
  numPartsKFCV = params.numPartsKFCV;
  logLikls = zeros(numPartsKFCV, 1);
  numData = size(X, 1);
  numDims = size(X, 2);

  if isfield(params, 'sampleWeights')
    W = params.sampleWeights;
  end

  for kFoldIter = 1:params.numAvgPartsKFCV
    % Set the partition up
    testStartIdx = round( (kFoldIter-1)*numData/numPartsKFCV + 1 );
    testEndIdx = round( kFoldIter*numData/numPartsKFCV );
    trainIndices = [1:(testStartIdx-1), (testEndIdx+1):numData]';
    testIndices = [testStartIdx: testEndIdx]';
    numTestData = testEndIdx - testStartIdx;
    numTrainData = numData - numTestData;
    % Separate Training and Validation sets
    Xtr = X(trainIndices, :);
    Xte = X(testIndices, :);
    if isfield(params, 'sampleWeights')
      params.sampleWeights = W(trainIndices);
    end
    % Now Obtain the kde using Xtr
    kdeTr = kdeGivenBW(Xtr, h, smoothness, params);
    % Compute Log Likelihood
    Pte = kdeTr(Xte);
    logPte = log(Pte);
    isInfLogPte = isinf(logPte);
    % If fewer than 10% are infinities, then remove them
    if sum(isInfLogPte) < 0.1*numTestData
      logPte = logPte(~isInfLogPte);
      logLikls(kFoldIter) = mean(logPte);
    else
%       fprintf('%d/%d  =%0.4f, points had -inf loglikl. Quitting\n', ...
%         sum(isInfLogPte), numTestData, sum(isInfLogPte)/numTestData);
      logLikls(kFoldIter) = -inf;
      break;
    end
  end

  avgLogLikl = mean(logLikls);

end
