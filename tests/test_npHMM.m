% Unit tests for npHMM

addpath(genpath('../lib'));

clc;
clear all;
close all;

chebfun_setup();

% Set the verbosity level
verbose = 1;

% Fix the seed for reproducibility
rng(101);

% Initialize HMM
m = 6;
obsBoundary = [-1, 1];
HMM = npHMM(m, obsBoundary);
HMM.randInit();

% --------------------------------------------------------------------------- %
% Simple tests
% --------------------------------------------------------------------------- %

% Plot the emission PDFs
if verbose
  figure; plot(cat(2, HMM.O.pdfs{:}));
  title('Emission Probabilities');
  fprintf('Cond(T): %0.5f\nSum of chebpdfs: %s\n', ...
          cond(HMM.T.prob), mat2str(sum(cat(2, HMM.O.pdfs{:}))));
  fprintf('\n');
end

% Test sampling from the model
N = 100;
l = 10;
[X, H] = HMM.sample(N, l);

assert(all(size(X) == [N, l]));
assert(all(all(X > obsBoundary(1))));
assert(all(all(X < obsBoundary(2))));

assert(all(size(H) == [N, l]));
assert(all(all(H >= 1)));
assert(all(all(H <= m)));

% Test the forward pass of the forward-backward algorithm
params = []; params.scaled = 1;
[logA, logB, logG, logX, loglik] = HMM.fwdback(X(1, :), params);
assert(all(size(logA) == [m, l]));
assert(all(size(logB) == [m, l]));
assert(all(size(logG) == [m, l]));
assert(all(size(logX) == [m, m]));
assert(all(abs(logsumexp(logA, 1)) < 2 * eps));

% --------------------------------------------------------------------------- %
% Test log likelihoods, joint probabilities, and conditional probabilities
% --------------------------------------------------------------------------- %
l = 30; tl = 5; %tl = ceil(1*l);
[XX, HH] = HMM.sample(N, l + tl);
X = XX(:, 1:l); H = HH(:, 1:l);
Xtarg = XX(:, (l+1):end); Htarg = HH(:, (l+1):end);
% Just sample randomly
YY = mean(obsBoundary) + rand(N, l+tl) * (obsBoundary(2) - mean(obsBoundary));
Y = YY(:, 1:l);
Ytarg = YY(:, (l+1):end); % To test for conditional probability.

% Test log likelihood
loglikX = HMM.computeLoglik(X);
loglikY = HMM.computeLoglik(Y);
assert(sum(loglikX) > sum(loglikY));
if verbose
  fprintf('Log likelihood (length = %d)\n', l);
  fprintf('  Sampled Seq: %0.5f +/- %0.5f.\n', ...
          mean(loglikX), std(loglikX) / sqrt(N));
  fprintf('  Random Seq : %0.5f +/- %0.5f.\n', ...
          mean(loglikY), std(loglikY) / sqrt(N));
  fprintf('\n');
end

% Test log joint probabilities
logJointX = HMM.computeLogJointProb(X);
logJointY = HMM.computeLogJointProb(Y);
assert(sum(logJointX) > sum(logJointY));
if verbose
  fprintf('Log joint probabilities (length = %d) \n', l);
  fprintf('  Sampled Seq: %0.5f +/- %0.5f.\n', ...
          mean(logJointX), std(logJointX) / sqrt(N));
  fprintf('  Random Seq : %0.5f +/- %0.5f.\n', ...
          mean(logJointY), std(logJointY) / sqrt(N));
  fprintf('\n');
end

% Test log conditional probabilities probabilities
logCondX = HMM.computeLogCondProb(Xtarg, X);
logCondY = HMM.computeLogCondProb(Ytarg, X);
assert(sum(logCondX) > sum(logCondY));
if verbose
  fprintf('Log conditional Pr (length=%d, target-length=%d)\n', l, tl);
  fprintf('  Sampled Seq: %0.5f +/- %0.5f.\n', ...
          mean(logCondX), std(logCondX) / sqrt(N));
  fprintf('  Random Seq : %0.5f +/- %0.5f.\n', ...
          mean(logCondY), std(logCondY) / sqrt(N));
  fprintf('\n');
end

% --------------------------------------------------------------------------- %
% Check predictive probabilities
% --------------------------------------------------------------------------- %
if verbose
  numSteps = 1;
  predPdfSums = zeros(1, m);
  predObsFig = figure;
  for i = 1:m
    predPdf = HMM.predObsPdf(i, numSteps);
    if i <= m && numSteps <= 1
      subplot(2, 3, i); plot(predPdf);
      axis([obsBoundary, 0, 1.1*max(max(cat(2, HMM.O.pdfs{:})))]);
    end
    predPdfSums(i) = sum(predPdf);
    titleStr = sprintf('predObs(%d)', i);
    title(titleStr);
  end
  fprintf('Sum of predictive PDFs: %s.\n\n', mat2str(predPdfSums));
end

% Test for predictive probability using sequences
if verbose
  nextPdfWithState = HMM.predObsPdf(H(1, end), 1);
  nextPdfWithObs = HMM.nextObsPdfGivenSeq(X(1,:));
  figure;
  plot(nextPdfWithState, 'b-', 'linewidth', 1.6); hold on;
  plot(nextPdfWithObs, 'r--', 'linewidth', 1);
  legend('Using the last state', 'Using only observations');
  title('Next Observation Pdf');
end

% --------------------------------------------------------------------------- %
% Test computation of P1, P21, and P321
% --------------------------------------------------------------------------- %
[P1, P21, P321] = HMM.constructJointProbs(verbose);

% Make sure P21 and P321 are computed correctly
[X, ~] = HMM.sample(100, 3);
assert(sum(log(P21(X(:, 2), X(:, 1)))) > sum(log(P21(X(:, 1), X(:, 2)))));
assert(sum(log(P321(X(:, 3), X(:, 2), X(:, 1)))) > ...
       sum(log(P321(X(:, 1), X(:, 2), X(:, 3)))));
