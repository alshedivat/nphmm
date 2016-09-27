% Unit tests for SPEC fitting of npHMM.
% Just checks that everything runs and outputs different statistics.

addpath(genpath('../lib'));

clear all;
close all;
clc;

chebfun_setup();

% Fix seed
rng(42);
TEST_DENSITY_ESTIMATION = true;
% TEST_DENSITY_ESTIMATION = false;

% Problem parameters
m = 5;
obsBoundary = [-1.0, 1.0];
width = obsBoundary(2) - obsBoundary(1);

hmm = npHMM(m, obsBoundary);
hmm.randInit();

% Approximation parameters
options = [];
options.verbose = true;
options.tol = 2e-2;

% Estimation parameters
options.estimate_P1 = 1;
options.estimate_P21 = 1;
estM = []; % m to be used for estimation.

% experiment parameters
N = 1000; % Number of samples
T = 3; % Sequence length

fprintf('Generating Samples ... ');
[X, H] = hmm.sample(N, T);
fprintf('Done.\n\n');

% Simple test: Estimate densities and check if they are ok
if TEST_DENSITY_ESTIMATION
  fprintf('Constructing true joint densities ...\n');
  [P1c, P21c, P321c] = hmm.constructJointProbs();
  fprintf('Done.\n\n');

  fprintf('Constructing KDE joint densities ...\n');
  [BW1, P1hat] = kdePickBW(X(:, 1));
  [BW2, P21hat] = kdePickBW(X(:, [2 1]));
  [BW3, P321hat] = kdePickBW(X(:, [3 2 1]));
  fprintf('BW1 = %0.4f, BW2 = %0.4f, BW3 = %0.4f\n', BW1, BW2, BW3);

  tic
  eP1c = chebfun(@(x) kde1cheb(x, P1hat), obsBoundary);
  eP1c = eP1c / sum(eP1c);
  toc, tic
  eP21c = chebfun2(@(x, y) kde2cheb(x, y, P21hat), ...
                   repmat(obsBoundary, 1, 2), 'eps', 1e-8);
  eP21c = eP21c / sum2(eP21c);
  toc, tic
  eP321c = chebfun3(@(x, y, z) kde3cheb(x, y, z, P321hat), ...
                    repmat(obsBoundary, 1, 3), ...
                    'tech', 'minSamples', 10, 'fiberDim', 2);
  eP321c = eP321c / sum3(eP321c);
  toc
  eeP21c = sum(eP321c, 1);
  eeP1c = sum(eeP21c, 2);
  fprintf('Done.\n\n');

  fprintf('Errors: P1: %0.4f, P21: %0.4f, P21e: %0.4f.\n', ...
          norm(P1c - eP1c), norm(P21c - eP21c), norm(eP21c - eeP21c));

  [~, b1, binf, Bx, U, Bxc] = ...
    computeObsRepr(P1c, P21c, P321c, m, obsBoundary, options);
  [~, eb1, ebinf, eBx, eU, eBxc] = ...
    computeObsRepr(eP1c, eP21c, eP321c, m, obsBoundary, options);

  fprintf('Errors: b1: %0.4f, binf: %0.4f, Bx: %0.4f, U: %0.4f.\n\n', ...
          norm(b1 - eb1), norm(binf - ebinf), norm(Bxc - eBxc), norm(U - eU));

  figure;
    plot(P1c, 'r'); hold on; plot(eP1c, 'b'); plot(eeP1c, 'g');
    plot(X(:, 1), 0.2 * rand(N, 1), 'kx');
  figure;
    subplot(1, 3, 1); plot(P21c); title('True');
    subplot(1, 3, 2); plot(eP21c); title('From P21');
    subplot(1, 3, 3); plot(eeP21c); title('From P321');
end

% Estimate the parameters
fprintf('Estimating the HMM:\n======================================\n');
tic,
estHMM = npHMM_SPEC(X, estM, obsBoundary, options);
endTime = toc;
fprintf('Time taken to learn npHMM: %0.4f s\n', endTime);

bestHMM = npObsHMM(obsBoundary, b1, binf, Bx, U, P1c, P21c, P321c);

% Testing for predictive probability.
l = 10;
T = hmm.sample(1, l);
nextPdfWithObsTrue = hmm.nextObsPdfGivenSeq(T);
nextPdfWithBest = bestHMM.nextObsPdfGivenSeq(T);
nextPdfWithEst = estHMM.nextObsPdfGivenSeq(T);
figure; hold on;
  plot(nextPdfWithObsTrue, 'r-', 'linewidth', 2.0);
  plot(nextPdfWithBest, 'g--', 'linewidth', 1.2);
  plot(nextPdfWithEst, 'b-', 'linewidth', 1);
hold off
fprintf('Estimation error for next pdf: %0.4f.\n', ...
        norm(nextPdfWithObsTrue - nextPdfWithEst, 'fro'));
fprintf('Estimation error for next pdf lower bound: %0.4f.\n', ...
        norm(nextPdfWithObsTrue - nextPdfWithBest, 'fro'));
fprintf('Estimated Integrals: True=%0.4f, Best=%0.4f, Est=%0.4f.\n', ...
        sum(nextPdfWithObsTrue), sum(nextPdfWithBest), sum(nextPdfWithEst));
