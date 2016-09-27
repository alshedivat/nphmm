% Tests for EM fitting of npHMM.
% Just checks that everything runs.

addpath(genpath('../lib'));

clc;
clear all;
close all;

chebfun_setup();

% Fix the seed for reproducibility
rng(42);

m = 5;
N = 100;
l = 3;
obsBoundary = [-1, 1];

% Create ground-truth HMM
options = [];
options.use_cheb = true;
hmm = npHMM(m, obsBoundary, [], [], [], options);
hmm.randInit();

% Generate sequences
[X, H] = hmm.sample(N, l);

% Fit another HMM
params.verbose = false;
estHMM = npHMM_EM(X, m, obsBoundary, params);

% Testing the transition probabilities
fprintf('Transition matrix error: %0.4f\n', ...
        norm(hmm.T.prob - estHMM.T.prob, 'fro'));

% Testing for predictive probability
l = 10;
T = hmm.sample(1, l);
G = linspace(obsBoundary(1), obsBoundary(2))';
nextPdfWithObsTrue = hmm.nextObsPdfGivenSeq(T);
nextPdfWithEst = estHMM.nextObsPdfGivenSeq(T);
fprintf('Estimation error for the next pdf: %0.4f.\n', ...
        norm(nextPdfWithObsTrue - nextPdfWithEst, 'fro'));
fprintf('Estimated Integrals: True=%0.4f, Est=%0.4f.\n', ...
        sum(nextPdfWithObsTrue), sum(nextPdfWithEst));
