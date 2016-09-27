% Unit test for getObsRepr.m

addpath(genpath('../lib'));

clear all;
close all;
clc;

chebfun_setup();

rng(123);

m = 6;
obsBoundary = [-1, 1];
width = obsBoundary(2) - obsBoundary(1);

hmm = npHMM(m, obsBoundary);
hmm.randInit();

% Construct joint probabilities
[P1, P21, P321] = hmm.constructJointProbs();
[obsHMM, b1, binf, Bx, U] = computeObsRepr(P1, P21, P321, m, obsBoundary);


UtO = U' * cat(2, hmm.O.pdfs{:});
fprintf('\nU''O, rank: %d, cond: %0.4f.\n', rank(UtO), cond(UtO));
fprintf('Testing computed parameters: b1, binf, Bx.\n');
fprintf('The following quantities should be close to zero:\n');
fprintf('  ||b_1 - U''O*pi|| = %ef.\n', norm(b1 - UtO*hmm.Pi.prob) );
fprintf('  ||b_inf - inv(U''O)*1|| = %0.ef\n', norm(binf' - ones(1, m) / UtO));
xtest = 0.1;
tic,
Bxtest = Bx(xtest);
toc,
Axtest = hmm.Ax(xtest);
fprintf('  ||Bx(%.1f) - (U''O)*Ax(%.1f)*inv(U''O)||_fro = %0.ef\n', ...
         xtest, xtest, norm(Bxtest-UtO*Axtest/UtO, 'fro'));


fprintf('\nTesting Joint & Conditional Probabilities.\n');
l = 3; tl = 1;
[XX, HH] = hmm.sample(1, l + tl);
X = XX(:, 1:l); H = HH(:, 1:l);
Xtarg = XX(:, (l+1):end); Htarg = HH(:, (l+1):end);
logJointTP = hmm.computeLogJointProb(X);
logJointOP = obsHMM.computeLogJointProb(X);
fprintf('LJP:: true-params: %0.4f, obs-params: %0.4f,   diff: %0.4f\n', ...
        logJointTP, logJointOP, abs(logJointTP - logJointOP));

logCondTP = hmm.computeLogCondProb(Xtarg, X);
logCondOP = obsHMM.computeLogCondProb(Xtarg, X);
fprintf('LCP:: true-params: %0.4f, obs-params: %0.4f,   diff: %0.4f\n', ...
  logCondTP, logCondOP, abs(logCondTP-logCondOP));


fprintf('\nTesting for Next Step predictive PDF.\n');
tic,
nextPdfWithState = hmm.predObsPdf(H(1, end), tl);
toc,
tic,
nextPdfWithObsOP = obsHMM.nextObsPdfGivenSeq(X);
toc,
tic,
nextPdfWithObsTP = hmm.nextObsPdfGivenSeq(X);
toc,
figure;
plot(nextPdfWithState, 'b-.', 'linewidth', 1); hold on;
plot(nextPdfWithObsTP, 'g-', 'linewidth', 1.6);
plot(nextPdfWithObsOP, 'r--', 'linewidth', 1);
legend('Using the last state', 'Using observations true-params', ...
       'Using observations obs-params');
title('Next Observation Pdf');
fprintf('Estimated Integrals: True=%0.4f, Obs=%0.4f.\n', ...
        sum(nextPdfWithObsTP), sum(nextPdfWithObsOP));

