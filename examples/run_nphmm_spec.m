% Spectral learning of npHMM

clear all;
close all;
clc;

chebfun_setup();

% Fix seed
rng(12345);

% Problem parameters
m = 4; % m to be used for estimation
obsBoundary = [-1.0, 1.0];
width = obsBoundary(2) - obsBoundary(1);

% Load data
fprintf('Loading data ... ');
load(sprintf('data_m%d.mat', m));
fprintf('Done.\n');

% Experiment parameters
Ntr = size(Xtr, 1); % number of training sequences
Nts = size(Xts, 1); % number of testing sequences

numN = 10; % number of Ns to consider
logMinN = log10(1000);
logMaxN = log10(Ntr); % log10 number of samples
numSamplesRange = ceil(logspace(logMinN, logMaxN, numN));
numRuns = 5; % number of runs for each sample size

% Training options
options = [];
options.verbose = 0;

% Auxiliary
x = chebfun('x', obsBoundary);

% Compute the optimal values (using the true HMM)
fprintf('Computing estimates for the true model ... ');
TrueErr = 0;
nextPdfTrue = cell(Nts, 1);
for j = 1:Nts
    % True predictive probability
    nextPdfTrue{j} = hmm.nextObsPdfGivenSeq(Xts(j,1:end-1));

    % Predictive error of the true model
    TrueErr = TrueErr + abs(mean(x .* nextPdfTrue{j}) - Xts(j,end));
end
TrueErr = TrueErr / Nts;
fprintf('Done.\n');

L1 = zeros(2, numN);
EstErr = zeros(2, numN);
TrainTime = zeros(2, numN);
for i = 1:numN
    N = numSamplesRange(i);
    fprintf('--------------------------------------------------\n', N);
    fprintf('N = %d\n', N);
    fprintf('--------------------------------------------------\n', N);

    currL1 = zeros(1, numRuns);
    currEstErr = zeros(1, numRuns);
    currTrainTime = zeros(1, numRuns);
    for r = 1:numRuns
        fprintf('Run %d:\n', r);

        % Select a subset of data
        rng(42 * r);
        idx = randsample(Ntr, N);
        Y = Xtr(idx, :);

        % Train the model
        fprintf('\tTraining NP-HMM-SPEC ... ');
        tic, estHMM = learnNPHMM(Y, m, 'SPEC', obsBoundary, options);
        currTrainTime(r) = toc;
        fprintf('Done.\n');

        % Test the model
        fprintf('\tTesting NP-HMM-SPEC ... ');
        for j = 1:Nts
            nextPdfEst = estHMM.nextObsPdfGivenSeq(Xts(j,1:end-1));

            % Predictive probability L1 error
            currL1(r) = currL1(r) + norm(nextPdfTrue{j} - nextPdfEst, 1);

            % Predictive error
            currEstErr(r) = currEstErr(r) + ...
                            abs(mean(x .* nextPdfEst) - Xts(j,end));
        end
        currL1(r) = currL1(r) / Nts;
        currEstErr(r) = currEstErr(r) / Nts;
        fprintf('Done.\n');

        fprintf('\tL1: %.4f,\t', currL1(r));
        fprintf('EstErr: %.4f\n', currEstErr(r));
        fprintf('\tTrainTime: %.2f\n', currTrainTime(r));
    end
    L1(:,i) = [mean(currL1); std(currL1)];
    EstErr(:,i) = [mean(currEstErr); std(currEstErr)];
    TrainTime(:,i) = [mean(currTrainTime); std(currTrainTime)];

    fprintf('\n');
    fprintf('L1: %.4f +- %.4f\n', L1(1,i), L1(2,i));
    fprintf('EstErr: %.4f +- %.4f,\t', EstErr(1,i), EstErr(2,i));
    fprintf('TrueErr: %.4f\n', TrueErr);
    fprintf('TrainTime: %.4f +- %.4f\n', TrainTime(1,i), TrainTime(2,i));

end
