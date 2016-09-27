% Generate synthetic data

% Fix seed
rng(42);

% Data parameters
m = 5;  % number of hidden states
T = 3;  % sequence length
Ntr = 100000;
Nts = 100;
obsBoundary = [-1.0 1.0];

% Create HMM
hmm = npHMM(m, obsBoundary);
hmm.randInit();

% Generate data
[Xtr, ~] = hmm.sample(Ntr, T);
[Xts, ~] = hmm.sample(Nts, T);

% Save data
save('data/synthetic.mat', 'hmm', 'Xtr', 'Xts');
