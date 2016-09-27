function X = sampleGaussMixtures(numSamples, numDims, numMix)
% Samples from a GMM

  centres = repmat(linspace(0, 1, numMix)', 1, numDims);
  sigma = 0.22*sqrt(numDims)/max(numMix-1, 1);
  
  % First obtain the Gaussians
  Z = sigma * randn(numSamples, numDims);

  % Now determine the centres
  centreIdxs = randi([1 numMix], numSamples, 1);

  X = Z + centres(centreIdxs, :);
%   sigma, centres, centres(centreIdxs, :), X,
end

