function [samples] = sampleGMM(numSamples, mus, sigmas, mixProbs)
  p = rand(numSamples, 1);
  ind = bindex(p, cumsum([0 mixProbs']));
  samples = sigmas(ind) .* randn(numSamples, 1) + mus(ind);
end

