function samples = sampleTruncGMM(numSamples, mus, sigmas, mixProbs, truncBoundaries)
  totalCollSamples = 0;
  samples = zeros(numSamples, 1);

  budget = numSamples;
  while totalCollSamples < numSamples
    numRemSamples = max(10, ceil(3 *(numSamples - totalCollSamples)));
    S = sampleGMM(numRemSamples, mus, sigmas, mixProbs);
    valid = (S>truncBoundaries(1)) & (S<truncBoundaries(2));
    validSamples = S(valid);


    if size(validSamples, 1) >= numSamples - totalCollSamples
      endIdx = numSamples;
    else
      endIdx = totalCollSamples + size(validSamples,1);
    end
    samples( (totalCollSamples+1): endIdx) = ...
      validSamples( 1:(endIdx-totalCollSamples), :);  

    totalCollSamples = totalCollSamples + size(validSamples,1);
  end

end
