function [samplers, pdfs] = genNPpdfs(numPdfs, boundary)
% A script to generate some pdfs.
% Inputs:
%  - numPdfs: the number of pdfs
%  - boundary: a 1x2 array specifying the interval.
% Outputs:
%  - samplers: a numPdfs x 1 cell where samplers{i}(n) will generate n samplers from
%    the i^th pdf.
%  - pdfs: A numPdfs x 1 cell array of functions where pdfs{i}(t) will evaluate the
%    i^th pdf at the points in t.
% numPdfs should be less than or equal to 9.

  UNIF_PROB = 0.0;
  TURB_PROB = 0.3;
  GMM_PROB = 1 - UNIF_PROB - TURB_PROB;

  if nargin < 2 | isempty(boundary)
    boundary = [0 1];
  end

  % Some parameters
  width = boundary(2) - boundary(1);
  numTurbMixtures = 20;
  turbSigmas = 0.05 * width * ones(numTurbMixtures, 1);
  turbCentres = linspace(boundary(1) + 0.1 * width, ...
                  boundary(2) - 0.1 * width, ...
                  numTurbMixtures)';

  numGaussians = 4;
%   mixProbs = [0.0  0.0  0.0  1.0; ...
%               1.0  0.0  0.0  0.0; ...
%               0.0  0.0  1.0  0.0; ...
%               0.0  1.0  0.0  0.0; ...
%               0.7  0.0  0.3  0.0; ...
%               0.0  0.3  0.0  0.7; ...
%               0.0  0.0  0.5  0.5; ...
%               0.5  0.5  0.0  0.0; ...
%               0.4  0.1  0.1  0.4];
%   mixSigmas = 0.1 * width * ones(numGaussians, 1);
  numGaussians = 7;
  mixProbs = [ ...
              0.0  0.7  0.0  0.0  0.3  0.0  0.0; ...
              0.0  0.0  0.3  0.0  0.0  0.7  0.0; ...
              0.0  0.0  0.0  0.0  0.5  0.0  0.5; ...
              0.5  0.0  0.5  0.0  0.0  0.0  0.0; ...
              0.4  0.0  0.1  0.0  0.1  0.0  0.4; ...
              0.0  0.0  0.0  0.0  0.0  0.0  1.0; ...
              1.0  0.0  0.0  0.0  0.0  0.0  0.0; ...
              0.0  0.0  0.0  0.0  1.0  0.0  0.0; ...
              0.0  0.0  1.0  0.0  0.0  0.0  0.0; ...
              0.0  0.2  0.2  0.2  0.2  0.2  0.0];
  mixSigmas = 0.07 * width * ones(numGaussians, 1);
  mixProbs = mixProbs(1:numPdfs, :);

  % Centers
  mixCentres = linspace(boundary(1) + width * 0.2, ...
                        boundary(2) - width * 0.2, ...
                        numGaussians)';

  % Compute normalizing constant for the turbulence GMMs.
  turbUnNormPdf = @(t) gmmPdf(t, turbCentres, turbSigmas, ...
                              ones(numTurbMixtures, 1) / numTurbMixtures);
  turbNorm = integral(turbUnNormPdf, boundary(1), boundary(2));
  turbPdf = @(t) turbUnNormPdf(t) / turbNorm;

  samplers = cell(numPdfs, 1);
  pdfs = cell(numPdfs, 1);

  for i = 1:numPdfs
    samplers{i} = @(n) sampleMixPdf(n, UNIF_PROB, TURB_PROB, GMM_PROB, ...
                                    turbCentres, mixCentres, turbSigmas, ...
                                    mixSigmas, mixProbs(i, :)', boundary);

    % compute the normalizing constant for the pdf
    currGMMUnNormPdf = @(t) gmmPdf(t, mixCentres, mixSigmas, mixProbs(i, :)');
    currGridNorm = integral(currGMMUnNormPdf, boundary(1), boundary(2));
    currGMMPdf = @(t) currGMMUnNormPdf(t)/currGridNorm;

    pdfs{i} = @(t) UNIF_PROB * 1/width + ...
                   TURB_PROB * turbPdf(t) + ...
                   GMM_PROB * currGMMPdf(t);
  end

end



function val = gmmPdf(tt, centres, sigmas, mixProbs)

  [s1, s2] = size(tt);
  t = tt(:);

  numCentres = size(centres, 1);
  n = size(t, 1);

  gaussPdfVals = zeros(n, numCentres);
  for i = 1:numCentres
    gaussPdfVals(:,i) = normpdf(t, centres(i), sigmas(i));
  end
  mixPdfVals = bsxfun(@times, gaussPdfVals, mixProbs');
  val = sum(mixPdfVals, 2);

  val = reshape(val, s1, s2);

end



function samples = sampleMixPdf(numSamples, UNIF_PROB, TURB_PROB, GMM_PROB, ...
                                turbCentres, mixCentres, turbSigmas, ...
                                mixSigmas, mixProbs, boundary)

  % preliminaries
  numTurbMixtures = size(turbCentres, 1);
  width = boundary(2) - boundary(1);

  p1 = rand(numSamples, 1);
  ind1 = bindex(p1, cumsum([0 UNIF_PROB, TURB_PROB, GMM_PROB]));

  uniIdxs = (ind1 == 1);
  turbIdxs = (ind1 == 2);
  gmmIdxs = (ind1 == 3);
  numUniIdxs = sum(uniIdxs);
  numTurbIdxs = sum(turbIdxs);
  numGMMIdxs = sum(gmmIdxs);

  % Sample the uniform points
  uniSamples = boundary(1) + (boundary(2) - boundary(1)) * rand(numUniIdxs, 1);

  % Sample turbulence points
  turbProbs = ones(numTurbMixtures, 1) / numTurbMixtures;
  turbSamples = sampleTruncGMM(numTurbIdxs, ...
                               turbCentres, turbSigmas, turbProbs, ...
                               boundary);

  % Sample GMM points
  gmmSamples = sampleTruncGMM(numGMMIdxs, ...
                              mixCentres, mixSigmas, mixProbs, boundary);

  samples = zeros(numSamples, 1);
  samples(uniIdxs) = uniSamples;
  samples(turbIdxs) = turbSamples;
  samples(gmmIdxs) = gmmSamples;

end
