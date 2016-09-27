function hmm = learnNPHMM(X, m, method, obsBoundary, params)
  % Fit npHMM to the given data using provided method.
  if strcmp(method, 'EM')
    hmm = npHMM_EM(X, m, obsBoundary, params);
  elseif strcmp(method, 'SPEC')
    hmm = npHMM_SPEC(X, m, obsBoundary, params);
  else
    error('Unknown fitting method: %s', method);
  end
end
