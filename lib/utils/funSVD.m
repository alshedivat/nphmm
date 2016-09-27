function [U, S, V] = funSVD(f, gridX, gridY)
% Input:
% f: a two dimensional function - it takes a nx2 vector as input and returns an nx1
%    vector as the output.
% gridX, gridY: NVxNU grids indicating where f should be evaluated for the
% approximation.
% Output:
% U, V: Two arrays 

  [NV, NU] = size(gridX);

  allGridPts = [gridX(:), gridY(:)];
  allFVals = f(allGridPts);
  fGrid = reshape(allFVals, NV, NU); 

  % Compute the SVD on fGrid
  [Ug, Sg, Vg] = svd(fGrid');
  

end

