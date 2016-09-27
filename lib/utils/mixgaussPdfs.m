function pdfs = mixgaussPdfs(mu, Sigma, mixmat)
  m  = size(mu, 2);

  pdfs = cell(m, 1);
  for j = 1:m
    pdfs{j} = @(x) mixgaussFun(x, mu(:,j,:), Sigma(:,:,j,:), mixmat(j,:));
  end
end

function pdf = mixgaussFun(x, mu, Sigma, mixmat)
  K = size(mixmat, 2);

  pdf = 0;
  for k = 1:K
    pdf = pdf + mixmat(k) .* gaussian_prob(x, mu(:,1,k), Sigma(:,:,1,k));
  end
end
