function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).

      covarianceMatrix = zeros(n,n);
      
      for i = 1:m
        x = X(i,:);
        covarianceMatrix = covarianceMatrix + x'*x;
        
        
      endfor
      
      covarianceMatrix = covarianceMatrix/m;
      
      % find the eigen vector
      
      [U,S,V] = svd(covarianceMatrix);
      
      
        
      
        
       
        








% =========================================================================

end
