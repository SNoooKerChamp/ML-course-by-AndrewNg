function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Now we have the z vector which is basically the fake hypothesis.
% To run the sigmoid function on each and every element of the vector we will do the following

 g = arrayfun(@(x) 1/(1+e^(-x)), z);
 
 






% =============================================================

end
