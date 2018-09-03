function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

         fakeHypothesis = X*theta;
         % In logistic regression we need to find out the sigmoid of every element of the fakehypthesis
         hypothesis = sigmoid(fakeHypothesis);
         
         % Now we will calculate the costFunction
         logHypothesis = arrayfun(@(x) log(x), hypothesis);
         logOnediffHypo = arrayfun(@(x) log(1-x), hypothesis);
         oneDiffY = 1-y;
         
         sqrSum =0;
         
         for i=2:size(theta)
           sqrSum = sqrSum + theta(i)^2;
         endfor
         
         J = -(1/m)*(((logHypothesis')*y)+((logOnediffHypo')*oneDiffY)) + (lambda/(2*m))*sqrSum;
         
         % Now we need to find the gradient;
         proxyTheta = theta;
         proxyTheta(1) = 0;
         grad = (1/m)*(X')*(hypothesis-y) + (lambda/m)*proxyTheta;






% =============================================================

end
