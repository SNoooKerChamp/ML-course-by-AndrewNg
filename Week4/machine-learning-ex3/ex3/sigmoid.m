function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = arrayfun(@(x) 1/(1+e^(-x)), z);
end
