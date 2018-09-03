function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), 
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), 
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

 greekDelta1 = zeros(size(Theta1));
 greekDelta2 = zeros(size(Theta2));
 
 

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


   % add ones to X
   X = [ones(m,1) X];
   
   for i=1:m
     number = y(i);
     y_proxy = zeros(num_labels,1);
     for j=1:num_labels
       if number==j
         y_proxy(j) = 1;
       else
         y_proxy(j) = 0;
       endif
       
     endfor
     x_t = X(i,:);
     % Need to perform forward Propagation
     y1 = Theta1*(x_t');
     a1 = arrayfun(@(x) (1/(1+(e^(-x)))), y1);
     % add 1 to a1
     a1 = [1;a1];
     y2 = Theta2*a1;
     hypo = arrayfun(@(x) (1/(1+(e^(-x)))), y2);
     
     diffY_proxy = 1-y_proxy;
     logHypo = arrayfun(@(x) log(x), hypo);
     oneDifflogHypo =  arrayfun(@(x) log(1-x), hypo);
     
     % Now we have to calculate the cost
     J = J + (((logHypo')*y_proxy) + ((oneDifflogHypo')*diffY_proxy));
     
     delta3 = hypo - y_proxy; % this is for the outer most layer;
     delta2 = (Theta2')*delta3.*a1.*(1-a1);
     % delta of only hidden and output layer is formulated
     % remove the first row of delta 2
     delta2 = delta2(2:end);
     greekDelta1 = greekDelta1 + delta2*x_t;
     % x_t is in row form and not in collumn form and that is why its transpose is not taken.
     
     greekDelta2 = greekDelta2 + delta3*(a1');
     
   endfor
   
   J = (-1/m)*J;
   % We need to incorporate regularization as  well
   % remove the collumns 1 from theta1 and theta2
   Theta1_proxy = Theta1;
    Theta1_proxy(:,[1]) = []; % This basically removes the first row . We dont want to take into account the bias term.
    Theta2_proxy = Theta2;
    Theta2_proxy(:,[1]) = [];
    
    J = J + (lambda/(2*m))*(sum(sum(Theta1_proxy.^2)) + sum(sum(Theta2_proxy.^2)));
    
     Theta1_grad = (1/m)*greekDelta1;
     Theta2_grad = (1/m)*greekDelta2;
     
     % we need to extract the 1st collumn from both the delta and the theta matrices
     grad1Col1 = Theta1_grad(:,1);
     grad2Col2 = Theta2_grad(:,1);
     
     Theta1_grad(:,[1]) = []; % it basically removes the 1st collumn
     Theta2_grad(:,[1]) = [];
     
     Theta1Col1 = Theta1(:,1);
     Theta2Col2 = Theta2(:,1);
     
      Theta1(:,[1]) = [];
     Theta2(:,[1]) = [];
     
     Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
     Theta2_grad = Theta2_grad + (lambda/m)*Theta2;
     
     Theta1_grad = [grad1Col1 Theta1_grad];
     Theta2_grad = [grad2Col2 Theta2_grad];
     
     Theta1 = [Theta1Col1 Theta1];
     Theta2 = [Theta2Col2 Theta2];
     
      
      
     
    
     



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
