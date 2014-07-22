% v2hall: to transform from visible (input) variables to all hidden (output) variables
%
% Hall = h2val(dnn, V)
%
%
%Output parameters:
% Hall: all hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes
%
%
%Input parameters:
% dnn: the Deep Neural Network model (dbn, rbm)
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes
%
%
%Version: 20130727

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:                                     %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hall = v2hall(dnn, V)

if( isequal(dnn.type(3:5), 'RBM') )
    Hall = cell(1,1);
    Hall{1} = sigmoid( bsxfun(@plus, V * dnn.W, dnn.b ) );

elseif( isequal(dnn.type(3:5), 'DBN') )
    nrbm = numel( dnn.rbm );
    Hall = cell(nrbm,1);
    H0 = V;
    for i=1:nrbm
        H1 = v2h( dnn.rbm{i}, H0 );
        H0 = H1;
        Hall{i} = H1;
    end
end

