% v2h: to transform from visible (input) variables to hidden (output) variables
%
% H = h2v(dnn, V)
%
%
%Output parameters:
% H: hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes
%
%
%Input parameters:
% dnn: the Deep Neural Network model (dbn, rbm)
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes
%
%
%Example:
% datanum = 1024;
% outputnum = 16;
% inputnum = 4;
%
% inputdata = rand(datanum, outputnum);
%
% dnn = randRBM( inputnum, outputnum );
% outputdata = v2h( dnn, input );
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
function H = v2h(dnn, V)

if( isequal(dnn.type, 'BBRBM') )
    H = sigmoid( bsxfun(@plus, V * dnn.W, dnn.b ) );

elseif( isequal(dnn.type, 'GBRBM') )
    v = bsxfun(@rdivide, V, dnn.sig );
    H = sigmoid( bsxfun(@plus, v * dnn.W, dnn.b ) );   

elseif( isequal(dnn.type(3:5), 'DBN') )
    nrbm = numel( dnn.rbm );
    H0 = V;
    for i=1:nrbm
        H1 = v2h( dnn.rbm{i}, H0 );
        H0 = H1;
    end
    H = H1;
    
end
