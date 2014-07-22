% trainDBN: training the Deep Belief Nets (DBN) model by back projection algorithm
%
% [dbn rmse] = trainDBN( dbn, IN, OUT, opts)
%
%
%Output parameters:
% dbn: the trained Deep Belief Nets (DBN) model
% rmse: the rmse between the teaching data and the estimates
%
%
%Input parameters:
% dbn: the initial Deep Belief Nets (DBN) model
% IN: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes
% OUT: teaching hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes
% opts (optional): options
%
% options (defualt value):
%  opts.LayerNum: # of tarining RBMs counted from output layer (all layer)
%  opts.MaxIter: Maxium iteration number (100)
%  opts.InitialMomentum: Initial momentum until InitialMomentumIter (0.5)
%  opts.InitialMomentumIter: Iteration number for initial momentum (5)
%  opts.FinalMomentum: Final momentum after InitialMomentumIter (0.9)
%  opts.WeightCost: Weight cost (0.0002)
%  opts.DropOutRate: List of Dropout rates for each layer (0)
%  opts.StepRatio: Learning step size (0.01)
%  opts.BatchSize: # of mini-batch data (# of all data)
%  opts.Object: specify the object function ('Square')
%              'Square' 
%              'CrossEntorpy'
%  opts.Verbose: verbose or not (false)
%
%
%Example:
% datanum = 1024;
% outputnum = 16;
% hiddennum = 8;
% inputnum = 4;
% 
% inputdata = rand(datanum, inputnum);
% outputdata = rand(datanum, outputnum);
% 
% dbn = randDBN([inputnum, hiddennum, outputnum]);
% dbn = pretrainDBN( dbn, inputdata );
% dbn = SetLinearMapping( dbn, inputdata, outputdata );
% dbn = trainDBN( dbn, inputdata, outputdata );
% 
% estimate = v2h( dbn, inputdata );
%
%
%Reference:
%for details of the dropout
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.
%
%
%Version: 20131204

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:                                     %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dbn rmse] = trainDBN( dbn, IN, OUT, opts)

% Important parameters
InitialMomentum = 0.5;     % momentum for first five iterations
FinalMomentum = 0.9;       % momentum for remaining iterations
WeightCost = 0.0002;       % costs of weight update
InitialMomentumIter = 5;

MaxIter = 100;
StepRatio = 0.01;
BatchSize = 0;
Verbose = false;

Layer = 0;
strbm = 1;

nrbm = numel( dbn.rbm );
DropOutRate = zeros(nrbm,1);

OBJECTSQUARE = 1;
OBJECTCROSSENTROPY = 2;
Object = OBJECTSQUARE;

if( exist('opts' ) )
 if( isfield(opts,'MaxIter') )
  MaxIter = opts.MaxIter;
 end
 if( isfield(opts,'InitialMomentum') )
  InitialMomentum = opts.InitialMomentum;
 end
 if( isfield(opts,'InitialMomentumIter') )
  InitialMomentumIter = opts.InitialMomentumIter;
 end
 if( isfield(opts,'FinalMomentum') )
  FinalMomentum = opts.FinalMomentum;
 end
 if( isfield(opts,'WeightCost') )
  WeightCost = opts.WeightCost;
 end
 if( isfield(opts,'DropOutRate') )
  DropOutRate = opts.DropOutRate;
  if( numel(DropOutRate) == 1 )
      DropOutRate = ones(nrbm,1) * DropOutRate;
  end
 end
 if( isfield(opts,'StepRatio') )
  StepRatio = opts.StepRatio;
 end
 if( isfield(opts,'BatchSize') )
  BatchSize = opts.BatchSize;
 end
 if( isfield(opts,'Verbose') )
  Verbose = opts.Verbose;
 end
 if( isfield(opts,'Layer') )
  Layer = opts.Layer;
 end
 if( isfield(opts,'Object') )
  if( strcmpi( opts.object, 'Square' ) )
   Object = OBJECTSQUARE;
  elseif( strcmpi( opts.object, 'CrossEntropy' ) )
   Object = OBJECTCROSSENTROPY;
  end
 end
end

num = size(IN,1);
if( BatchSize <= 0 )
  BatchSize = num;
end

if( Layer > 0 )
    strbm = nrbm - Layer + 1;
end

deltaDbn = dbn;
for n=strbm:nrbm
    deltaDbn.rbm{n}.W = zeros(size(dbn.rbm{n}.W));
    deltaDbn.rbm{n}.b = zeros(size(dbn.rbm{n}.b));
end

if( Layer > 0 )
    strbm = nrbm - Layer + 1;
end

if( sum(DropOutRate > 0) )
    OnInd = GetOnInd( dbn, DropOutRate, strbm );
    for n=max([2,strbm]):nrbm
        dbn.rbm{n}.W = dbn.rbm{n}.W / numel(OnInd{n-1}) * size(dbn.rbm{n-1}.W,2);
    end
end

for iter=1:MaxIter
    
    % Set momentum
	if( iter <= InitialMomentumIter )
		momentum = InitialMomentum;
	else
		momentum = FinalMomentum;
    end
    
	ind = randperm(num);
	for batch=1:BatchSize:num		
		bind = ind(batch:min([batch + BatchSize - 1, num]));
        
        trainDBN = dbn;
        if( DropOutRate > 0 )
            [trainDBN OnInd] = GetDroppedDBN( trainDBN, DropOutRate, strbm );
            Hall = v2hall( trainDBN, IN(bind,OnInd{1}) );
        else
             Hall = v2hall( trainDBN, IN(bind,:) );
        end
      
        for n=nrbm:-1:strbm
            derSgm = Hall{n} .* ( 1 - Hall{n} );
            if( n+1 > nrbm )
                der = ( Hall{nrbm} - OUT(bind,:) );
                if( Object == OBJECTSQUARE )
                    der = derSgm .* der;
                end
            else
                der = derSgm .* ( der * trainDBN.rbm{n+1}.W' );
            end

            if( n-1 > 0 )
                in = Hall{n-1};
            else
                if( DropOutRate > 0 )
                    in = IN(bind,OnInd{1});
                else
                    in = IN(bind,:);
                end
            end

            in = cat(2, ones(numel(bind),1), in);

            deltaWb = in' * der / numel(bind);
            deltab = deltaWb(1,:);
            deltaW = deltaWb(2:end,:);

            if( strcmpi( dbn.rbm{n}.type, 'GBRBM' ) )
				deltaW = bsxfun( @rdivide, deltaW, trainDBN.rbm{n}.sig' );
            end

            deltaDbn.rbm{n}.W = momentum * deltaDbn.rbm{n}.W;
            deltaDbn.rbm{n}.b = momentum * deltaDbn.rbm{n}.b;

            if( DropOutRate > 0 )
                if( n == nrbm )
                    deltaDbn.rbm{n}.W(OnInd{n},:) = deltaDbn.rbm{n}.W(OnInd{n},:) - StepRatio * deltaW;
                    deltaDbn.rbm{n}.b = deltaDbn.rbm{n}.b - StepRatio * deltab;
                else
                    deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1}) = deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1}) - StepRatio * deltaW;
                    deltaDbn.rbm{n}.b(1,OnInd{n+1}) = deltaDbn.rbm{n}.b(1,OnInd{n+1}) - StepRatio * deltab;
                end
            else
                deltaDbn.rbm{n}.W = deltaDbn.rbm{n}.W - StepRatio * deltaW;
                deltaDbn.rbm{n}.b = deltaDbn.rbm{n}.b - StepRatio * deltab;
            end
        end
            
        for n=strbm:nrbm            
            dbn.rbm{n}.W = dbn.rbm{n}.W + deltaDbn.rbm{n}.W;
            dbn.rbm{n}.b = dbn.rbm{n}.b + deltaDbn.rbm{n}.b;  
        end
    end
    
    if( Verbose )
        tdbn = dbn;
        if( sum(DropOutRate > 0) )
            OnInd = GetOnInd( tdbn, DropOutRate, strbm );
            for n=max([2,strbm]):nrbm
                tdbn.rbm{n}.W = tdbn.rbm{n}.W * numel(OnInd{n-1}) / size(tdbn.rbm{n-1}.W,2);
            end
        end
        out = v2h( tdbn, IN );
        err = power( OUT - out, 2 );
        rmse = sqrt( sum(err(:)) / numel(err) );
        fprintf( '%3d : %9.4f\n', iter, rmse );
    end
end

if( sum(DropOutRate > 0) )
    OnInd = GetOnInd( dbn, DropOutRate, strbm );
    for n=max([2,strbm]):nrbm
        dbn.rbm{n}.W = dbn.rbm{n}.W * numel(OnInd{n-1}) / size(dbn.rbm{n-1}.W,2);
    end
end
