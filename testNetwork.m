% 35 neurons, inc = 1.4537, dec = 0.31498 -> Error rate = 0.14286
trainingInput = x;
trainingInput(1:10:end, :) = [];
trainingOutput = y;
trainingOutput(1:10:end, :) = [];
[tI, tO] = ANNdata(trainingInput, trainingOutput);

validatingInput = x(1:10:end, :);
validatingOutput = y(1:10:end, :);
[vI, vO] = ANNdata(validatingInput, validatingOutput);

net = feedforwardnet([35], 'trainrp');
net = configure(net, tI, tO);
net.trainParam.delt_inc = 1.4537;
net.trainParam.delt_dec = 0.31498;

% Train with the training set, use the validation set to avoid
% overfitting and do not use a test set.
totalInput = [tI, vI];
totalOutput = [tO, vO];
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:size(tI, 2);
net.divideParam.valInd = (size(tI, 2) + 1):size(totalInput, 2);
net.divideParam.testInd = [];

net = train(net, totalInput, totalOutput);

matrix = confusionmatrix();
matrix.updateWithoutConvert(net, vI, vO);
matrix.getAccuracy()