% 25 neurons, inc = 1.1964, dec = 0.23687 -> Error rate = 0.10989
trainingInput = x;
trainingInput(1:10:end, :) = [];
trainingOutput = y;
trainingOutput(1:10:end, :) = [];

testInput = x(1:10:end, :);
testOutput = y(1:10:end, :);
[tI, tO] = ANNdata(trainingInput, trainingOutput);
[tstI, tstO] = ANNdata(testInput, testOutput);

net = feedforwardnet([25], 'trainrp');
net = configure(net, tI, tO);
net.trainParam.delt_inc = 1.1964;
net.trainParam.delt_dec = 0.23687;

% Train with the training and validation set, use the test set to avoid
% overfitting.
totalInput = [tI, tstI];
totalOutput = [tO, tstO];
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:size(tI, 2);
net.divideParam.valInd = (size(tI, 2) + 1):size(totalInput, 2);
net.divideParam.testInd = [];

net = train(net, totalInput, totalOutput);

matrix = confusionmatrix();
matrix.updateWithoutConvert(net, tstI, tstO);
disp(matrix.Matrix);
disp(matrix.getAccuracy);