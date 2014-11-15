% Use the same random seed each time the weights are initialised so results
% are reproducible.
rng(1001, 'twister');

confusedMatrix = confusionmatrix();

for i=1:10,
    % Remove 10% of x to form the 9fold data
    foldInput = x;
    foldInput(i:10:end,:) = [];
    
    foldOutput = y;
    foldOutput(i:10:end) = [];
        
    % Use the other 10% of x for test data
    testInput = x(i:10:end,:);
    testOutput = y(i:10:end);
    
    % Split the 9fold data into training and validating
    trainInput = foldInput;
    trainInput(i:10:end,:) = [];
    
    trainOutput = foldOutput;
    trainOutput(i:10:end) = [];
    
    validateInput = foldInput(i:10:end,:);
    validateOutput = foldOutput(i:10:end);    
    
    [tI, tO] = ANNdata(trainInput, trainOutput);
    
    args = ga_optimise_gdm(trainInput, trainOutput, validateInput, validateOutput);
    
    opti(i, :) = args;
    
    if args(4) > 0
        net = feedforwardnet([args(1), args(2)], 'traingdm');
    else
        net = feedforwardnet([args(1)], 'traingdm');
    end
    
    net.trainParam.lr = args(3);
    net.trainParam.mc = args(4);
    net = configure(net, tI , tO);
    
    % Train the network with the train data, stop early using the
    % validation data to prevent overfitting. But do not use the test data
    % yet.
    valInd = i:10:size(tI, 2);
    trainInd = (1:size(tI, 2));
    trainInd(valInd) = [];
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainInd;
    net.divideParam.valInd = valInd;
    net.divideParam.testInd = [];
    
    net = train(net, tI, tO);

    % Update the confusion matrix with the test data for this fold.
    confusedMatrix.update(net, testInput, testOutput);
end

fprintf('Optimum parameters:\n');
disp(opti);

% Get the average statistics from the confusion matrix.
recallRates = zeros(1, 6);
precision   = zeros(1, 6);
f1Scores    = zeros(1, 6);
accuracy    = confusedMatrix.getAccuracy();
for class=1:6,
    recallRates(class) = confusedMatrix.getRecallRate(class);
    precision(class)   = confusedMatrix.getPrecision(class);
    f1Scores(class)    = confusedMatrix.getFScore(class, 1);
end

% Print all statistics.
fprintf('Confusion matrix:\n');
disp(confusedMatrix.Matrix);

fprintf('Average recall rates: ');
disp(recallRates);

fprintf('Average precision:    ');
disp(precision);

fprintf('Average f1 scores:    ');
disp(f1Scores);

fprintf('Average accuracy:     ');
disp(accuracy);
