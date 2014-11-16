resultB=zeros(floor(length(y)/10),10);
resultD=zeros(floor(length(y)/10),10);

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
    trainingInput = foldInput;
    trainingInput(i:10:end,:) = [];
    
    trainingOutput = foldOutput;
    trainingOutput(i:10:end) = [];
    
    % NOTE: Validatation data should probably only depend on the fold data,
    % not the fold number (i) as well. But this shouldn't make a
    % difference.
    validatingInput = foldInput(i:10:end,:);
    validatingOutput = foldOutput(i:10:end);

    [tI, tO] = ANNdata(trainingInput, trainingOutput);
    [vI, vO] = ANNdata(validatingInput, validatingOutput);
    
    args = ga_optimise_gda(tI, tO, vI, vO);
    
    opti(i, :) = args;
    
    if args(6) > 0
        net = feedforwardnet([args(1), args(2)], 'traingda');
    else
        net = feedforwardnet([args(1)], 'traingda');
    end

    totalInput = [tI, vI];
    totalOutput = [tO, vO];
    
    net = configure(net, totalInput, totalOutput);
    
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:size(tI, 2);
    net.divideParam.valInd = (size(tI, 2) + 1):size(totalInput, 2);
    net.divideParam.testInd = [];
    
    net.trainParam.lr     = args(3);
    net.trainParam.lr_inc = args(4);
    net.trainParam.lr_dec = args(5);
    
    % Suppress output and train.
    net.trainParam.show = NaN;
    net = train(net, totalInput, totalOutput);
    
    % Update the confusion matrix with the test data for this fold.
    confusedMatrix.update(net, testInput, testOutput);
end

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
