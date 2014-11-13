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
    trainInput = foldInput;
    trainInput(i:10:end,:) = [];
    
    trainOutput = foldOutput;
    trainOutput(i:10:end) = [];
    
    validateInput = foldInput(i:10:end,:);
    validateOutput = foldOutput(i:10:end);    
    
    [tI, tO] = ANNdata(trainInput, trainOutput);
    
    args = ga_optimise_gda(trainInput, trainOutput, validateInput, validateOutput);
    
    opti(i, :) = args;
    
    if args(6) > 0
        net = feedforwardnet([args(1), args(2)], 'traingda');
    else
        net = feedforwardnet([args(1)], 'traingda');
    end
    net.trainParam.lr     = args(3);
    net.trainParam.lr_inc = args(4);
    net.trainParam.lr_dec = args(5);
    net = configure(net, tI , tO);    
    net = train(net, tI, tO);

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
