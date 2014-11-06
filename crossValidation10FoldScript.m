resultB=zeros(floor(length(y)/10),10);
resultD=zeros(floor(length(y)/10),10);

confusedMatrix = confusionmatrix();

for i=1:10,
    
    % Remove the current 10% of the input and output.
    foldInput = x;
    foldInput(i:10:end,:) = [];
    testInput = x(i:10:end,:);
    
    foldOutput = y;
    foldOutput(i:10:end) = [];
    testOutput = y(i:10:end);
        
    [tI, tO] = ANNdata(foldInput, foldOutput);    
    net = feedforwardnet([5]); 
    net = configure(net, tI, tO);
    net.trainParam.epochs = 10;
    train(net, tI, tO);
    
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
