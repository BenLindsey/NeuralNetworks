confusedMatrix = confusionmatrix();

innerLoops = 2;

% Bounds on each parameter from the genetic search.
layerOneNeurons = 40:50;
layerTwoNeurons = 10:20;
deltInc = 1.1:0.01:1.2;
deltDec = 0.5:0.02:0.7;

% Create a matrix where each row is one set of parameters.
parameters = zeros(numel(layerOneNeurons) * numel(layerTwoNeurons) ...
    * numel(deltInc) * numel(deltDec), 4);
row = 1;
for a=1:numel(layerOneNeurons),
    for b=1:numel(layerTwoNeurons),
        for c=1:numel(deltInc),
            for d=1:numel(deltDec),
                parameters(row, 1) = layerOneNeurons(a);
                parameters(row, 2) = layerTwoNeurons(b);
                parameters(row, 3) = deltInc(c);
                parameters(row, 4) = deltDec(d);
                row = row + 1;
            end
        end
    end
end

for fold=1:10,
    % Remove 10% of x to form the 9fold data
    foldInput = x;
    foldInput(fold:10:end,:) = [];
    
    foldOutput = y;
    foldOutput(fold:10:end) = [];
        
    % Use the other 10% of x for test data
    testInput = x(fold:10:end,:);
    testOutput = y(fold:10:end);
    
    % Inner loop for parameter optimisation.
    for innerFold=1:innerLoops
        % Split the 9fold data into 'innerLoops' training and 1 validating
        trainInput = foldInput;
        trainInput(innerFold:innerLoops:end,:) = [];

        trainOutput = foldOutput;
        trainOutput(innerFold:innerLoops:end) = [];

        validateInput = foldInput(innerFold:innerLoops:end,:);
        validateOutput = foldOutput(innerFold:innerLoops:end);

        [tI, tO] = ANNdata(trainInput, trainOutput);
        
        for i=1,numel(parameters(:,1)),
            % Build a network with the current set of parameters, then
            % train on the training data.
            parameterSet = parameters(i,:);
            net = feedforwardnet([parameterSet(1), parameterSet(2)], 'trainrp');
            net.trainParam.delt_inc = parameterSet(3);
            net.trainParam.delt_dec = parameterSet(4);
            net = configure(net, tI, tO);
            net = train(net, tI, tO);
            
            % Find the performance of the network using the validation
            % data.
            innerConfusionMatrix = confusionMatrix();
            innerConfusionMatrix.update(net, validateInput, validateOutput);
        end
        
        % TODO BUILD NET FROM BEST ARGS net = optimiseTwoLayers(trainInput, trainOutput, validateInput, validateOutput);
        %net = ga_optimise_rp(trainInput, trainOutput, validateInput, validateOutput);
%         net = feedforwardnet([44, 18], 'trainrp');
%         net.trainParam.delt_inc = 1.1479;
%         net.trainParam.delt_dec = 0.6660;
%         net = configure(net, tI, tO);
%         [net, tr] = train(net, tI, tO);
    end
    
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
