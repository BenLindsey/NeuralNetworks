function fit = nn_fitness_gdE( args, trainingInput, trainingOutput, validatingInput, validatingOutput )
    rng(1001, 'twister');
    if args(4) > 0
        net = feedforwardnet([args(1), args(2)], 'traingd');
    else
        net = feedforwardnet([args(1)], 'traingd');
    end
    
    net = configure(net, trainingInput, trainingOutput);
    
    net.trainParam.lr = args(3);
    net.trainParam.epochs=3000;
    
    % Train with the training set, use the validation set to avoid
    % overfitting and do not use a test set.
    totalInput = [trainingInput, validatingInput];
    totalOutput = [trainingOutput, validatingOutput];
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:size(trainingInput, 2);
    net.divideParam.valInd = (size(trainingInput, 2) + 1):size(totalInput, 2);
    net.divideParam.testInd = [];
    
    % Suppress output and train.
    net.trainParam.show = NaN;
    net = train(net, totalInput, totalOutput);
    
    matrix = confusionmatrix();
    matrix.updateWithoutConvert(net, validatingInput, validatingOutput);
    fit = 1 - matrix.getAccuracy();

    if args(4) > 0
        disp([num2str(args(1)), ' and ', num2str(args(2)), ' neurons, lr = ', num2str(args(3)), ...
             ' -> Error rate = ', num2str(fit)]);
    else
        disp([num2str(args(1)), ' neurons, lr = ', num2str(args(3)), ...
             ' -> Error rate = ', num2str(fit)]);
    end
end

