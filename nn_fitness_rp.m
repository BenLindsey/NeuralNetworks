function fit = nn_fitness_rp( args, trainingInput, trainingOutput, validatingInput, validatingOutput )
    rng(1001, 'twister');
    if args(5) > 0
        net = feedforwardnet([args(1), args(2)], 'trainrp');
    else
        net = feedforwardnet([args(1)], 'trainrp');
    end
    
    net = configure(net, trainingInput, trainingOutput);
    
    net.trainParam.delt_inc = args(3);
    net.trainParam.delt_dec = args(4);
    
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

    if args(5) > 0
        disp([num2str(args(1)), ' and ', num2str(args(2)), ' neurons, inc = ', num2str(args(3)), ...
            ', dec = ', num2str(args(4)), ' -> Error rate = ', num2str(fit)]);
    else
        disp([num2str(args(1)), ' neurons, inc = ', num2str(args(3)), ...
            ', dec = ', num2str(args(4)), ' -> Error rate = ', num2str(fit)]);
    end
end

