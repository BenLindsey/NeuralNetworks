function [ fit ] = nn_fitness_gda( args, trainingInput, trainingOutput, validatingInput, validatingOutput )
    if args(2) > 0
        net = feedforwardnet([args(1), args(2)], 'trainrp');
    else
        net = feedforwardnet([args(1)], 'trainrp');
    end
    
    net = configure(net, trainingInput, trainingOutput);
    net.trainParam.delt_inc = args(3);
    net.trainParam.delt_dec = args(4);
    
    net = train(net, trainingInput, trainingOutput);
    
    matrix = confusionmatrix();
    matrix.update(net, validatingInput, validatingOutput);
    fit = 1 - matrix.getAccuracy();

    disp([num2str(args(1)), ' and ', num2str(args(2)), ' neurons, inc = ', num2str(args(3)), ...
        ', dec = ', num2str(args(4)), ' -> Error rate = ', num2str(fit)]);
end

