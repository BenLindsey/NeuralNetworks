function [ fit ] = nn_fitness_gda( args, trainingInput, trainingOutput, validatingInput, validatingOutput )
    if args(2) > 0
        net = feedforwardnet([args(1), args(2)], 'traingda');
    else
        net = feedforwardnet([args(1)], 'traingda');
    end
    
    net = configure(net, trainingInput, trainingOutput);
    
    net.trainParam.lr     = args(3);
    net.trainParam.lr_inc = args(4);
    net.trainParam.lr_dec = args(5);
    
    net = train(net, trainingInput, trainingOutput);
    
    matrix = confusionmatrix();
    matrix.update(net, validatingInput, validatingOutput);
    fit = 1 - matrix.getAccuracy();

    disp(['Neurons:', num2str(args(1)), '|', num2str(args(2)), ...
          ' Lr:', num2str(args(3)), ' Lr_inc:', num2str(args(4)), ...
          ' Lr_dec:', num2str(args(5)),' -> ', num2str(fit)]);
end

