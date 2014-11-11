function [ fit ] = nn_fitness_gdm( args, trainingInput, trainingOutput, validatingInput, validatingOutput )
    if args(2) > 0
        net = feedforwardnet([args(1), args(2)], 'traingdm');
    else
        net = feedforwardnet([args(1)], 'traingdm');
    end
    
    net = configure(net, trainingInput, trainingOutput);
    
    net.trainParam.lr = args(3);
    net.trainParam.mc = args(4);
    
    net = train(net, trainingInput, trainingOutput);
    
    matrix = confusionmatrix();
    matrix.update(net, validatingInput, validatingOutput);
    fit = 1 - matrix.getAccuracy();

    disp(['Neurons:', num2str(args(1)), '|', num2str(args(2)), ...
          ' Lr:', num2str(args(3)), ' Mc:', num2str(args(4)), ...
          ' -> ', num2str(fit)]);
end

