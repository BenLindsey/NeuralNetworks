function [ fit ] = nn_fitness( args, trainingInput, trainingOutput, validatingInput, validatingOutput )
    if args(2) > 0
        net = feedforwardnet([args(1), args(2)]);
    else
        net = feedforwardnet([args(1)]);
    end
    
    net = configure(net, trainingInput, trainingOutput);
    net = train(net, trainingInput, trainingOutput);
    
    matrix = confusionmatrix();
    matrix.update(net, validatingInput, validatingOutput);
    fit = 1 - matrix.getAccuracy();
    
    disp([num2str(args(1)), ' ', num2str(args(2)), ' -> ', num2str(fit)]);
end

