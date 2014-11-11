function [ fit ] = nn_fitness( args, trainingInput, trainingOutput, validatingInput, validatingOutput )
    if length(args) > 1 && args(2) > 0
        net = feedforwardnet([args(1), args(2)], 'trainrp');
    else
        net = feedforwardnet([args(1)], 'trainrp');
    end
    
    net = configure(net, trainingInput, trainingOutput);
    net = train(net, trainingInput, trainingOutput);
    
    matrix = confusionmatrix();
    matrix.update(net, validatingInput, validatingOutput);
    fit = 1 - matrix.getAccuracy();
    
    
    if length(args) > 1
        disp([num2str(args(1)), ' ', num2str(args(2)), ' -> ', num2str(fit)]);
    else
        disp([num2str(args(1)), ' -> ', num2str(fit)]);
    end
end

