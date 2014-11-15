function [ fit ] = nn_fitness_gdm( args, trainingInput, trainingOutput, validatingInput, validatingOutput )
    if args(4) > 0
        net = feedforwardnet([args(1), args(2)], 'traingdm');
    else
        net = feedforwardnet([args(1)], 'traingdm');
    end
    
    net = configure(net, trainingInput, trainingOutput);
    
    net.trainParam.lr = args(3);
    net.trainParam.mc = args(4);
    
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
        disp(['Neurons:', num2str(args(1)), '|', num2str(args(2)), ...
              ' Lr:', num2str(args(3)), ' Mc:', num2str(args(4)), ...
              ' -> ', num2str(fit)]);
    else
        disp(['Neurons:', num2str(args(1)), ...
              ' Lr:', num2str(args(3)), ' Mc:', num2str(args(4)), ...
              ' -> ', num2str(fit)]);
    end
end

