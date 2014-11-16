function [ fit ] = nn_fitness_gda( args, trainingInput, trainingOutput, validatingInput, validatingOutput ) 
    rng(1001, 'twister');
    if args(6) > 0
        net = feedforwardnet([args(1), args(2)], 'traingda');
    else
        net = feedforwardnet([args(1)], 'traingda');
    end
    
    net = configure(net, trainingInput, trainingOutput);
    
    net.trainParam.lr     = args(3);
    net.trainParam.lr_inc = args(4);
    net.trainParam.lr_dec = args(5);
    
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

    if length(args) > 4 && args(6) > 0
    disp(['Neurons:', num2str(args(1)), '|', num2str(args(2)), ...
          ' Lr:', num2str(args(3)), ' Lr_inc:', num2str(args(4)), ...
          ' Lr_dec:', num2str(args(5)),' -> ', num2str(fit)]);
    elseif length(args) > 4
    disp(['Neurons:', num2str(args(1)), ...
          ' Lr:', num2str(args(3)), ' Lr_inc:', num2str(args(4)), ...
          ' Lr_dec:', num2str(args(5)),' -> ', num2str(fit)]);
    else
    disp(['Neurons:', num2str(args(1)), ...
          ' Lr:', num2str(args(2)), ' Lr_inc:', num2str(args(3)), ...
          ' Lr_dec:', num2str(args(4)),' -> ', num2str(fit)]);
    end
end

