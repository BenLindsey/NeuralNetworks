function [ net ] = ga_optimise_da(trainingInput, trainingOutput, validatingInput, validatingOutput) 
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer, NeuronsInSecondLayer ]
    nargs = 2;    % Number of arguments
    LB = [1 0];   % Lower bounds for arguments
    UB = [10 10]; % Upper bounds for arguments
    IC = [1, 2];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    %options = gaoptimset; % Default options
    options = gaoptimset('UseParallel', true, 'Vectorized', 'off'); % Timelimit(seconds) constraint
                                          % ONLY checked after 1st gen.
                                         
    % Training data
    [tI, tO] = ANNdata(trainingInput, trainingOutput);  
    
    % Calculate fit for the given optimisation arguments. Lower is better.
    FitnessFunction = @(args) nn_fitness(args, ...
        tI, tO, validatingInput, validatingOutput);
    
    % Result of genetic algorithm search, where x is discovered values
    % for OPTIMISATION ARGUMENTS
    [x, fval] = ga(FitnessFunction, nargs, [], [], [], [], ...
                    LB, UB, [], IC, options);
    disp('Using values:');
    disp(x);
    net = feedforwardnet([x(1), x(2)]);
    net = configure(net, tI, tO);
    net = train(net, tI, tO);
end

