function [ net ] = optimise(trainingInput, trainingOutput, validatingInput, validatingOutput) 
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer ]
    nargs = 1; % Number of arguments
    LB = [1];  % Lower bounds for arguments
    UB = [10]; % Upper bounds for arguments
    IC = [1];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    options = gaoptimset; % Default options
    % options = gaoptimset('TimeLimit', 2); % Timelimit(seconds) constraint
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
    
    net = feedforwardnet([x(1)]);
    net = configure(net, tI, tO);
    net = train(net, tI, tO);
end

