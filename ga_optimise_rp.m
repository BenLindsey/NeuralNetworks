function [ net ] = ga_optimise_rp(trainingInput, trainingOutput, validatingInput, validatingOutput) 
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer, NeuronsInSecondLayer, delta_inc, delta_dec ]
    nargs = 5;           % Number of arguments
    LB = [1,   0,   1,   1, 0]; % Lower bounds for arguments
    UB = [100, 100, 100, 2, 1]; % Upper bounds for arguments
    %LB = [20, 20, 1.2, 0.3];
    %UB = [30, 30, 1.5, 0.6];
    IC = [1, 2, 3];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    %options = gaoptimset; % Default options
    options = gaoptimset('UseParallel', true, 'Vectorized', 'off', ...
        'TimeLimit', 2, 'PlotFcns', {@gaplotbestf, @gaplotbestindiv}, ...
        'Generations', 10, 'PopulationSize', 50);
    
    % Training data
    [tI, tO] = ANNdata(trainingInput, trainingOutput);  
    
    % Calculate fit for the given optimisation arguments. Lower is better.
    FitnessFunction = @(args) nn_fitness_rp(args, ...
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

