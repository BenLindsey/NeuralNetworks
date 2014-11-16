function [ args ] = ga_optimise_gdm(trainingInput, trainingOutput, validatingInput, validatingOutput) 
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer, NeuronsInSecondLayer, LearningRate, Momentum ]
    nargs = 5;    % Number of arguments
    LB = [6 6 0 0 0];   % Lower bounds for arguments
    UB = [45 45 1 1 1]; % Upper bounds for arguments
    IC = [1, 2, 5];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    %options = gaoptimset; % Default options
    options = gaoptimset('UseParallel', true, ...
                         'Vectorized', 'off', ...
                         'PopulationSize', 100, ...
                         'Generations', 15, ...
                         'StallGenLimit', 5, ...
                         'PlotFcns', {@gaplotbestf, @gaplotbestindiv}); % Timelimit(seconds) constraint
                                          % ONLY checked after 1st gen.
                                         
    % Training data
    [tI, tO] = ANNdata(trainingInput, trainingOutput);
    [vI, vO] = ANNdata(validatingInput, validatingOutput);
    
    % Calculate fit for the given optimisation arguments. Lower is better.
    FitnessFunction = @(args) nn_fitness_gdm(args, tI, tO, vI, vO);
    
    % Result of genetic algorithm search, where x is discovered values
    % for OPTIMISATION ARGUMENTS
    [args, fval] = ga(FitnessFunction, nargs, [], [], [], [], ...
                    LB, UB, [], IC, options)
                
    disp('Using values:');
    disp(args);
end

