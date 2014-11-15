function [ args ] = ga_optimise_gdm(trainingInput, trainingOutput, validatingInput, validatingOutput) 
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer, NeuronsInSecondLayer, LearningRate, Momentum ]
    nargs = 4;    % Number of arguments
    LB = [1 1 0 0];   % Lower bounds for arguments
    UB = [100 100 1 1]; % Upper bounds for arguments
    IC = [1, 2];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    %options = gaoptimset; % Default options
    options = gaoptimset('UseParallel', true, ...
                         'Vectorized', 'off', ...
                         'PopulationSize', 50, ... 
                         'Generations', 15, ...
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

