function [ args ] = ga_optimise_gda(trainingInput, trainingOutput, validatingInput, validatingOutput) 
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer, NeuronsInSecondLayer, LearningRate, LearningInc, LearningDec ]
    nargs = 6;    % Number of arguments
    LB = [6 6 0 1 0 0];   % Lower bounds for arguments
    UB = [45 45 1 2 1 1]; % Upper bounds for arguments
    IC = [1, 2, 6];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    %options = gaoptimset; % Default options
    options = gaoptimset('UseParallel', true, ...
                         'Vectorized', 'off', ...
                         'PopulationSize', 100, ...
                         'Generations', 30, ...
                         'StallGenLimit', 5, ...
                         'PlotFcns', {@gaplotbestf, @gaplotbestindiv});
    
    % Training data
    [tI, tO] = ANNdata(trainingInput, trainingOutput);
    [vI, vO] = ANNdata(validatingInput, validatingOutput);
                                         
    % Calculate fit for the given optimisation arguments. Lower is better.
    FitnessFunction = @(args) nn_fitness_gda(args, tI, tO, vI, vO);
    
    % Result of genetic algorithm search, where x is discovered values
    % for OPTIMISATION ARGUMENTS
    [args, fval] = ga(FitnessFunction, nargs, [], [], [], [], ...
                    LB, UB, [], IC, options)
                
    disp('Using values:');
    disp(args);
end

