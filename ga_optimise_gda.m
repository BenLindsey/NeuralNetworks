function [ net ] = ga_optimise_gda(trainingInput, trainingOutput, validatingInput, validatingOutput) 
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer, NeuronsInSecondLayer, LearningRate, LearningInc, LearningDec ]
    nargs = 6;    % Number of arguments
    LB = [1 1 0 1 0 0];   % Lower bounds for arguments
    UB = [50 50 1 2 1 1]; % Upper bounds for arguments
    IC = [1, 2, 6];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    %options = gaoptimset; % Default options
    options = gaoptimset('UseParallel', true, ...
                         'Vectorized', 'off', 'PopulationSize', 50, ... 
                         'Generations', 50, ...
                         'PlotFcns', @gaplotbestf); % Timelimit(seconds) constraint
                                          % ONLY checked after 1st gen.
                                         
    % Training data
    [tI, tO] = ANNdata(trainingInput, trainingOutput);  
    
    % Calculate fit for the given optimisation arguments. Lower is better.
    FitnessFunction = @(args) nn_fitness_gda(args, ...
        tI, tO, validatingInput, validatingOutput);
    
    % Result of genetic algorithm search, where x is discovered values
    % for OPTIMISATION ARGUMENTS
    [x, fval] = ga(FitnessFunction, nargs, [], [], [], [], ...
                    LB, UB, [], IC, options)
                
    disp('Using values:');
    disp(x);
end

