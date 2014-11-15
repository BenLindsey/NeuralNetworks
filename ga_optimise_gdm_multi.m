function [ net ] = ga_optimise_gdm_multi(trainingInput, trainingOutput, validatingInput, validatingOutput, emotion) 
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer, NeuronsInSecondLayer, LearningRate, Momentum ]
    nargs = 4;    % Number of arguments
    LB = [1 1 0 0];   % Lower bounds for arguments
    UB = [100 100 1 1]; % Upper bounds for arguments
    IC = [1, 2];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    %options = gaoptimset; % Default options
    options = gaoptimset('TimeLimit', 2, 'UseParallel', true, 'Vectorized', 'off', ...
                         'PopulationSize', 50, ... 
                         'Generations', 10, ...
                         'PlotFcns', @gaplotbestf); % Timelimit(seconds) constraint
                                          % ONLY checked after 1st gen.
                                         
    % Training data
    [tI, tO] = ANNdata(trainingInput, trainingOutput);
    
    for i = 1:6
        if (i ~= emotion)
            tO(i, :) = zeros(1, size(tO, 2));
        end
    end
    
    % Calculate fit for the given optimisation arguments. Lower is better.
    FitnessFunction = @(args) nn_fitness_gdm(args, ...
        tI, tO, validatingInput, validatingOutput);
    
    % Result of genetic algorithm search, where x is discovered values
    % for OPTIMISATION ARGUMENTS
    [x, fval] = ga(FitnessFunction, nargs, [], [], [], [], ...
                    LB, UB, [], IC, options)
                
    disp('Using values:');
    disp(x);
    net = x;
end

