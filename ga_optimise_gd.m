function args = ga_optimise_gd(trainingInput, trainingOutput, validatingInput, validatingOutput)
    % OPTIMISATION ARGUMENTS:
    % [ NeuronsInFirstLayer, NeuronsInSecondLayer, delta_inc, delta_dec, UseSecondLayer ]
    nargs = 4;           % Number of arguments
    LB = [6, 6, 0, 0]; % Lower bounds for arguments
    UB = [45, 45, 1, 1]; % Upper bounds for arguments
    IC = [1, 2, 4];  % Integer constraints (by index of OPTIMISATION ARGUMENTS)
    %options = gaoptimset; % Default options
    options = gaoptimset('UseParallel', true, ...
                         'Vectorized', 'off', 'PopulationSize', 100, ...
                         'StallGenLimit', 5, 'Generations',30,...
                         'PlotFcns', {@gaplotbestf, @gaplotbestindiv});
    
    % Training data
    [tI, tO] = ANNdata(trainingInput, trainingOutput);
    [vI, vO] = ANNdata(validatingInput, validatingOutput);
    
    % Calculate fit for the given optimisation arguments. Lower is better.
    FitnessFunction = @(args) nn_fitness_gd(args, tI, tO, vI, vO);
    
    % Result of genetic algorithm search, where x is discovered values
    % for OPTIMISATION ARGUMENTS
    [args, fval] = ga(FitnessFunction, nargs, [], [], [], [], ...
                    LB, UB, [], IC, options);
    disp('Found optimum parameters:');
    disp(args);
end
