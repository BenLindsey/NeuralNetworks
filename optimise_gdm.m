i = 1;
foldInput = x;
foldInput(i:10:end,:) = [];

foldOutput = y;
foldOutput(i:10:end) = [];

% Use the other 10% of x for test data
testInput = x(i:10:end,:);  
testOutput = y(i:10:end); 

ga_optimise_gdm(foldInput, foldOutput, testInput, testOutput);
