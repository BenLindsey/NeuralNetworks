i = 1;
foldInput = x;
foldInput(i:10:end,:) = [];

foldOutput = y;
foldOutput(i:10:end) = [];

% Use the other 10% of x for test data
testInput = x(i:10:end,:);
testOutput = y(i:10:end);

% Split the 9fold data into training and validating
trainInput = foldInput;
trainInput(i:10:end,:) = [];

trainOutput = foldOutput;
trainOutput(i:10:end) = [];

validateInput = foldInput(i:10:end,:);
validateOutput = foldOutput(i:10:end);    

ga_optimise_gdm(trainInput, trainOutput, validateInput, validateOutput);