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

net1 = ga_optimise_gdm_multi(trainInput, trainOutput, validateInput, validateOutput, 1);
net2 = ga_optimise_gdm_multi(trainInput, trainOutput, validateInput, validateOutput, 2);
net3 = ga_optimise_gdm_multi(trainInput, trainOutput, validateInput, validateOutput, 3);
net4 = ga_optimise_gdm_multi(trainInput, trainOutput, validateInput, validateOutput, 4);
net5 = ga_optimise_gdm_multi(trainInput, trainOutput, validateInput, validateOutput, 5);
net6 = ga_optimise_gdm_multi(trainInput, trainOutput, validateInput, validateOutput, 6);

disp(net1);
disp(net2);
disp(net3);
disp(net4);
disp(net5);
disp(net6);


