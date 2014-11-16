function predictions = testANN( net, x2 )
    % Run the given network with the input data.
    [input, output] = ANNdata(x2, []);
    
    output = sim(net, input);
    
    % Only output the maximum value in each row.
    predictions = NNout2labels(output);
end
