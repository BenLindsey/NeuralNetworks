function predictions = testANN( net, x2 )
    % Run the given network with the input data.
    output = sim(net, x2);
    
    % Only output the maximum value in each row.
    predictions = NNout2labels(output);
end
