function [ net ] = buildFinalNN( gd_optiset, gda_optiset, gdm_optiset, rp_optiset, x, y)
    [gdm_accuracy, gdm_params ] = findBestForTrainingFunction(gdm_optiset, 'traingdm', 5, @assign_gdm, x, y)
    [rp_accuracy, rp_params ] = findBestForTrainingFunction(rp_optiset, 'trainrp', 5, @assign_rp, x, y)
    [gd_accuracy, gd_params ] = findBestForTrainingFunction(gd_optiset, 'traingd', 4, @assign_gd, x, y)
    [gda_accuracy, gda_params ] = findBestForTrainingFunction(gda_optiset, 'traingda', 6, @assign_gda, x, y)
    
    scores = [gda_accuracy, gdm_accuracy, rp_accuracy, gd_accuracy];
    best = max(scores(:));
    
    if best == gda_accuracy
    	net = buildNet(1, x, y, gda_params, 6, 'traingda', @assign_gda);
    elseif best == gdm_accuracy
        net = buildNet(1, x, y, gdm_params, 5, 'traingdm', @assign_gdm);
    elseif best == rp_accuracy
        net = buildNet(1, x, y, rp_params, 5, 'trainrp', @assign_rp);
    else
        net = buildNet(1, x, y, gd_params, 4, 'traingd', @assign_gd);
    end
end

function [ result ] = assign_gda( net, args )
    net.trainParam.lr     = args(3);
    net.trainParam.lr_inc = args(4);
    net.trainParam.lr_dec = args(5);
    result = net;
end

function [ result ] = assign_rp( net, args )
    net.trainParam.delt_inc = args(3);
    net.trainParam.delt_dec = args(4);
    result = net;
end

function [ result ] = assign_gdm( net, args )
    net.trainParam.lr = args(3);
    net.trainParam.mc = args(4);
    result = net;
end

function [ result ] = assign_gd( net, args )
    net.trainParam.lr = args(3);
    result = net;
end

function [ net ] = buildNet(i, input, output, args, layerIndex, functionName, argAssigner)
    rng(1001, 'twister');
    
    if layerIndex == 0 || args(layerIndex) > 0
        net = feedforwardnet([args(1), args(2)], functionName);
    else
        net = feedforwardnet([args(1)], functionName);
    end
    
    [tI, tO] = ANNdata(input, output);
    valInd = i:10:size(tI, 2);
    trainInd = (1:size(tI, 2));
    trainInd(valInd) = [];
    
    net = configure(net, tI, tO);
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainInd;
    net.divideParam.valInd = valInd;
    net.divideParam.testInd = [];

    net = argAssigner(net, args);

    % Suppress output and train.
    net.trainParam.show = NaN;
    net = train(net, tI, tO);
end

function [ max_accuracy, max_inputs ] = findBestForTrainingFunction( optiset, functionName, layerIndex, argAssigner, x, y )
    max_accuracy = 0;
    max_inputs = [];
    
    if isempty(optiset)
        return
    end

    parfor j=1:10
        confusedMatrix = confusionmatrix();

        for i=1:10,
            % Remove 10% of x to form the 9fold data
            foldInput = x;
            foldInput(i:10:end,:) = [];

            foldOutput = y;
            foldOutput(i:10:end) = [];

            % Use the other 10% of x for test data
            testInput = x(i:10:end,:);
            testOutput = y(i:10:end);

            args = optiset(j, :);

            net = buildNet(i, foldInput, foldOutput, args, layerIndex, functionName, argAssigner);
            
            % Update the confusion matrix with the test data for this fold.
            confusedMatrix.update(net, testInput, testOutput);
        end

        results(j) = confusedMatrix.getAccuracy();
    end

    max_accuracy = max(results(:));
    max_index = find(results == max_accuracy);
    max_inputs = optiset(max_index, :);
end

