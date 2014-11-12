classdef confusionmatrix < handle
    
    properties
        Matrix
    end

    methods
        function this = confusionmatrix()

            % Initialise the confusion matrix.
            this.Matrix = ones(6, 6);
        end
        
        function update(this, net, inputs, outputs)
            % Test the forest on each entry in the training data, then
            % increment the corresponding element in the matrix.
            for row=1:length(inputs),
                [i, o] = ANNdata(inputs(row, :), outputs(row));
                emotion = testANN(net, i);
                actualEmotion = outputs(row);
                % TODO: Check that this case can be ignored.
                if emotion ~= -1
                    this.Matrix(actualEmotion, emotion) = ...
                        this.Matrix(actualEmotion, emotion) + 1;
                end
            end
        end
        
        % Functions for part III, these functions calculate the basic
        % properties from the slides;
        
        % Return the accuracy of the matrix.
        function accuracy = getAccuracy(this)
            accuracy = trace(this.Matrix) / sum(this.Matrix(:));
        end

        % Return the recall rate of the given class.
        function recallRate = getRecallRate(this, class)
            recallRate = this.Matrix(class, class) / sum(this.Matrix(class, :));
        end
        
        % Return the precision of the given class.
        function precision = getPrecision(this, class)
            precision = this.Matrix(class, class) / sum(this.Matrix(:, class));
        end
        
        % Return the f score of the given class with parameter alpha.
        function fScore = getFScore(this, class, alpha)
            recallRate = this.getRecallRate(class);
            precision = this.getPrecision(class);
            
            fScore = (1 + alpha^2) * (precision * recallRate) ...
                / (alpha^2 * precision + recallRate);
        end
    end
end
