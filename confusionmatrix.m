classdef confusionmatrix < handle
    
    properties
        Matrix
    end

    methods
        function this = confusionmatrix()

            % Initialise the confusion matrix.
            this.Matrix = zeros(6, 6);
        end
        
        function update(this, net, inputs, outputs)
            % Test the forest on each entry in the training data, then
            % increment the corresponding element in the matrix.
            [i, o] = ANNdata(inputs, outputs);
            emotion = testANN(net, i);
            for idx=1:length(emotion)
                this.Matrix(outputs(idx), emotion(idx)) = ...
                        this.Matrix(outputs(idx), emotion(idx)) + 1;
            end        
            

        end
        
        % Identical to update, but assumes that ANNdata has already been
        % called on the parameters.
        function updateWithoutConvert(this, net, inputs, outputs)
            for col=1:size(inputs, 2),
                emotion = testANN(net, inputs(:, col));
                actualEmotion = find(outputs(:, col), 1);
                this.Matrix(actualEmotion, emotion) = ...
                    this.Matrix(actualEmotion, emotion) + 1;
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
