classdef HyperParameters
    properties
        Table
    end
    methods
        function obj = HyperParameters()
            % hyperParameters.hyperIndex = hyperIndex;
            TimeSteps = [1,2];
            PredLength = [1,2];
            optimizerSet ={'adam','sgdm','rmsprop'};
            maxEpochs =[200, 400, 600];
            numHiddenUnits = [100,200,400];
            learningRate = [0.005];
            say = 0;
            for l = 1 : numel(TimeSteps)
                for p = 1 : numel(PredLength)
                    for m = 1 : numel(learningRate)
                        for k = 1 : numel(numHiddenUnits)
                            for i = 1 : numel(optimizerSet)
                                for j = 1 : numel(maxEpochs)
                                    say = say + 1;
                                    var1{say} = optimizerSet{i};
                                    var2(say) = maxEpochs(j);
                                    var3(say) = numHiddenUnits(k);
                                    var4(say) = learningRate(m);
                                    var5(say) = TimeSteps(l);
                                    var6(say) = PredLength(p);
                                end
                            end
                        end
                    end
                end
            end
            obj.Table = ...
                table([1:say]', var5', var6', var4', var3', var1', var2', ...
                'VariableNames', {'indexNo', 'TimeSteps', 'PredLength', ...
                'LearningRate', 'HiddenUnits', 'Optimizer','MaxEpochs'});
 
             
        end
    end     
end