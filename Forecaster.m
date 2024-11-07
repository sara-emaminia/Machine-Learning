classdef Forecaster
    properties
        MimariNo %1: bilstm, 2: lstm.
        HyperParameters % HyperTable class.
        HyperIndex = 1 ; % change accordingly before train. 
        TrainRatio = 0.8; % default.
        Data % must be given.
        FeatureDimension,% will be determined according to StrategyNo.
        NumResponses,% will be determined according to StrategyNo.
        DataTrain,% will be determined according to TrainRatio.
        DataTest,% will be determined according to TestRatio.
        XTrain, YTrain, XTest, YTest,
        % XTrain_, YTrain_, XTest_, YTest_,
        muTrain, muTest,
        sigTrain, sigTest,
        StandardizationBool=0 %if 1 standardize data, if 0 donot.
        Net, % Deep Neural Network.
        YTrainPredicted, YTestPredicted
        PerfText = 'MAE,MSE,RMSE,MAPE,SMAPE,BIAS,CORRELATION';
        Options

        YTrainTwoStepAheadPredictions
        YTrainTwoStepAheadGroundTruth

        YTrainOneStepAheadPredictions
        YTrainOneStepAheadGroundTruth

        YTestTwoStepAheadPredictions
        YTestTwoStepAheadGroundTruth

        YTestOneStepAheadPredictions
        YTestOneStepAheadGroundTruth

        TwoStepAheadPredPerformancesOnTrainData
        TwoStepAheadPredPerformancesOnTestData
        OneStepAheadPredPerformancesOnTrainData
        OneStepAheadPredPerformancesOnTestData
    end

    methods
        function this = Forecaster(MimariNo)
            this.HyperParameters = HyperParameters();
            this.MimariNo = MimariNo;
        end

        function this = trainAndCalculatePerformances(this)
            this = this.prepareDataset;
            this.NumResponses = size(this.YTrain,1);
            this.FeatureDimension = size(this.XTrain,1);

            numHiddenUnits = this.HyperParameters.Table(this.HyperIndex,:).HiddenUnits;

            if this.MimariNo == 1
                layers = [ ...
                    sequenceInputLayer(this.FeatureDimension)
                    bilstmLayer(numHiddenUnits)
                    fullyConnectedLayer(this.NumResponses)
                    regressionLayer];
            elseif this.MimariNo == 2
                layers = [ ...
                    sequenceInputLayer(this.FeatureDimension)
                    lstmLayer(numHiddenUnits)
                    dropoutLayer(0.5)
                    fullyConnectedLayer(this.NumResponses)
                    regressionLayer];
            end

            this.Options = trainingOptions(this.HyperParameters.Table(this.HyperIndex,:).Optimizer, ... %adam sgmd vs.
                'MaxEpochs',this.HyperParameters.Table(this.HyperIndex,:).MaxEpochs, ...
                'GradientThreshold',1, ...
                'MiniBatchSize', 64, ...
                'ValidationPatience', 5, ...
                'InitialLearnRate',this.HyperParameters.Table(this.HyperIndex,:).LearningRate, ...
                'LearnRateSchedule','none', ...
                'LearnRateDropPeriod',125, ...
                'LearnRateDropFactor',0.2, ...
                'Verbose',0, ...
                'Plots','training-progress', ...%'SequenceLength',168*4)
                'SequenceLength',168*4, ...
                ExecutionEnvironment='gpu');

            this.Net = trainNetwork(this.XTrain,this.YTrain,layers,this.Options);
            
            this = this.calcPerformances;
        end
      
        function  this = calcPerformances(this)
                % Train
                net = resetState(this.Net);
                [net, this.YTrainPredicted]= predictAndUpdateState(net,this.XTrain);

                % Test
                net = resetState(this.Net);
                [~, this.YTestPredicted]= predictAndUpdateState(net,this.XTest);

                if this.NumResponses == 2

                    this.YTrainTwoStepAheadPredictions=this.YTrainPredicted(2,:);
                    this.YTrainTwoStepAheadGroundTruth=this.YTrain(2,:);
                    this.YTrainOneStepAheadPredictions=this.YTrainPredicted(1,:);
                    this.YTrainOneStepAheadGroundTruth=this.YTrain(1,:);

                    this.YTestTwoStepAheadPredictions=this.YTestPredicted(2,:);
                    this.YTestTwoStepAheadGroundTruth=this.YTest(2,:);
                    this.YTestOneStepAheadPredictions=this.YTestPredicted(1,:);
                    this.YTestOneStepAheadGroundTruth=this.YTest(1,:);

                elseif this.NumResponses == 1

                    this.YTrainTwoStepAheadPredictions=nan(1, numel(this.YTrainPredicted));
                    this.YTrainTwoStepAheadGroundTruth=nan(1, numel(this.YTrainPredicted));
                    this.YTrainOneStepAheadPredictions=this.YTrainPredicted(1,:);
                    this.YTrainOneStepAheadGroundTruth=this.YTrain(1,:);

                    this.YTestTwoStepAheadPredictions=nan(1, numel(this.YTestPredicted));
                    this.YTestTwoStepAheadGroundTruth=nan(1, numel(this.YTestPredicted));
                    this.YTestOneStepAheadPredictions=this.YTestPredicted(1,:);
                    this.YTestOneStepAheadGroundTruth=this.YTest(1,:);

                end

                %Train
                this.TwoStepAheadPredPerformancesOnTrainData = ...
                    this.calcPerformance(...
                    this.YTrainTwoStepAheadGroundTruth*this.sigTrain+this.muTrain,...
                    this.YTrainTwoStepAheadPredictions*this.sigTrain+this.muTrain);

                this.OneStepAheadPredPerformancesOnTrainData = ...
                    this.calcPerformance(...
                    this.YTrainOneStepAheadGroundTruth*this.sigTrain+this.muTrain,...
                    this.YTrainOneStepAheadPredictions*this.sigTrain+this.muTrain);

                %Test
                this.TwoStepAheadPredPerformancesOnTestData = ...
                    this.calcPerformance(...
                    this.YTestTwoStepAheadGroundTruth*this.sigTest+this.muTest,...
                    this.YTestTwoStepAheadPredictions*this.sigTest+this.muTest);

                this.OneStepAheadPredPerformancesOnTestData = ...
                    this.calcPerformance(...
                    this.YTestOneStepAheadGroundTruth*this.sigTest+this.muTest,...
                    this.YTestOneStepAheadPredictions*this.sigTest+this.muTest);

        end

        function  plotPerformances(this)
            figure,
            plot(this.YPredSeries), hold on
            plot(this.YTestSeries)
            legend('forecast','actual')
        end

        function this = prepareDataset(this)
            if size(this.Data,2)>1
                error('This is for one-dimensional and column data')
            end
            this.Data=this.Data(:);%make it one dimensional column data anyway.
            TrainSize = floor(this.TrainRatio * numel(this.Data));
            TestSize = numel(this.Data) - TrainSize;
            this.DataTrain = this.Data(1:TrainSize);
            this.DataTest = this.Data(TrainSize+1:end);
            % Standardize Data
            if this.StandardizationBool==0
                this.muTrain = 0; %mean(this.DataTrain,2);
                this.sigTrain = 1; %std(this.DataTrain,0,2);
                this.muTest = 0; %mean(this.DataTest,2);
                this.sigTest = 1; % std(this.DataTest,0,2);
            else
                this.muTrain = mean(this.DataTrain);
                this.sigTrain = std(this.DataTrain,0);
                this.muTest = mean(this.DataTest);
                this.sigTest = std(this.DataTest,0);
            end

            if sum(isnan(this.muTrain))|| sum(isnan(this.sigTrain)) || sum(isnan(this.muTest))...
                    || sum(isnan(this.sigTest))
                error('NAN')
            end

            L = this.HyperParameters.Table(this.HyperIndex,:).TimeSteps;
            P = this.HyperParameters.Table(this.HyperIndex,:).PredLength;

                this.XTrain = zeros(L, TrainSize-L-P+1);
                this.YTrain = zeros(P, TrainSize-L-P+1);
                this.XTest = zeros(L, TestSize-P+1);
                this.YTest = zeros(P, TestSize-P+1);

                for i=1:L
                k = i;
                    for j=1:TrainSize-L-P+1
                        this.XTrain(i,j) = (this.DataTrain(k) - this.muTrain) / this.sigTrain;
                        k = k + 1;
                    end
                end

                for i=1:P
                k = i;
                    for j=1:TrainSize-L-P+1
                        this.YTrain(i,j) = (this.DataTrain(k+L) - this.muTrain) / this.sigTrain;
                        k = k + 1;
                    end
                end

                for i=1:L
                k = i;
                    for j=1:TestSize-P+1
                        this.XTest(i,j) = (this.Data(k+TrainSize-L) - this.muTest) / this.sigTest;
                        k = k + 1;
                    end
                end

                for i=1:P
                k = i;
                    for j=1:TestSize-P+1
                        this.YTest(i,j) = (this.Data(k+TrainSize) - this.muTest) / this.sigTest;
                        k = k + 1;
                    end
                end
        end
    end

    methods (Access = private)

        function vals = calcPerformance(this,ya,yf)
            ya = ya(:);
            yf = yf(:);
            err = ya-yf;
            MAE = mean(abs(err));
            MSE = mean(err.^2 );
            RMSE = sqrt(MSE);

            errpct = abs(err)./abs(ya)*100;
            MAPE = mean(errpct(~isinf(errpct)));
            errpct = abs(err)./(abs(ya) + abs(yf));
            SMAPE = mean(errpct(~isinf(errpct)))*200;
            BIAS = mean(err) ;
            CORRELATION = corr(double(ya),double(yf)) ;
            vals = [MAE, MSE, RMSE, MAPE, SMAPE, BIAS, CORRELATION];
        end
    end
end
