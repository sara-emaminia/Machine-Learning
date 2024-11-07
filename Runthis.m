clc;
clear all;
rng(1)
mimari_no = 1;
t = Forecaster(mimari_no);

for i =  1 : 108
    load data.mat
    t.Data=data;
    t.StandardizationBool=1;
    hyperIndex = i;
    t.HyperIndex = hyperIndex;
    t = t.trainAndCalculatePerformances;
    MAPEOneStepTrain(i) = t.OneStepAheadPredPerformancesOnTrainData(4);
    MAPETwoStepTrain(i) = t.TwoStepAheadPredPerformancesOnTrainData(4);
    MAPEOneStepTest(i) = t.OneStepAheadPredPerformancesOnTestData(4);
    MAPETwoStepTest(i) = t.TwoStepAheadPredPerformancesOnTestData(4);
end
MAPEOneStepTrain = MAPEOneStepTrain';
MAPETwoStepTrain = MAPETwoStepTrain';
MAPEOneStepTest = MAPEOneStepTest';
MAPETwoStepTest = MAPETwoStepTest';

ErrorTable = table(MAPEOneStepTrain, MAPETwoStepTrain, MAPEOneStepTest, MAPETwoStepTest, ...
    'VariableNames', {'MAPEOneStepTrain', 'MAPETwoStepTrain', 'MAPEOneStepTest', 'MAPETwoStepTest'});

    % Plot
    figure
    plot(length(t.DataTrain)+2:length(t.Data), ...
        t.YTestTwoStepAheadGroundTruth*t.sigTest+t.muTest,'b', 'LineWidth', 2), 
    hold on
    plot(length(t.DataTrain)+2:length(t.Data), ...
        t.YTestTwoStepAheadPredictions*t.sigTest+t.muTest, '--', 'Color', 'r', 'LineWidth', 1)
    title('2-Step Ahead test forecasting')
    legend('Original test data', 'Forecasting test data')
    xlabel('day')
    ylabel('Energy Generation(kWh)')
    figure
    plot(t.YTrainOneStepAheadPredictions,'b', 'LineWidth', 1.5), hold on
    plot(t.YTrainOneStepAheadGroundTruth, 'r', 'LineWidth', 1)
    title('Train Performance')