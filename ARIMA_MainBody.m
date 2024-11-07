clc; clear; close all;

% Load the data
load data.mat;

T = length(data);
index = 1:length(data);

% tH : fitted trend, st : Seasonal Component,
% dt : deseasonalized data, bt : Irregular Component
[tH,st,dt,bt] = EstimateDataComponent(data,1);

% Perform ADF test
[h, pValue, stat, cValue_data] = adftest(dt);

% Display ADF test results
fprintf('ADF Test Result: \n');
fprintf('Test Statistic: %f\n', stat);
fprintf('P-Value: %f\n', pValue);

% Split data (dt) into train and test
train_size = round(0.8 * length(dt));
train_data = dt(1:train_size);
test_data = dt(train_size+1:end);

figure;
subplot(2,1,1);
autocorr(train_data, 'NumLags', 40);
title('ACF of Training Data');

subplot(2,1,2);
parcorr(train_data, 'NumLags', 40);
title('PACF of Training Data');

% Fitting ARIMA on the train data
model = arima(2,1,0);
fit = estimate(model, train_data);

% Forecast the train data

y_train(1:3) = train_data(1:3);
for i = 4:train_size
    [y_train(i),YMSE] = forecast(fit, 1, train_data(1:i));
end

% Forecast the test data
num_test_steps = length(test_data);
for i=1:num_test_steps
    [y(i), yMSE] = forecast(fit, 1, [train_data;test_data(1:i)]);
end

y_train = y_train';
y = y';

train_residuals = train_data - y_train;

% Plot Train Residuals
figure;
plot(1:train_size,train_residuals);
title('Train Data Forecast Residuals');
ylabel('Value(kWh)');

% ACF & PACF of Train Residuals
figure;
subplot(2,1,1);
autocorr(train_residuals, 'NumLags', 40);
title('ACF of Train Data Forecast Residuals');

subplot(2,1,2);
parcorr(train_residuals, 'NumLags', 40);
title('PACF of Train Data Forecast Residuals');

% Fitting ARIMA on the train forecast residuals
model_residuals = arima(2,0,2);
fit_residuals = estimate(model_residuals, train_residuals);

% Forecast the test data forecast residuals
test_residuals = test_data - y;
y_test_residuals(1) = forecast(fit_residuals, 1, train_residuals);
for i=2:num_test_steps
    [y_test_residuals(i), yMSE] = forecast(fit_residuals, 1, [train_residuals;test_residuals(1:i-1)]);
end

% Adding seasonal component
y = y + st(train_size+1:end);
y_train = y_train + st(1:train_size);
train_data = train_data + st(1:train_size);
test_data = test_data + st(train_size+1:end);

% Test Forecast Plot
figure;
hold on;
plot(train_size+1:length(dt), test_data, 'b', 'LineWidth', 1.5);
plot(train_size+1:length(dt), y, 'r', 'LineWidth', 1);
ylim([-50 400]);
title('Initial Test Forecast and Original Test plot');
xlabel('day');
ylabel('Energy Generation(kWh)');
legend('Original Data', 'Forecast');

% Test residuals Plot
figure;
plot(train_size+1:length(dt), y_test_residuals, 'r', 'LineWidth', 1);
title('Test Residuals Forecast');
xlabel('day');
ylabel('Energy Generation(kWh)');

y_test_residuals = y_test_residuals';

y_test = y + y_test_residuals;

% Calculate MAPE
mape_y = mape(test_data, y);
mape_y_test = mape(test_data, y_test);

% Final Test Forecast Plot
figure;
hold on;
plot(train_size+1:length(dt), test_data, 'b', 'LineWidth', 1.5);
plot(train_size+1:length(dt), y_test, 'r', 'LineWidth', 1);
ylim([-50 400]);
title('Final Test Forecast and Original Test plot');
xlabel('day');
ylabel('Energy Generation(kWh)');
legend('Original Data', 'Forecast');

% Training data, Test data, and the forecast Plot
figure;
hold on;
plot(1:train_size, train_data, 'k', 'LineWidth', 1.5); % Training data
plot(train_size+1:length(dt), test_data, 'g', 'LineWidth', 1.5); % Test data
plot(train_size+1:length(dt), y_test, 'r', 'LineWidth', 1.5); % Forecast
title('Final Test Data Forecast');
xlabel('day');
ylabel('Energy Generation(kWh)');
legend('Training Data', 'Test Data', 'Forecast');
hold off;