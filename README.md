<span style="font-size:32px;">
This repository contains MATLAB scripts for forecasting solar power generation using the ARIMA, LSTM and BiLSTM models.
</span>  

## Files in this repository

### data.mat
Contains the dataset for solar power generation used in this study. The data includes energy production measurements over time, which have been preprocessed for seasonal, trend, and irregular analysis.  

### ARIMA_MainBody.m
The main script that implements the ARIMA forecasting model on the preprocessed data. This script loads the data.mat file, applies seasonal adjustments, fits the ARIMA model to the deseasonalized data, and generates forecasts.

### EstimateDataComponent.m
This script identifies the seasonal, trend, and irregular components of the solar energy data. It prepares the data for ARIMA by decomposing it, enabling the model to better capture the patterns in the time series. 

### Forcaster.m
The Forecaster class provides tools for forecasting time series data using LSTM or BiLSTM networks. It includes data preparation, model configuration, training, and performance evaluation.
#### Properties
##### MimariNo
Determines the network architecture; 1 for BiLSTM and 2 for LSTM.
##### HyperParameters
Stores hyperparameters for model training, including optimizer, learning rate, and hidden units.
##### TrainRatio
Fraction of data used for training (default 0.8).
##### Data
The main input time series data.
##### FeatureDimension & NumResponses
Dimensions of the input and output; set based on the data and strategy.
##### DataTrain & DataTest
Training and test splits based on TrainRatio.
##### XTrain, YTrain, XTest, YTest
Training and test sets for input-output sequences.
##### muTrain, muTest, sigTrain, sigTest
Means and standard deviations for data normalization.
##### StandardizationBool
Specifies if data should be standardized.
##### Net
The trained neural network.
##### YTrainPredicted, YTestPredicted
Predictions for training and test sets.
##### PerfText
Metrics to evaluate performance, including MAE, MSE, RMSE, MAPE, SMAPE, BIAS, and CORRELATION.
##### Options
Training options for the neural network.
##### YTrain or YTest, One or Two Step Ahead Prediction and Performances properties:
These 12 properties Stores true values or predictions values or performance metrics for one-step and two-step ahead forecasts.

#### Methods
##### Forecaster(MimariNo)
Initializes the Forecaster object with a specified architecture (MimariNo) and sets up HyperParameters.
##### trainAndCalculatePerformances
-Prepares data for training, sets up the BiLSTM or LSTM network architecture, configures training options, and trains the network.  
-Chooses network layers based on MimariNo: BiLSTM layers if MimariNo=1 or LSTM layers with dropout if MimariNo=2.  
-Configures training options with specified optimizer, epochs, and learning rate, and uses GPU if available.  
-After training, it calls calcPerformances to evaluate prediction accuracy.
##### calcPerformances
-Uses the trained network to generate predictions for both training and test sets.  
-Splits predictions into one-step and two-step ahead forecasts (if applicable).  
-Calls calcPerformance to calculate metrics (MAE, MSE, RMSE, MAPE, SMAPE, BIAS, and CORRELATION) for each prediction type (one-step and two-step, for both training and test sets).
##### plotPerformances
Plots the actual vs. forecasted values to visually evaluate the model’s performance on the test data.
##### prepareDataset
-Prepares and splits the data into training and test sets based on TrainRatio.  
-Standardizes data if StandardizationBool is set to 1.  
-Creates sequences of data for input (XTrain, XTest) and target (YTrain, YTest) values using a sliding window based on hyperparameter settings (TimeSteps and PredLength).
#### Private Methods
##### calcPerformance
-Calculates various performance metrics between actual (ya) and predicted (yf) values, including MAE, MSE, RMSE, MAPE, SMAPE, BIAS, and correlation.  
-Returns a vector of metrics to evaluate forecast accuracy.

### HyperParameters.m
The HyperParameters class generates a table of hyperparameter configurations for optimizing the BiLSTM/LSTM model. This table stores various combinations of parameters that can be used during model training to test different configurations.
#### Properties
##### Table
A table that holds all possible combinations of hyperparameters, including TimeSteps, PredLength, LearningRate, HiddenUnits, Optimizer, and MaxEpochs.
#### Methods
##### Constructor (HyperParameters())
-Initializes the Table property by generating all combinations of specified hyperparameters  
-Iterates through each possible combination of these parameters and stores them in a structured table (Table). This table can be accessed to select specific hyperparameter configurations during model training.

### Runthis.m
The Runthis.m script is the main entry point for executing the BiLSTM/LSTM forecasting model. This script orchestrates the entire process by initializing classes, configuring hyperparameters, preparing data, training the model, and evaluating its performance.

### ForecasterPSO.m
The ForecasterPSO class is similar to the Forecaster class, with a few key differences to support Particle Swarm Optimization (PSO) for hyperparameter tuning:  

-**Hyperparameter Handling**: Unlike Forecaster, ForecasterPSO does not use a HyperParameters table. Instead, it directly defines the hyperparameters as class properties, such as numHiddenUnits, Optimizer, MaxEpochs, and LearningRate. These values are likely set by the PSO algorithm externally before training.  

-**PSO Integration**: The class is designed to work with a PSO framework, where hyperparameters are updated iteratively by the PSO algorithm. This approach allows dynamic optimization of the model’s configuration without needing predefined combinations in a table.  

-**Simplified Initialization**: In ForecasterPSO, the constructor (ForecasterPSO(MimariNo)) only sets the MimariNo property. It doesn’t initialize hyperparameters since these are adjusted by the PSO during optimization.

-**Data Preparation Adjustments**: The prepareDataset method uses FeatureDimension and NumResponses directly, without relying on indices to reference pre-defined hyperparameter settings.

### RunthisPSO.m
The RunthisPSO.m script is similar to Runthis.m but includes a few significant differences to support Particle Swarm Optimization (PSO) for hyperparameter tuning:  

-**PSO Implementation**: RunthisPSO.m uses a PSO algorithm to optimize hyperparameters dynamically. It initializes particles (each representing a unique set of hyperparameters) and iteratively updates their positions and velocities to minimize forecasting error, adapting based on the best results.  

-**Hyperparameter Ranges**: Instead of predefined values, this script defines ranges for continuous hyperparameters (maxEpochs, numHiddenUnits, learningRate) and sets for discrete hyperparameters (TimeSteps, PredLength, optimizerSet). PSO searches within these ranges, refining the configurations with each iteration.  

-**Error Evaluation Function**: evaluateError uses the ForecasterPSO class to set and train the LSTM/BiLSTM model with the hyperparameters from each particle’s position, then evaluates the model’s error to guide PSO optimization.  

-**Selection Functions**: The script includes utility functions (sigmoid, softmax, oneHot) to handle categorical and continuous parameter updates, ensuring appropriate ranges and probability-based choices for parameters like the optimizer type.  


## Note
Ensure all files are in the same directory before running with MATLAB.
