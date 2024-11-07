clc;
clear all;
rng(1234567891)
load data.mat

% Define the continuous hyperParameter's sets
maxEpochs =[500, 1000];
numHiddenUnits = [250, 400];
learningRate = [0.005, 0.001];

% Discrete hyperparameters will be addressed next
% TimeSteps = [1,2];
% PredLength = [1,2];
% optimizerSet ={'adam','sgdm','rmsprop'};

% Start of PSO
% Number of particles and iterations for PSO
numParticles = 20;
maxIterations = 50;

% PSO Parameters
w = 0.7;
c1 = 1.5; 
c2 = 1.5;

% Initialize Particles (struct is used to handle both continuous and
% discrete hyperparameters' sets more conveniently)
particles = struct();

for i = 1 : numParticles
    % Initialize Positions
    particles(i).position.TimeSteps = randi([1, 2]);
    particles(i).position.PredLength = randi([1, 2]);
    particles(i).position.optimizer = randi([1, 3]);  % 1: adam, 2: sgdm, 3: rmsprop
    particles(i).position.maxEpochs = unifrnd(maxEpochs(1), maxEpochs(2));
    particles(i).position.numHiddenUnits = unifrnd(numHiddenUnits(1), numHiddenUnits(2));
    particles(i).position.learningRate = unifrnd(learningRate(1), learningRate(2));

    % Initialize Velocities
    particles(i).velocity = struct();
    particles(i).velocity.TimeSteps = randn();
    particles(i).velocity.PredLength = randn();
    particles(i).velocity.optimizer = randn(1, 3);
    particles(i).velocity.maxEpochs = randn();
    particles(i).velocity.numHiddenUnits = randn();
    particles(i).velocity.learningRate = randn();

    % Ensure learning rate is finite
    if ~isfinite(particles(i).position.learningRate) || ~isnan(particles(i).position.learningRate)
        particles(i).position.learningRate = learningRate(1);
    end

    % Evaluate LSTM error
    particles(i).error = evaluateError(particles(i).position, data);

    % Initialize personal bests
    particles(i).bestPosition = particles(i).position;
    particles(i).bestError = particles(i).error;
end

% Initialize global best
[~, bestIdx] = min([particles.error]);
globalBestPosition = particles(bestIdx).position;
globalBestError = particles(bestIdx).error;

% PSO Iterations
for iter = 1:maxIterations
    for i = 1:numParticles
        % Update velocities
        particles(i).velocity.maxEpochs = w * particles(i).velocity.maxEpochs + ...
            c1 * rand() * (particles(i).bestPosition.maxEpochs - particles(i).position.maxEpochs) + ...
            c2 * rand() * (globalBestPosition.maxEpochs - particles(i).position.maxEpochs);
            
        particles(i).velocity.numHiddenUnits = w * particles(i).velocity.numHiddenUnits + ...
            c1 * rand() * (particles(i).bestPosition.numHiddenUnits - particles(i).position.numHiddenUnits) + ...
            c2 * rand() * (globalBestPosition.numHiddenUnits - particles(i).position.numHiddenUnits);
            
        particles(i).velocity.learningRate = w * particles(i).velocity.learningRate + ...
            c1 * rand() * (particles(i).bestPosition.learningRate - particles(i).position.learningRate) + ...
            c2 * rand() * (globalBestPosition.learningRate - particles(i).position.learningRate);

        particles(i).velocity.TimeSteps = w * particles(i).velocity.TimeSteps + ...
            c1 * rand() * (particles(i).bestPosition.TimeSteps - particles(i).position.TimeSteps) + ...
            c2 * rand() * (globalBestPosition.TimeSteps - particles(i).position.TimeSteps);

        particles(i).velocity.PredLength = w * particles(i).velocity.PredLength + ...
            c1 * rand() * (particles(i).bestPosition.PredLength - particles(i).position.PredLength) + ...
            c2 * rand() * (globalBestPosition.PredLength - particles(i).position.PredLength);

        % oneHot code take an index (1 or 2 or 3) as optimizer representer
        % and generates [1,0,0]='adam' or [0,1,0]='sgdm' or [0,0,1]='rmsprop'
        particles(i).velocity.optimizer = w * particles(i).velocity.optimizer + ...
            c1 * rand() * (oneHot(particles(i).bestPosition.optimizer, 3) - oneHot(particles(i).position.optimizer, 3)) + ...
            c2 * rand() * (oneHot(globalBestPosition.optimizer, 3) - oneHot(particles(i).position.optimizer, 3));
        
        % Update positions
        particles(i).position.maxEpochs = particles(i).position.maxEpochs + particles(i).velocity.maxEpochs;
        particles(i).position.numHiddenUnits = particles(i).position.numHiddenUnits + particles(i).velocity.numHiddenUnits;
        particles(i).position.learningRate = particles(i).position.learningRate + particles(i).velocity.learningRate;
        
        % Sigmoid function guarantees that between 1 and 2 will be selected
        particles(i).position.TimeSteps = 1 + (sigmoid(particles(i).velocity.TimeSteps) >= 0.5);
        particles(i).position.PredLength = 1 + (sigmoid(particles(i).velocity.PredLength) >= 0.5);

        % Select optimizer based on softmax probabilities
        probabilities = softmax(particles(i).velocity.optimizer);
        particles(i).position.optimizer = randsample(1:3, 1, true, probabilities);

        % Boundary checks for continuous variables
        particles(i).position.maxEpochs = min(max(particles(i).position.maxEpochs, maxEpochs(1)), maxEpochs(2));
        particles(i).position.numHiddenUnits = min(max(particles(i).position.numHiddenUnits, numHiddenUnits(1)), numHiddenUnits(2));
        particles(i).position.learningRate = min(max(particles(i).position.learningRate, learningRate(1)), learningRate(2));

        % Ensure learning rate is finite
        if ~isfinite(particles(i).position.learningRate) || ~isnan(particles(i).position.learningRate)
            particles(i).position.learningRate = learningRate(1);
        end
        
        % Evaluate LSTM error
        particles(i).error = evaluateError(particles(i).position, data);

        % Update personal best
        if particles(i).error < particles(i).bestError
            particles(i).bestPosition = particles(i).position;
            particles(i).bestError = particles(i).error;
        end

        % Update global best
        if particles(i).error < globalBestError
            globalBestPosition = particles(i).position;
            globalBestError = particles(i).error;
        end
    end
    disp(iter);
end

% Sigmoid function
function s = sigmoid(x)
    s = 1 / (1 + exp(-x));
end

% Softmax function
function prob = softmax(x)
    expX = exp(x - max(x)); % for numerical stability
    prob = expX / sum(expX);
end

% One-hot encoding
function onehot = oneHot(index, numCategories)
    onehot = zeros(1, numCategories);
    onehot(index) = 1;
end

% Error evaluation function for LSTM (using ForecasterPSO)
function error = evaluateError(position, data)
    rng(1234567891)
    % Create an object of ForecasterPSO
    mimari_no = 1;
    t = ForecasterPSO(mimari_no);
    t.Data = data;
    t.StandardizationBool=1;

    % Set LSTM hyperparameters
    t.FeatureDimension = position.TimeSteps;
    t.NumResponses = position.PredLength;

    switch position.optimizer
        case 1
            t.Optimizer = 'adam';
        case 2
            t.Optimizer = 'sgdm';
        case 3
            t.Optimizer = 'rmsprop';
    end

    t.MaxEpochs = round(position.maxEpochs);
    t.numHiddenUnits = round(position.numHiddenUnits);
    t.LearningRate = position.learningRate;

    % Run the LSTM and
    t = t.trainAndCalculatePerformances;

    % Get the One-Step ahead error on Test Data
    error = t.TwoStepAheadPredPerformancesOnTestData(4);
end