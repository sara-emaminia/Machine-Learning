function [tH,st,dt,bt] = EstimateDataComponent(data,trend_degree)

T = length(data);
index = 1:length(data);

t = (1:T)';
if trend_degree == 1
    X = [ones(T,1) t];
elseif trend_degree == 2
    X = [ones(T,1) t t.^2];
elseif trend_degree ==3
    X = [ones(T,1) t t.^2 t.^3];
else
    disp('Higher trend degree than 3 is not supported');
    return;
end

b = X\data;
tH = X*b;

% Original data & Trend Plot
figure;
plot(index, data, 'b', 'LineWidth', 1); % Original data
xlabel('Index');
ylabel('Difference in kWh');
legend('Original Data');
hold on;
h2 = plot(index,tH,'r','LineWidth',2);
legend(h2,'Quadratic Trend Estimate')
hold off;

% Detrending
xt = data - tH;

% Seasonal component extracting
period = 365;
fullCycles = floor(T / period);  % Number of complete cycles
remainingDays = mod(T, period);  % Remaining days after full cycles
ye = repmat((1:period)', fullCycles, 1);
if remainingDays > 0
    ye = [ye; (1:remainingDays)'];  % Append 1 to remainingDays
end

sX = dummyvar(ye);
  
bS = sX\xt;
st = sX*bS;

% Seasonal Component Plot
figure;
plot(index,st);
ylabel 'Energy Generation(kWh)';
title('Parametric Estimate of Seasonal Component (Indicators)');

% Deseasonalized data
dt = data - st;

% Deseasonalized Data
figure;
plot(index,dt);
title('Daily Energy Generation(kWh) (Deseasonalized)');
ylabel('Energy Generation(kWh)');

% Estimate Irregular Component
bt = data - tH - st;

% Irregular Component
figure;
plot(index,bt);
title('Irregular Component');
ylabel('Energy Generation(kWh)');
end