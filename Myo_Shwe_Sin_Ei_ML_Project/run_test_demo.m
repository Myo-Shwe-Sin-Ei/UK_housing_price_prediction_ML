%% UK Housing Price Prediction: Ridge Regression vs Random Forest Testing Demo Code

clear; clc; close all;

fprintf('UK Housing Price Prediction\n');
fprintf('-----------------------------------------------------------\n\n');

%% this is to load the trained models

fprintf('Loading trained models and preprocessing info\n');

load('models/ridge_model.mat');
load('models/rf_model.mat');
load('models/preprocessing_info.mat');
load('models/data_split.mat');

fprintf('Ridge Regression loaded (lambda = %.4f)\n', ridge_best_lambda);
fprintf('Random Forest loaded (%d trees, MinLeaf=%d)\n', ...
    rf_best_params.NumTrees, rf_best_params.MinLeafSize);

%% this is to print dataset info

fprintf('\nDataset Information\n');
fprintf('-----------------------------------------------------------\n\n');
fprintf('Training + Validation set: %d samples\n', size(X_trainval, 1));
fprintf('Test set: %d samples\n', size(X_test, 1));
fprintf('Number of features: %d\n', size(X_test, 2));

%% this is for features

fprintf('\nFeatures Used\n');
fprintf('-----------------------------------------------------------\n\n');
for i = 1:length(feature_names)
    fprintf('  %2d. %s\n', i, feature_names{i});
end

%% this is to standardize test data for Ridge using training stats

fprintf('\nPreprocessing Test Data\n');
fprintf('-----------------------------------------------------------\n\n');
fprintf('Using standardization parameters from training data\n');
fprintf('Mean (first 5 features): [%s]\n', sprintf('%.3f ', preprocessing_info.mu(1:5)));
fprintf('Std  (first 5 features): [%s]\n', sprintf('%.3f ', preprocessing_info.sigma(1:5)));

X_test_scaled = (X_test - preprocessing_info.mu) ./ preprocessing_info.sigma;

%% this is for model predictions

fprintf('\nMaking Predictions\n');
fprintf('-----------------------------------------------------------\n');

% Ridge regression
fprintf('\nRidge Regression predictions\n');
tic;
X_test_aug = [ones(size(X_test_scaled, 1), 1), X_test_scaled];
y_pred_ridge = X_test_aug * ridge_model.weights;
ridge_time = toc;
fprintf(' - Prediction time: %.4f seconds\n', ridge_time);
fprintf(' - Model complexity: %d weights (including intercept)\n', length(ridge_model.weights));

% Random Forest predictions
fprintf('\nRandom Forest predictions\n');
tic;
y_pred_rf = predict(rf_model, X_test);
if iscell(y_pred_rf)
    y_pred_rf = cell2mat(y_pred_rf);
end
rf_time = toc;
fprintf(' - Prediction time: %.4f seconds\n', rf_time);
fprintf(' - Model complexity: %d trees\n\n', rf_best_params.NumTrees);

%% this is to calculate metrics

fprintf('Test Set Metrics\n');
fprintf('-----------------------------------------------------------\n');

% Ridge metrics
ridge_rmse = sqrt(mean((y_test - y_pred_ridge).^2));
ridge_mae = mean(abs(y_test - y_pred_ridge));
ridge_r2 = 1 - sum((y_test - y_pred_ridge).^2) / sum((y_test - mean(y_test)).^2);

% Random Forest metrics
rf_rmse = sqrt(mean((y_test - y_pred_rf).^2));
rf_mae = mean(abs(y_test - y_pred_rf));
rf_r2 = 1 - sum((y_test - y_pred_rf).^2) / sum((y_test - mean(y_test)).^2);

fprintf('\n%-25s %-15s %-15s\n', 'Metric', 'Ridge', 'Random Forest');
fprintf('%-25s %-15s %-15s\n', repmat('-', 1, 25), repmat('-', 1, 15), repmat('-', 1, 15));
fprintf('%-25s %-15.4f %-15.4f\n', 'RMSE (log-price)', ridge_rmse, rf_rmse);
fprintf('%-25s %-15.4f %-15.4f\n', 'MAE (log-price)', ridge_mae, rf_mae);
fprintf('%-25s %-15.4f %-15.4f\n', 'R-squared', ridge_r2, rf_r2);
fprintf('%-25s %-15.4f %-15.4f\n', 'Prediction Time (s)', ridge_time, rf_time);

%% this is converting back to original price scale

y_test_original = exp(y_test) - 1;
y_pred_ridge_original = exp(y_pred_ridge) - 1;
y_pred_rf_original = exp(y_pred_rf) - 1;

rmse_ridge_gbp = sqrt(mean((y_test_original - y_pred_ridge_original).^2));
rmse_rf_gbp = sqrt(mean((y_test_original - y_pred_rf_original).^2));

mae_ridge_gbp = mean(abs(y_test_original - y_pred_ridge_original));
mae_rf_gbp = mean(abs(y_test_original - y_pred_rf_original));

fprintf('%-25s £%-14.0f £%-14.0f\n', 'RMSE (original £)', rmse_ridge_gbp, rmse_rf_gbp);
fprintf('%-25s £%-14.0f £%-14.0f\n\n', 'MAE (original £)', mae_ridge_gbp, mae_rf_gbp);

%% this is to print sample predictions

fprintf('Sample Predictions (First 10 Test Samples)\n');
fprintf('-----------------------------------------------------------\n');

fprintf('\n%-6s %-15s %-15s %-15s %-12s %-12s\n', ...
    'Index', 'Actual (£)', 'Ridge (£)', 'RF (£)', 'Ridge Err%', 'RF Err%');
fprintf('%-6s %-15s %-15s %-15s %-12s %-12s\n', ...
    repmat('-', 1, 6), repmat('-', 1, 15), repmat('-', 1, 15), ...
    repmat('-', 1, 15), repmat('-', 1, 12), repmat('-', 1, 12));

for i = 1:min(10, length(y_test))
    ridge_err = 100 * abs(y_test_original(i) - y_pred_ridge_original(i)) / y_test_original(i);
    rf_err = 100 * abs(y_test_original(i) - y_pred_rf_original(i)) / y_test_original(i);
    
    fprintf('%-6d £%-14.0f £%-14.0f £%-14.0f %-11.1f%% %-11.1f%%\n\n', ...
        i, y_test_original(i), y_pred_ridge_original(i), y_pred_rf_original(i), ...
        ridge_err, rf_err);
end

%% this is for visualizations

% Figure 1: Residual Distribution
figure('Position', [100, 100, 1000, 400], 'Name', 'Residual Distribution');

residuals_ridge = y_test - y_pred_ridge;
residuals_rf = y_test - y_pred_rf;

subplot(1, 2, 1);
histogram(residuals_ridge, 40, 'FaceColor', [0.2, 0.4, 0.8], 'Normalization', 'pdf');
hold on;
x_range = linspace(min(residuals_ridge), max(residuals_ridge), 100);
plot(x_range, normpdf(x_range, mean(residuals_ridge), std(residuals_ridge)), 'r-', 'LineWidth', 2);
xline(0, 'k--', 'LineWidth', 1.5);
hold off;
xlabel('Residual (log-price)', 'FontSize', 11);
ylabel('Density', 'FontSize', 11);
title(sprintf('Ridge Residuals\nMean=%.4f, Std=%.4f', mean(residuals_ridge), std(residuals_ridge)), 'FontSize', 12);
legend('Histogram', 'Normal Fit', 'Zero Line');
grid on;

subplot(1, 2, 2);
histogram(residuals_rf, 40, 'FaceColor', [0.8, 0.4, 0.2], 'Normalization', 'pdf');
hold on;
x_range = linspace(min(residuals_rf), max(residuals_rf), 100);
plot(x_range, normpdf(x_range, mean(residuals_rf), std(residuals_rf)), 'r-', 'LineWidth', 2);
xline(0, 'k--', 'LineWidth', 1.5);
hold off;
xlabel('Residual (log-price)', 'FontSize', 11);
ylabel('Density', 'FontSize', 11);
title(sprintf('Random Forest Residuals\nMean=%.4f, Std=%.4f', mean(residuals_rf), std(residuals_rf)), 'FontSize', 12);
legend('Histogram', 'Normal Fit', 'Zero Line');
grid on;

sgtitle('Residual Distribution Comparison', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: Error Distribution in GBP
figure('Position', [100, 100, 800, 400], 'Name', 'Prediction Errors in GBP');

errors_ridge_gbp = y_test_original - y_pred_ridge_original;
errors_rf_gbp = y_test_original - y_pred_rf_original;

subplot(1, 2, 1);
histogram(errors_ridge_gbp/1000, 40, 'FaceColor', [0.2, 0.4, 0.8]);
xlabel('Prediction Error (£ thousands)', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title('Ridge: Price Prediction Errors', 'FontSize', 12);
xline(0, 'r--', 'LineWidth', 2);
grid on;

subplot(1, 2, 2);
histogram(errors_rf_gbp/1000, 40, 'FaceColor', [0.8, 0.4, 0.2]);
xlabel('Prediction Error (£ thousands)', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title('Random Forest: Price Prediction Errors', 'FontSize', 12);
xline(0, 'r--', 'LineWidth', 2);
grid on;

sgtitle('Prediction Errors in Original Price Scale', 'FontSize', 14, 'FontWeight', 'bold');

%% this is to print the better model stats

fprintf('\nModel Comparison\n');
fprintf('-----------------------------------------------------------\n\n');
if rf_rmse < ridge_rmse
    improvement = 100 * (ridge_rmse - rf_rmse) / ridge_rmse;
    fprintf('Better model: Random Forest\n');
    fprintf(' - %.1f%% lower RMSE (log-price)\n', improvement);
    fprintf(' - R² = %.4f vs %.4f\n', rf_r2, ridge_r2);
else
    improvement = 100 * (rf_rmse - ridge_rmse) / rf_rmse;
    fprintf('Better model: Ridge Regression\n');
    fprintf(' - %.1f%% lower RMSE (log-price)\n', improvement);
    fprintf(' - R² = %.4f vs %.4f\n', ridge_r2, rf_r2);
end

fprintf('\nSpeed Comparison:\n');
fprintf(' - Ridge prediction time: %.4f seconds\n', ridge_time);
fprintf(' - Random Forest prediction time: %.4f seconds\n', rf_time);
if ridge_time < rf_time
    fprintf(' - Ridge is %.1fx faster\n\n', rf_time/ridge_time);
else
    fprintf(' - Random Forest is %.1fx faster\n\n', ridge_time/rf_time);
end

fprintf('Demo Complete!');