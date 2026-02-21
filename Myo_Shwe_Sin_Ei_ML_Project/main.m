%% UK Housing Price Prediction: Ridge Regression vs Random Forest Training Code

clear; clc; close all;

% setting random seed
rng(42);

% this code is for output directories
if ~exist('models', 'dir'), mkdir('models'); end
if ~exist('results', 'dir'), mkdir('results'); end

%% this is for loading the data

data = readtable('uk_housing_clean.csv');
fprintf('%d samples, %d features\n\n', height(data), width(data));
disp(head(data, 5));

%% this is for exploratory data analysis

% Price statistics
price = data.Price;
stats.Mean = mean(price);
stats.Std = std(price);
stats.Min = min(price);
stats.Q1 = prctile(price, 25);
stats.Median = median(price);
stats.Q3 = prctile(price, 75);
stats.Max = max(price);
stats.Skewness = skewness(price);
stats.Kurtosis = kurtosis(price);
stats.IQR = stats.Q3 - stats.Q1;

stats_table = struct2table(stats);
disp(stats_table);

% Log-transformed price
log_price = log1p(price);

% Figure: Price Distribution (raw vs log-transformed)
fig_price_dist = figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
histogram(price, 50, 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'white');
xlabel('Price (£)', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Distribution of House Prices (Original)', 'FontSize', 14);
hold on;
xline(stats.Mean, 'r-', 'LineWidth', 2, 'Label', sprintf('Mean: £%.0f', stats.Mean));
xline(stats.Median, 'g--', 'LineWidth', 2, 'Label', sprintf('Median: £%.0f', stats.Median));
hold off;

subplot(1, 2, 2);
histogram(log_price, 50, 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'white');
xlabel('log(1 + Price)', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Distribution of House Prices (Log-transformed)', 'FontSize', 14);
hold on;
xline(mean(log_price), 'r-', 'LineWidth', 2, 'Label', sprintf('Mean: %.2f', mean(log_price)));
xline(median(log_price), 'g--', 'LineWidth', 2, 'Label', sprintf('Median: %.2f', median(log_price)));
hold off;

sgtitle('Effect of Log Transformation on Price Distribution', 'FontSize', 16, 'FontWeight', 'bold');

%% this is for features correlation

n_data = height(data);

% this is encoding variables for correlation
property_map = containers.Map({'D', 'S', 'T', 'F', 'O'}, {1, 2, 3, 4, 5});
property_encoded = zeros(n_data, 1);
for i = 1:n_data
    if isKey(property_map, data.Property_Type{i})
        property_encoded(i) = property_map(data.Property_Type{i});
    end
end

old_new_encoded = strcmp(data.Old_New, 'Y');
duration_encoded = strcmp(data.Duration, 'L');
ppd_encoded = strcmp(data.PPD_Category_Type, 'B');

month_map = containers.Map({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec'}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
month_encoded = zeros(n_data, 1);
for i = 1:n_data
    if isKey(month_map, data.Month{i})
        month_encoded(i) = month_map(data.Month{i});
    end
end

% this is to create correlation matrix
feature_matrix = [log_price, property_encoded, old_new_encoded, duration_encoded, ...
    data.Year, month_encoded, ppd_encoded];

corr_feature_names = {'log(Price)', 'Property Type', 'Old/New', 'Duration', ...
    'Year', 'Month', 'PPD Category'};

corr_matrix = corrcoef(feature_matrix);

% Figure: correlation heatmap
fig_corr = figure('Position', [100, 100, 900, 750]);
h = heatmap(corr_feature_names, corr_feature_names, corr_matrix);
h.Title = 'Correlation Matrix: Features vs Target (log Price)';
h.XLabel = 'Features';
h.YLabel = 'Features';
h.ColorLimits = [-1, 1];
h.CellLabelFormat = '%.2f';
h.FontSize = 10;

%% this is for time series analysis

month_order = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', ...
    'Oct', 'Nov', 'Dec'};
month_num_map = containers.Map(month_order, 1:12);

month_numeric = zeros(n_data, 1);
for i = 1:n_data
    month_numeric(i) = month_num_map(data.Month{i});
end

year_month = data.Year * 100 + month_numeric;

% Figure: times series analysis
unique_ym = unique(year_month);
n_periods = length(unique_ym);

monthly_mean = zeros(n_periods, 1);
monthly_median = zeros(n_periods, 1);
monthly_count = zeros(n_periods, 1);

for i = 1:n_periods
    mask = year_month == unique_ym(i);
    prices_period = data.Price(mask);
    monthly_mean(i) = mean(prices_period);
    monthly_median(i) = median(prices_period);
    monthly_count(i) = sum(mask);
end

years_ts = floor(unique_ym / 100);
months_ts = mod(unique_ym, 100);
dates_ts = datetime(years_ts, months_ts, 15);

fig_ts = figure('Position', [50, 50, 1400, 900]);

% subplot 1: Price over time
subplot(2, 2, 1);
yyaxis left;
plot(dates_ts, monthly_mean / 1000, 'b-', 'LineWidth', 1.5);
hold on;
plot(dates_ts, monthly_median / 1000, 'r--', 'LineWidth', 1.5);
ylabel('Price (£ thousands)', 'FontSize', 11);
yyaxis right;
plot(dates_ts, monthly_count, 'g:', 'LineWidth', 1.5);
ylabel('Number of Transactions', 'FontSize', 11);
xlabel('Date', 'FontSize', 11);
title('UK Housing: Price Trends Over Time (2013-2023)', 'FontSize', 13);
legend('Mean Price', 'Median Price', 'Transaction Count', 'Location', 'northwest');
grid on;

% subplot 2: Year-over-Year change
subplot(2, 2, 2);
yearly_stats = grpstats(data, 'Year', {'mean', 'median', 'numel'}, 'DataVars', 'Price');
yoy_change = diff(yearly_stats.mean_Price) ./ yearly_stats.mean_Price(1:end-1) * 100;
years_yoy = yearly_stats.Year(2:end);

bar(years_yoy, yoy_change, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'black');
hold on;
yline(0, 'k--', 'LineWidth', 1);
for i = 1:length(yoy_change)
    if yoy_change(i) < 0
        bar(years_yoy(i), yoy_change(i), 'FaceColor', [0.9, 0.3, 0.3], 'EdgeColor', 'black');
    end
end
xlabel('Year', 'FontSize', 11);
ylabel('Year-over-Year Change (%)', 'FontSize', 11);
title('Annual Price Growth Rate', 'FontSize', 13);
grid on;

% subplot 3: Price by property type over time
subplot(2, 2, 3);
property_types = {'D', 'S', 'T', 'F'};
property_labels = {'Detached', 'Semi-detached', 'Terraced', 'Flat'};
colors_prop = [0.2, 0.4, 0.8; 0.8, 0.4, 0.2; 0.2, 0.7, 0.3; 0.7, 0.2, 0.7];

hold on;
for p = 1:length(property_types)
    mask_prop = strcmp(data.Property_Type, property_types{p});
    prop_data = data(mask_prop, :);
    prop_yearly = grpstats(prop_data, 'Year', 'mean', 'DataVars', 'Price');
    plot(prop_yearly.Year, prop_yearly.mean_Price / 1000, '-o', 'LineWidth', ...
        2, 'Color', colors_prop(p, :), 'MarkerSize', 6, 'DisplayName', property_labels{p});
end
hold off;
xlabel('Year', 'FontSize', 11);
ylabel('Mean Price (£ thousands)', 'FontSize', 11);
title('Price Trends by Property Type', 'FontSize', 13);
legend('Location', 'northwest');
grid on;

% subplot 4: Seasonal patterns
subplot(2, 2, 4);
monthly_seasonal = zeros(12, 2);
for m = 1:12
    mask_month = month_numeric == m;
    monthly_seasonal(m, 1) = mean(data.Price(mask_month));
    monthly_seasonal(m, 2) = sum(mask_month);
end

yyaxis left;
bar(1:12, monthly_seasonal(:, 1) / 1000, 'FaceColor', [0.4, 0.6, 0.8], 'EdgeColor', 'white');
ylabel('Mean Price (£ thousands)', 'FontSize', 11);
yyaxis right;
plot(1:12, monthly_seasonal(:, 2), 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('Transaction Count', 'FontSize', 11);
xlabel('Month', 'FontSize', 11);
title('Seasonal Patterns in UK Housing Market', 'FontSize', 13);
xticks(1:12);
xticklabels(month_order);
xtickangle(45);
grid on;

sgtitle('Time Series Analysis: UK Housing Market (2013-2023)', 'FontSize', ...
    15, 'FontWeight', 'bold');

%% this is for categorical analysis

% Figure: categorical analysis
fig_cat = figure('Position', [50, 50, 1400, 900]);

% subplot 1: Property Type Distribution

subplot(2, 3, 1);
[counts_prop, types_prop] = groupcounts(data.Property_Type);
[counts_sorted, sort_idx] = sort(counts_prop, 'descend');
types_sorted = types_prop(sort_idx);
pie_labels = cell(length(types_sorted), 1);
for i = 1:length(types_sorted)
    pie_labels{i} = sprintf('%s (%.1f%%)', types_sorted{i}, 100*counts_sorted(i)/sum(counts_prop));
end
pie(counts_sorted, pie_labels);
title('Property Type Distribution', 'FontSize', 12);


% subplot 2: Old vs New
subplot(2, 3, 2);
[counts_on, ~] = groupcounts(data.Old_New);
bar(1:2, counts_on, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTickLabel', {'Established (Y)', 'New Build (N)'});
ylabel('Count', 'FontSize', 11);
title('Property Age Distribution', 'FontSize', 12);
grid on;

% subplot 3: Tenure Type
subplot(2, 3, 3);
[counts_dur, ~] = groupcounts(data.Duration);
bar(1:2, counts_dur, 'FaceColor', [0.3, 0.7, 0.4]);
set(gca, 'XTickLabel', {'Freehold (F)', 'Leasehold (L)'});
ylabel('Count', 'FontSize', 11);
title('Tenure Type Distribution', 'FontSize', 12);
grid on;

% subplot 4: Mean Price by Property Type

subplot(2, 3, 4);
prop_stats = grpstats(data, 'Property_Type', {'mean', 'std', 'numel'}, 'DataVars', 'Price');
prop_stats = sortrows(prop_stats, 'mean_Price', 'descend');
x_prop = 1:height(prop_stats);
bar(x_prop, prop_stats.mean_Price / 1000, 'FaceColor', [0.4, 0.6, 0.8]);
hold on;
errorbar(x_prop, prop_stats.mean_Price / 1000, prop_stats.std_Price / 1000 / 2, 'k.', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', prop_stats.Property_Type);
ylabel('Mean Price (£ thousands)', 'FontSize', 11);
title('Mean Price by Property Type (± 0.5 SD)', 'FontSize', 12);
grid on;

% subplot 5: Box plot by Year
subplot(2, 3, 5);
boxplot(log_price, data.Year);
xlabel('Year', 'FontSize', 11);
ylabel('log(1 + Price)', 'FontSize', 11);
title('Price Distribution by Year', 'FontSize', 12);
xtickangle(45);

% subplot 6: Top Counties
subplot(2, 3, 6);
county_stats = grpstats(data, 'County', {'mean', 'numel'}, 'DataVars', 'Price');
county_stats = sortrows(county_stats, 'numel_Price', 'descend');
top10_counties = county_stats(1:min(10, height(county_stats)), :);
barh(1:height(top10_counties), top10_counties.numel_Price, 'FaceColor', [0.6, 0.4, 0.7]);
set(gca, 'YTickLabel', top10_counties.County);
xlabel('Number of Transactions', 'FontSize', 11);
title('Top 10 Counties by Transaction Volume', 'FontSize', 12);
set(gca, 'YDir', 'reverse');
grid on;


sgtitle('Categorical Variable Analysis: UK Housing Data', 'FontSize', 15, 'FontWeight', 'bold');

%% this is train-test split to prevent data leakage before preprocessing

n = height(data);

% this is to hold out 10% of data for final testing
test_ratio = 0.10;
idx = randperm(n);
n_test = round(test_ratio * n);

test_idx = idx(1:n_test);
trainval_idx = idx(n_test+1:end);

% splitting raw data
data_test = data(test_idx, :);
data_trainval = data(trainval_idx, :);

fprintf('Training + Validation set: %d samples (%.1f%%)\n', length(trainval_idx), 100*(1-test_ratio));
fprintf('Testing set: %d samples (%.1f%%)\n', length(test_idx), 100*test_ratio);

%% this is for feature engineering

% this is to create structure to store encoded parameters
preprocessing_info = struct();
preprocessing_info.target_transform = 'log1p';

n_train = height(data_trainval);

% this is to fix the price distribution and encode target variable
y_trainval = log1p(data_trainval.Price);

% one-hot encoding for property types
property_types_enc = {'D', 'S', 'T', 'F', 'O'};
n_property = length(property_types_enc) - 1;
X_property_train = zeros(n_train, n_property);
for i = 1:n_property
    X_property_train(:, i) = strcmp(data_trainval.Property_Type, property_types_enc{i});
end
property_names = strcat('PropertyType_', property_types_enc(1:n_property));

% binary encodings for features with two categories
X_oldnew_train = double(strcmp(data_trainval.Old_New, 'Y'));
X_duration_train = double(strcmp(data_trainval.Duration, 'L'));
X_ppd_train = double(strcmp(data_trainval.PPD_Category_Type, 'B'));

X_year_train = data_trainval.Year;

% cyclical encoding for month to show seasonal trend
month_map_enc = containers.Map(...
    {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'}, ...
    1:12);
month_numeric_train = zeros(n_train, 1);
for i = 1:n_train
    month_numeric_train(i) = month_map_enc(data_trainval.Month{i});
end
X_month_sin_train = sin(2 * pi * month_numeric_train / 12);
X_month_cos_train = cos(2 * pi * month_numeric_train / 12);

% target encoding for high-cardinality location features
global_mean = mean(y_trainval);
smoothing_factor = 10;
preprocessing_info.global_mean_target = global_mean;
preprocessing_info.smoothing_factor = smoothing_factor;

% county encoding
unique_counties_train = unique(data_trainval.County);
county_encoding_map = containers.Map();
for i = 1:length(unique_counties_train)
    cat = unique_counties_train{i};
    mask = strcmp(data_trainval.County, cat);
    n_samples = sum(mask);
    cat_mean = mean(y_trainval(mask));
    smoothed_value = (n_samples * cat_mean + smoothing_factor * global_mean) / (n_samples + smoothing_factor);
    county_encoding_map(cat) = smoothed_value;
end
X_county_train = zeros(n_train, 1);
for i = 1:n_train
    X_county_train(i) = county_encoding_map(data_trainval.County{i});
end
preprocessing_info.county_encoding = county_encoding_map;

% district encoding
unique_districts_train = unique(data_trainval.District);
district_encoding_map = containers.Map();
for i = 1:length(unique_districts_train)
    cat = unique_districts_train{i};
    mask = strcmp(data_trainval.District, cat);
    n_samples = sum(mask);
    cat_mean = mean(y_trainval(mask));
    smoothed_value = (n_samples * cat_mean + smoothing_factor * global_mean) / (n_samples + smoothing_factor);
    district_encoding_map(cat) = smoothed_value;
end
X_district_train = zeros(n_train, 1);
for i = 1:n_train
    X_district_train(i) = district_encoding_map(data_trainval.District{i});
end
preprocessing_info.district_encoding = district_encoding_map;

% town encoding
unique_towns_train = unique(data_trainval.Town_City);
town_encoding_map = containers.Map();
for i = 1:length(unique_towns_train)
    cat = unique_towns_train{i};
    mask = strcmp(data_trainval.Town_City, cat);
    n_samples = sum(mask);
    cat_mean = mean(y_trainval(mask));
    smoothed_value = (n_samples * cat_mean + smoothing_factor * global_mean) / (n_samples + smoothing_factor);
    town_encoding_map(cat) = smoothed_value;
end
X_town_train = zeros(n_train, 1);
for i = 1:n_train
    X_town_train(i) = town_encoding_map(data_trainval.Town_City{i});
end
preprocessing_info.town_encoding = town_encoding_map;

% this is to combine all training features

X_trainval = [X_property_train, X_oldnew_train, X_duration_train, X_ppd_train, X_year_train, ...
     X_month_sin_train, X_month_cos_train, X_county_train, X_district_train, X_town_train];

feature_names = [property_names, {'IsOld'}, {'IsLeasehold'}, {'PPD_Type_B'}, ...
                 {'Year'}, {'Month_Sin'}, {'Month_Cos'}, {'County_Encoded'}, ...
                 {'District_Encoded'}, {'Town_Encoded'}];

preprocessing_info.feature_names = feature_names;
preprocessing_info.property_types = property_types_enc;

% this is to encode test data with training encoding maps
n_test = height(data_test);
y_test = log1p(data_test.Price);

X_property_test = zeros(n_test, n_property);
for i = 1:n_property
    X_property_test(:, i) = strcmp(data_test.Property_Type, property_types_enc{i});
end

X_oldnew_test = double(strcmp(data_test.Old_New, 'Y'));
X_duration_test = double(strcmp(data_test.Duration, 'L'));
X_ppd_test = double(strcmp(data_test.PPD_Category_Type, 'B'));

X_year_test = data_test.Year;

month_numeric_test = zeros(n_test, 1);
for i = 1:n_test
    month_numeric_test(i) = month_map_enc(data_test.Month{i});
end
X_month_sin_test = sin(2 * pi * month_numeric_test / 12);
X_month_cos_test = cos(2 * pi * month_numeric_test / 12);

% this is to apply training encoding maps to test data
X_county_test = zeros(n_test, 1);
for i = 1:n_test
    cat = data_test.County{i};
    if isKey(county_encoding_map, cat)
        X_county_test(i) = county_encoding_map(cat);
    else
        X_county_test(i) = global_mean; % we're using global mean for unseen categories
    end
end

X_district_test = zeros(n_test, 1);
for i = 1:n_test
    cat = data_test.District{i};
    if isKey(district_encoding_map, cat)
        X_district_test(i) = district_encoding_map(cat);
    else
        X_district_test(i) = global_mean;
    end
end

X_town_test = zeros(n_test, 1);
for i = 1:n_test
    cat = data_test.Town_City{i};
    if isKey(town_encoding_map, cat)
        X_town_test(i) = town_encoding_map(cat);
    else
        X_town_test(i) = global_mean;
    end
end

% this is to combine all testing features

X_test = [X_property_test, X_oldnew_test, X_duration_test, X_ppd_test, X_year_test, ...
     X_month_sin_test, X_month_cos_test, X_county_test, X_district_test, X_town_test];

fprintf('Features after preprocessing: %d\n', size(X_trainval, 2));
fprintf('\nFeature names:\n');
disp(feature_names');

%% Ridge Regression training

% this is to define lambda grid (more granular around typical optimal values)
lambda_grid = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000];
k_folds = 10;

fprintf('Training Ridge Regression...\n');
fprintf('Lambda grid: [%s]\n', sprintf('%.4f ', lambda_grid));
fprintf('CV folds: %d\n', k_folds);

% preallocating matrices
n_trainval = size(X_trainval, 1);
p = size(X_trainval, 2);
n_lambdas = length(lambda_grid);

ridge_cv_rmse = zeros(n_lambdas, k_folds);
ridge_cv_mae = zeros(n_lambdas, k_folds);
ridge_cv_r2 = zeros(n_lambdas, k_folds);

% this is to create random assignment of each sample to the folds
cv_indices = crossvalind('Kfold', n_trainval, k_folds);

fprintf('Cross-validation progress: ');

% this is to train on 9 folds and validate on 1 fold
for fold = 1:k_folds
    fprintf('%d ', fold);
    
    test_mask_cv = (cv_indices == fold);
    train_mask_cv = ~test_mask_cv;
    
    X_train_cv = X_trainval(train_mask_cv, :);
    y_train_cv = y_trainval(train_mask_cv);
    X_val_cv = X_trainval(test_mask_cv, :);
    y_val_cv = y_trainval(test_mask_cv);
    
    % this is to standardize for normal distribution for feature equality
    mu_train_cv = mean(X_train_cv, 1);
    sigma_train_cv = std(X_train_cv, 0, 1);
    sigma_train_cv(sigma_train_cv == 0) = 1;
    
    X_train_scaled = (X_train_cv - mu_train_cv) ./ sigma_train_cv;
    X_val_scaled = (X_val_cv - mu_train_cv) ./ sigma_train_cv;
    
    X_train_aug = [ones(size(X_train_scaled, 1), 1), X_train_scaled];
    X_val_aug = [ones(size(X_val_scaled, 1), 1), X_val_scaled];
    
    for l = 1:n_lambdas
        lambda = lambda_grid(l);
        I_reg = eye(p + 1);
        I_reg(1, 1) = 0;  % doesn't regularize intercept to avoid bias
        
        w = (X_train_aug' * X_train_aug + lambda * I_reg) \ (X_train_aug' * y_train_cv);
        y_pred_cv = X_val_aug * w;
        
        ridge_cv_rmse(l, fold) = sqrt(mean((y_val_cv - y_pred_cv).^2));
        ridge_cv_mae(l, fold) = mean(abs(y_val_cv - y_pred_cv));
        ss_res = sum((y_val_cv - y_pred_cv).^2);
        ss_tot = sum((y_val_cv - mean(y_val_cv)).^2);
        ridge_cv_r2(l, fold) = 1 - ss_res / ss_tot;
    end
end
fprintf('Done.\n');

ridge_mean_cv_rmse = mean(ridge_cv_rmse, 2);
ridge_std_cv_rmse = std(ridge_cv_rmse, 0, 2);
ridge_mean_cv_mae = mean(ridge_cv_mae, 2);
ridge_mean_cv_r2 = mean(ridge_cv_r2, 2);

[ridge_min_rmse, ridge_best_idx] = min(ridge_mean_cv_rmse);
ridge_best_lambda = lambda_grid(ridge_best_idx);

fprintf('Best lambda: %.4f (CV RMSE: %.4f ± %.4f)\n', ridge_best_lambda, ridge_min_rmse, ridge_std_cv_rmse(ridge_best_idx));

% this is to store Ridge results
ridge_results = struct();
ridge_results.lambda_grid = lambda_grid;
ridge_results.cv_rmse = ridge_cv_rmse;
ridge_results.cv_mae = ridge_cv_mae;
ridge_results.cv_r2 = ridge_cv_r2;
ridge_results.mean_cv_rmse = ridge_mean_cv_rmse;
ridge_results.std_cv_rmse = ridge_std_cv_rmse;
ridge_results.mean_cv_mae = ridge_mean_cv_mae;
ridge_results.mean_cv_r2 = ridge_mean_cv_r2;
ridge_results.best_lambda = ridge_best_lambda;
ridge_results.best_cv_rmse = ridge_min_rmse;
ridge_results.best_cv_rmse_std = ridge_std_cv_rmse(ridge_best_idx);
ridge_results.best_cv_r2 = mean(ridge_cv_r2(ridge_best_idx, :));
ridge_results.k_folds = k_folds;
ridge_results.fold_rmse = ridge_cv_rmse(ridge_best_idx, :);

% this is to train final Ridge model on all training data

mu_all = mean(X_trainval, 1);
sigma_all = std(X_trainval, 0, 1);
sigma_all(sigma_all == 0) = 1;

X_trainval_scaled = (X_trainval - mu_all) ./ sigma_all;
X_trainval_aug = [ones(n_trainval, 1), X_trainval_scaled];

I_reg = eye(p + 1);
I_reg(1, 1) = 0;

w_ridge_final = (X_trainval_aug' * X_trainval_aug + ridge_best_lambda * I_reg) \ (X_trainval_aug' * y_trainval);

y_train_pred_ridge = X_trainval_aug * w_ridge_final;
ridge_train_rmse = sqrt(mean((y_trainval - y_train_pred_ridge).^2));
ridge_train_r2 = 1 - sum((y_trainval - y_train_pred_ridge).^2) / sum((y_trainval - mean(y_trainval)).^2);

fprintf('Training RMSE: %.4f, R²: %.4f\n', ridge_train_rmse, ridge_train_r2);

% this is to store Ridge Regression model with standardization parameters
ridge_model = struct();
ridge_model.weights = w_ridge_final;
ridge_model.intercept = w_ridge_final(1);
ridge_model.coefficients = w_ridge_final(2:end);
ridge_model.lambda = ridge_best_lambda;
ridge_model.mu = mu_all;
ridge_model.sigma = sigma_all;
ridge_model.feature_names = feature_names;

ridge_results.train_rmse = ridge_train_rmse;
ridge_results.train_r2 = ridge_train_r2;

% updating preprocessing_info with the same standardization parameters
preprocessing_info.mu = mu_all;
preprocessing_info.sigma = sigma_all;

% Figure: Ridge Regression hyperparameter tuning
fig_ridge = figure('Position', [100, 100, 1000, 500]);

subplot(1, 2, 1);
errorbar(1:length(lambda_grid), ridge_mean_cv_rmse, ridge_std_cv_rmse, '-o', ...
    'LineWidth', 2, 'MarkerSize', 8, 'Color', [0.2, 0.4, 0.8], 'CapSize', 8);
hold on;
plot(ridge_best_idx, ridge_mean_cv_rmse(ridge_best_idx), 'rp', 'MarkerSize', 20, 'MarkerFaceColor', 'red');
hold off;
set(gca, 'XTick', 1:length(lambda_grid));
set(gca, 'XTickLabel', arrayfun(@num2str, lambda_grid, 'UniformOutput', false));
xtickangle(45);
xlabel('Regularization Parameter (λ)', 'FontSize', 12);
ylabel('Cross-Validation RMSE', 'FontSize', 12);
title(sprintf('Ridge Regression: λ Tuning (%d-fold CV)', k_folds), 'FontSize', 14);
grid on;

subplot(1, 2, 2);
plot(1:length(lambda_grid), ridge_mean_cv_r2, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0.8, 0.4, 0.2]);
hold on;
plot(ridge_best_idx, ridge_mean_cv_r2(ridge_best_idx), 'rp', 'MarkerSize', 20, 'MarkerFaceColor', 'red');
hold off;
set(gca, 'XTick', 1:length(lambda_grid));
set(gca, 'XTickLabel', arrayfun(@num2str, lambda_grid, 'UniformOutput', false));
xtickangle(45);
xlabel('Regularization Parameter (λ)', 'FontSize', 12);
ylabel('Cross-Validation R²', 'FontSize', 12);
title('Ridge Regression: R² vs λ', 'FontSize', 14);
grid on;

sgtitle('Ridge Regression Hyperparameter Tuning', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('Best lambda: %.4f\n', ridge_best_lambda);
fprintf('Best CV RMSE: %.4f\n', ridge_results.best_cv_rmse);

fprintf('Ridge Regression training complete.\n\n');

%% Random Forest Training

% this is to define hyperparameter grid
rf_NumTrees = [50, 100, 200];
rf_MinLeafSize = [1, 5, 10, 20];
rf_NumPredictorsToSample = [round(p/3), round(sqrt(p)), p];

fprintf('Training Random Forest...\n');
fprintf('NumTrees grid: [%s]\n', sprintf('%d ', rf_NumTrees));
fprintf('MinLeafSize grid: [%s]\n', sprintf('%d ', rf_MinLeafSize));
fprintf('CV folds: %d\n', k_folds);

[grid_trees, grid_leaf, grid_pred] = ndgrid(rf_NumTrees, rf_MinLeafSize, rf_NumPredictorsToSample);

n_combinations = numel(grid_trees);
trees_flat = grid_trees(:);
leaf_flat = grid_leaf(:);
pred_flat = grid_pred(:);

fprintf('Total combinations: %d\n', n_combinations);

rf_cv_rmse = zeros(n_combinations, k_folds);
rf_cv_mae = zeros(n_combinations, k_folds);
rf_cv_r2 = zeros(n_combinations, k_folds);

total_start = tic;
fprintf('Grid search progress: ');

for combo = 1:n_combinations
    if mod(combo, 5) == 0 || combo == 1
        fprintf('%d/%d ', combo, n_combinations);
    end
    
    num_trees = trees_flat(combo);
    min_leaf = leaf_flat(combo);
    num_pred = pred_flat(combo);
    
    for fold = 1:k_folds
        test_mask_cv = (cv_indices == fold);
        train_mask_cv = ~test_mask_cv;
        
        X_train_cv = X_trainval(train_mask_cv, :);
        y_train_cv = y_trainval(train_mask_cv);
        X_val_cv = X_trainval(test_mask_cv, :);
        y_val_cv = y_trainval(test_mask_cv);
        
        rf_temp = TreeBagger(num_trees, X_train_cv, y_train_cv, 'Method', 'regression', ...
            'MinLeafSize', min_leaf, 'NumPredictorsToSample', num_pred, ...
            'OOBPrediction', 'off');
        
        y_pred_cv = predict(rf_temp, X_val_cv);
        
        rf_cv_rmse(combo, fold) = sqrt(mean((y_val_cv - y_pred_cv).^2));
        rf_cv_mae(combo, fold) = mean(abs(y_val_cv - y_pred_cv));
        ss_res = sum((y_val_cv - y_pred_cv).^2);
        ss_tot = sum((y_val_cv - mean(y_val_cv)).^2);
        rf_cv_r2(combo, fold) = 1 - ss_res / ss_tot;
    end
end

grid_search_time = toc(total_start);
fprintf('\n  Grid search completed in %.1f seconds.\n', grid_search_time);

rf_mean_cv_rmse = mean(rf_cv_rmse, 2);
rf_std_cv_rmse = std(rf_cv_rmse, 0, 2);
rf_mean_cv_mae = mean(rf_cv_mae, 2);
rf_mean_cv_r2 = mean(rf_cv_r2, 2);

[rf_min_rmse, rf_best_idx] = min(rf_mean_cv_rmse);

rf_best_params = struct();
rf_best_params.NumTrees = trees_flat(rf_best_idx);
rf_best_params.MinLeafSize = leaf_flat(rf_best_idx);
rf_best_params.NumPredictorsToSample = pred_flat(rf_best_idx);

fprintf('Best: Trees=%d, MinLeaf=%d, NumPred=%d\n', ...
    rf_best_params.NumTrees, rf_best_params.MinLeafSize, rf_best_params.NumPredictorsToSample);
fprintf('Best CV RMSE: %.4f ± %.4f\n', rf_min_rmse, rf_std_cv_rmse(rf_best_idx));

% this is to store Random Forest results

rf_results = struct();
rf_results.param_grid.NumTrees = rf_NumTrees;
rf_results.param_grid.MinLeafSize = rf_MinLeafSize;
rf_results.param_grid.NumPredictorsToSample = rf_NumPredictorsToSample;
rf_results.all_combinations = table(trees_flat, leaf_flat, pred_flat, rf_mean_cv_rmse, rf_std_cv_rmse, ...
    'VariableNames', {'NumTrees', 'MinLeafSize', 'NumPredictorsToSample', 'MeanRMSE', 'StdRMSE'});
rf_results.cv_rmse = rf_cv_rmse;
rf_results.cv_mae = rf_cv_mae;
rf_results.cv_r2 = rf_cv_r2;
rf_results.mean_cv_rmse = rf_mean_cv_rmse;
rf_results.std_cv_rmse = rf_std_cv_rmse;
rf_results.mean_cv_mae = rf_mean_cv_mae;
rf_results.mean_cv_r2 = rf_mean_cv_r2;
rf_results.best_params = rf_best_params;
rf_results.best_cv_rmse = rf_min_rmse;
rf_results.best_cv_rmse_std = rf_std_cv_rmse(rf_best_idx);
rf_results.best_cv_r2 = mean(rf_cv_r2(rf_best_idx, :));
rf_results.k_folds = k_folds;
rf_results.fold_rmse = rf_cv_rmse(rf_best_idx, :);
rf_results.grid_search_time = grid_search_time;

% this is to train the final RF model on all training data

train_start = tic;
rf_model = TreeBagger(rf_best_params.NumTrees, X_trainval, y_trainval, ...
    'Method', 'regression', 'MinLeafSize', rf_best_params.MinLeafSize, ...
    'NumPredictorsToSample', rf_best_params.NumPredictorsToSample, ...
    'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');
train_time = toc(train_start);

fprintf('Final model trained in %.1f seconds.\n', train_time);

y_train_pred_rf = predict(rf_model, X_trainval);
rf_train_rmse = sqrt(mean((y_trainval - y_train_pred_rf).^2));
rf_train_r2 = 1 - sum((y_trainval - y_train_pred_rf).^2) / sum((y_trainval - mean(y_trainval)).^2);

fprintf('Training RMSE: %.4f, R²: %.4f\n', rf_train_rmse, rf_train_r2);

rf_results.train_rmse = rf_train_rmse;
rf_results.train_r2 = rf_train_r2;
rf_results.feature_importance = rf_model.OOBPermutedPredictorDeltaError;
rf_results.feature_names = feature_names;

fprintf('Random Forest training complete.\n\n');

%% this is for model comparison

comparison_metrics = {'CV RMSE (mean)', 'CV RMSE (std)', 'CV R² (mean)', 'Training RMSE', 'Training R²'};

ridge_values = [ridge_results.best_cv_rmse; ridge_results.best_cv_rmse_std; ...
                ridge_results.best_cv_r2; ridge_results.train_rmse; ridge_results.train_r2];

rf_values = [rf_results.best_cv_rmse; rf_results.best_cv_rmse_std; ...
             rf_results.best_cv_r2; rf_results.train_rmse; rf_results.train_r2];

comparison_table = table(comparison_metrics', ridge_values, rf_values, ...
    'VariableNames', {'Metric', 'Ridge_Regression', 'Random_Forest'});

disp('Cross-Validation Results Comparison:');
disp(comparison_table);

% Figure: Model Comparison: Ridge vs Random Forest

fig_comparison = figure('Position', [100, 100, 900, 500]);

subplot(1, 2, 1);
ridge_fold_rmse = ridge_results.fold_rmse(:);
rf_fold_rmse = rf_results.fold_rmse(:);

box_data = [ridge_fold_rmse; rf_fold_rmse];
box_groups = [repmat({'Ridge'}, length(ridge_fold_rmse), 1); ...
              repmat({'Random Forest'}, length(rf_fold_rmse), 1)];

boxplot(box_data, box_groups, 'Colors', [0.2, 0.4, 0.8; 0.8, 0.4, 0.2], 'Widths', 0.6);
ylabel('RMSE (log-price)', 'FontSize', 12);
title(sprintf('CV RMSE Comparison (%d Folds)', k_folds), 'FontSize', 13);
grid on;

subplot(1, 2, 2);
metrics_names = {'RMSE', 'MAE', 'R²'};
ridge_metrics = [ridge_results.best_cv_rmse, mean(ridge_results.mean_cv_mae), ridge_results.best_cv_r2];
rf_metrics = [rf_results.best_cv_rmse, mean(rf_results.mean_cv_mae), rf_results.best_cv_r2];

x_bar = 1:3;
width_bar = 0.35;
bar(x_bar - width_bar/2, ridge_metrics, width_bar, 'FaceColor', [0.2, 0.4, 0.8]);
hold on;
bar(x_bar + width_bar/2, rf_metrics, width_bar, 'FaceColor', [0.8, 0.4, 0.2]);
hold off;

set(gca, 'XTick', x_bar, 'XTickLabel', metrics_names);
ylabel('Metric Value', 'FontSize', 12);
title('CV Metrics Comparison', 'FontSize', 13);
legend('Ridge', 'Random Forest', 'Location', 'best');
grid on;

sgtitle('Model Comparison: Ridge vs Random Forest', 'FontSize', 15, 'FontWeight', 'bold');

%% this is for final evaluation on test set

% this is to standardize test set
X_test_scaled = (X_test - ridge_model.mu) ./ ridge_model.sigma;


% this is the ridge predictions on test set
tic;
X_test_aug = [ones(size(X_test_scaled, 1), 1), X_test_scaled];
y_pred_ridge = X_test_aug * ridge_model.weights;
ridge_predict_time = toc;


% this is the Random Forest predictions on test set
tic;
y_pred_rf = predict(rf_model, X_test);
if iscell(y_pred_rf)
    y_pred_rf = cell2mat(y_pred_rf);
end
rf_predict_time = toc;

% this is to calculate test metrics for Ridge

ridge_residuals = y_test - y_pred_ridge;
ridge_test_rmse = sqrt(mean(ridge_residuals.^2));
ridge_test_mae = mean(abs(ridge_residuals));
ridge_test_r2 = 1 - sum(ridge_residuals.^2) / sum((y_test - mean(y_test)).^2);

ridge_test_metrics = struct();
ridge_test_metrics.rmse = ridge_test_rmse;
ridge_test_metrics.mae = ridge_test_mae;
ridge_test_metrics.r2 = ridge_test_r2;

fprintf('Ridge Regression Test Metrics:\n');
fprintf('RMSE: %.4f, MAE: %.4f, R²: %.4f\n', ridge_test_rmse, ridge_test_mae, ridge_test_r2);


% this is to calculate test metrics for Random Forest
rf_residuals = y_test - y_pred_rf;
rf_test_rmse = sqrt(mean(rf_residuals.^2));
rf_test_mae = mean(abs(rf_residuals));
rf_test_r2 = 1 - sum(rf_residuals.^2) / sum((y_test - mean(y_test)).^2);

rf_test_metrics = struct();
rf_test_metrics.rmse = rf_test_rmse;
rf_test_metrics.mae = rf_test_mae;
rf_test_metrics.r2 = rf_test_r2;

fprintf('\nRandom Forest Test Metrics:\n');
fprintf('RMSE: %.4f, MAE: %.4f, R²: %.4f\n', rf_test_rmse, rf_test_mae, rf_test_r2);

% this is to display final results

fprintf('\nFinal Test Set Results\n');
fprintf('%-25s %-15s %-15s\n', 'Metric', 'Ridge', 'Random Forest');
fprintf('%-25s %-15.4f %-15.4f\n', 'RMSE (log-price)', ridge_test_metrics.rmse, rf_test_metrics.rmse);
fprintf('%-25s %-15.4f %-15.4f\n', 'MAE (log-price)', ridge_test_metrics.mae, rf_test_metrics.mae);
fprintf('%-25s %-15.4f %-15.4f\n', 'R-squared', ridge_test_metrics.r2, rf_test_metrics.r2);
fprintf('%-25s %-15.4f %-15.4f\n', 'Prediction Time (s)', ridge_predict_time, rf_predict_time);

% converting back to original price scale
y_test_original = exp(y_test) - 1;
y_pred_ridge_original = exp(y_pred_ridge) - 1;
y_pred_rf_original = exp(y_pred_rf) - 1;

rmse_ridge_original = sqrt(mean((y_test_original - y_pred_ridge_original).^2));
rmse_rf_original = sqrt(mean((y_test_original - y_pred_rf_original).^2));

fprintf('%-25s £%-14.2f £%-14.2f\n', 'RMSE (original £)', rmse_ridge_original, rmse_rf_original);

% this is the final results table

final_results = table({'Ridge Regression'; 'Random Forest'}, [ridge_test_metrics.rmse; rf_test_metrics.rmse], ...
    [ridge_test_metrics.mae; rf_test_metrics.mae], [ridge_test_metrics.r2; rf_test_metrics.r2], ...
    [rmse_ridge_original; rmse_rf_original], [ridge_predict_time; rf_predict_time], 'VariableNames', ...
    {'Model', 'RMSE_LogPrice', 'MAE_LogPrice', 'R_Squared', 'RMSE_GBP', 'PredictTime_s'});

disp(final_results);

%% this is for Residual Analysis

% Figure: Predicted vs Actual values

fig_pred_actual = figure('Position', [50, 50, 1200, 500]);

all_values = [y_test; y_pred_ridge; y_pred_rf];
min_val = min(all_values);
max_val = max(all_values);
padding = 0.05 * (max_val - min_val);

subplot(1, 2, 1);
scatter(y_test, y_pred_ridge, 20, [0.2, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
hold off;
xlim([min_val - padding, max_val + padding]);
ylim([min_val - padding, max_val + padding]);
axis square;
xlabel('Actual log(Price)', 'FontSize', 12);
ylabel('Predicted log(Price)', 'FontSize', 12);
title(sprintf('Ridge (RMSE=%.4f, R²=%.4f)', ridge_test_rmse, ridge_test_r2), 'FontSize', 13);
grid on;

subplot(1, 2, 2);
scatter(y_test, y_pred_rf, 20, [0.8, 0.4, 0.2], 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
hold off;
xlim([min_val - padding, max_val + padding]);
ylim([min_val - padding, max_val + padding]);
axis square;
xlabel('Actual log(Price)', 'FontSize', 12);
ylabel('Predicted log(Price)', 'FontSize', 12);
title(sprintf('Random Forest (RMSE=%.4f, R²=%.4f)', rf_test_rmse, rf_test_r2), 'FontSize', 13);
grid on;

sgtitle('Test Set: Predicted vs Actual Values', 'FontSize', 15, 'FontWeight', 'bold');

% Figure: Residual Analysis
fig_residuals = figure('Position', [50, 50, 1400, 700]);

residuals_ridge = y_test - y_pred_ridge;
residuals_rf = y_test - y_pred_rf;


% Ridge residuals
subplot(2, 3, 1);
scatter(y_pred_ridge, residuals_ridge, 15, [0.2, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.4);
hold on; yline(0, 'r--', 'LineWidth', 2); hold off;
xlabel('Fitted Values', 'FontSize', 11);
ylabel('Residuals', 'FontSize', 11);
title('Ridge: Residuals vs Fitted', 'FontSize', 12);
grid on;

subplot(2, 3, 2);
histogram(residuals_ridge, 50, 'FaceColor', [0.2, 0.4, 0.8], 'Normalization', 'pdf');
hold on;
x_norm = linspace(min(residuals_ridge), max(residuals_ridge), 100);
plot(x_norm, normpdf(x_norm, mean(residuals_ridge), std(residuals_ridge)), 'r-', 'LineWidth', 2);
hold off;
xlabel('Residual', 'FontSize', 11);
ylabel('Density', 'FontSize', 11);
title('Ridge: Residual Distribution', 'FontSize', 12);

subplot(2, 3, 3);
qqplot(residuals_ridge);
title('Ridge: Q-Q Plot', 'FontSize', 12);
grid on;

% RF residuals

subplot(2, 3, 4);
scatter(y_pred_rf, residuals_rf, 15, [0.8, 0.4, 0.2], 'filled', 'MarkerFaceAlpha', 0.4);
hold on; yline(0, 'r--', 'LineWidth', 2); hold off;
xlabel('Fitted Values', 'FontSize', 11);
ylabel('Residuals', 'FontSize', 11);
title('Random Forest: Residuals vs Fitted', 'FontSize', 12);
grid on;

subplot(2, 3, 5);
histogram(residuals_rf, 50, 'FaceColor', [0.8, 0.4, 0.2], 'Normalization', 'pdf');
hold on;
x_norm = linspace(min(residuals_rf), max(residuals_rf), 100);
plot(x_norm, normpdf(x_norm, mean(residuals_rf), std(residuals_rf)), 'r-', 'LineWidth', 2);
hold off;
xlabel('Residual', 'FontSize', 11);
ylabel('Density', 'FontSize', 11);
title('Random Forest: Residual Distribution', 'FontSize', 12);

subplot(2, 3, 6);
qqplot(residuals_rf);
title('Random Forest: Q-Q Plot', 'FontSize', 12);
grid on;

sgtitle('Residual Analysis: Ridge vs Random Forest', 'FontSize', 15, 'FontWeight', 'bold');

%% saving models and results

% this is to save trained models, preprocessing info and test set 

save('models/ridge_model.mat', 'ridge_model', 'ridge_best_lambda', 'ridge_results');
save('models/rf_model.mat', 'rf_model', 'rf_best_params', 'rf_results');
save('models/preprocessing_info.mat', 'preprocessing_info', 'feature_names');
save('models/data_split.mat', 'test_idx', 'trainval_idx', 'X_test', 'y_test', 'X_trainval', 'y_trainval');

% these are the final results
writetable(final_results, 'results/final_results.csv');
writetable(comparison_table, 'results/cv_comparison.csv');

fprintf('All models and results saved.\n');
fprintf('\nUK Housing Price Prediction Complete\n');
