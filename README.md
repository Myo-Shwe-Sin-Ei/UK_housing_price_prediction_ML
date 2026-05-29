A Comparison between Ridge Regression and Random Forest
applied to UK Housing Price Prediction

--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------
This project builds and compares two machine learning models, Ridge Regression
and Random Forest, to predict UK housing prices using sale transaction data from
2013-2023. The goal is to evaluate which model generalises better on this
dataset and to understand the trade-offs between an interpretable linear model
and a non-linear ensemble model.

Hypothesis:
    Random Forest will outperform Ridge Regression due to its ability to
    capture non-linear relationships between features.


--------------------------------------------------------------------------------
2. DATASET
--------------------------------------------------------------------------------
- Source : Enhanced version of UK housing transaction data from HM Land Registry
           on Kaggle (original coverage 1995-2023).
- Scope  : Cleaned to 249,766 samples with 10 features, focused on 2013-2023.
- Target : Property sale price.

Exploratory analysis highlighted heavy skew and severe outliers (including
impossible "trillion-pound" prices), a large gap between mean and median price,
dominance of flats/leasehold properties, Greater London accounting for ~8% of
records, and a seasonal peak in March (likely the end of the UK tax year).

https://www.kaggle.com/datasets/burhanimtengwa/uk-housing-cleaned/data?select=property_data_clean.csv

--------------------------------------------------------------------------------
3. PREPROCESSING
--------------------------------------------------------------------------------
- Log transformation of the target price to normalise the skewed distribution.
- One-hot encoding of property type (using "other" as the reference category to
  avoid the dummy variable trap).
- Binary encoding for two-category features (Old/New, Duration, PPD).
- Year encoded as a direct numeric value; month encoded cyclically to capture
  seasonal patterns.
- Target encoding for high-cardinality location features (County, District,
  Town) with a smoothing factor that blends category mean and global mean to
  shrink rare categories toward the mean.
- All encodings are computed on training data only to avoid data leakage.
  The computed encoding maps are stored in a preprocessing MATLAB file and later
  applied to the test data, using the global mean for unseen categories.
- Standardisation of all numeric features to zero mean and unit variance.


--------------------------------------------------------------------------------
4. MODELS
--------------------------------------------------------------------------------
4.1 Ridge Regression
    - Linear model with L2 regularisation (penalty proportional to the squared
      magnitude of coefficients).
    - Fast, interpretable, handles multicollinearity via coefficient shrinkage.
    - Assumes linear relationships; cannot capture complex interactions without
      explicit feature engineering.

4.2 Random Forest
    - Ensemble of decision trees trained with bootstrap aggregating (bagging);
      outputs the mean prediction across all trees.
    - Captures non-linear relationships and feature interactions; robust to
      outliers and noise.
    - Slower, computationally expensive, and less interpretable.


--------------------------------------------------------------------------------
5. METHODOLOGY
--------------------------------------------------------------------------------
- Train/test split : 9:1 ratio -> 224,789 training samples, 24,977 test samples.
- Validation       : 10-fold cross-validation on the training set (9 folds train,
                     1 fold validate) for every hyperparameter combination.
- Hyperparameter tuning : Grid Search with Cross-Validation.
- Evaluation metrics    : RMSE, R-squared, and MAE.


--------------------------------------------------------------------------------
6. HYPERPARAMETERS
--------------------------------------------------------------------------------
Ridge Regression
    - Lambda grid : [1, 5, 10, 50, 100, 500]
    - Best lambda selected by minimum CV RMSE.
    - Best parameter : lambda = 50

Random Forest
    - NumTrees              : [50, 100, 200]
    - MinLeafSize           : [1, 5, 10, 20]
    - NumPredictorsToSample : [p/3, sqrt(p), p]
    - Grid : 36 combinations x 10 folds
    - Best parameters : 200 trees, MinLeaf = 5, NumPred = p/3


--------------------------------------------------------------------------------
7. RESULTS
--------------------------------------------------------------------------------
                         Mean CV RMSE          MAE
    Ridge Regression     0.698 +/- 0.009       GBP 182,828
    Random Forest        0.608 +/- 0.008       GBP 167,947

- On overall evaluation metrics, Random Forest achieved lower CV RMSE with
  slightly lower standard deviation, indicating better accuracy and stability
  across the 10 folds, and a lower MAE.
- Residual analysis: Ridge residuals are approximately normal and centred at
  zero, but the Q-Q plot shows heavy tails (mis-prediction of outliers). Random
  Forest shows a more uniform spread across the prediction range but also has
  heavy tails at both ends.
- Both models perform best on mid-range properties (~GBP 150k-350k) and struggle
  at the extremes, where properties above GBP 1 million or below GBP 100k make up
  less than 10% of the data.
- Random Forest produced erratic individual mispredictions (e.g. index 6 with a
  246% over-prediction) due to insufficient training samples for low-range
  properties.


--------------------------------------------------------------------------------
8. CONCLUSION
--------------------------------------------------------------------------------
Although Random Forest scored better on the aggregate metrics, Ridge Regression
is judged the better model for this specific dataset: its failures are more
predictable, whereas Random Forest's mispredictions are erratic and risky for
housing price estimation. The dominant limitation is the dataset itself, where
the skewed distribution and a lack of distinguishing features prevented either
model from generalising at the price extremes.

Key takeaways:
- A simpler regularised linear model can be more stable than a complex model.
- Balanced samples and high-quality features matter more than model complexity.
- Strong aggregate metrics can mask completely wrong individual predictions.


--------------------------------------------------------------------------------
9. FUTURE WORK
--------------------------------------------------------------------------------
- Split the data between standard and luxury housing to prevent skewing overall
  predictions.
- Add higher-quality features (e.g. distance to public transport, crime rates,
  school proximity) to help Random Forest distinguish property tiers.
- Implement gradient-boosting algorithms to handle outliers and correct the
  errors of previous trees.


--------------------------------------------------------------------------------
10. REFERENCES
--------------------------------------------------------------------------------
- Begum, S. (2023). House Price Prediction by Machine Learning Technique - An
  Empirical Study. doi:10.1007/978-981-99-5354-7_7.
- Bolotov, D. (2025). Six Approaches to Time Series Smoothing. Medium.
- GeeksforGeeks (2020). Feature Encoding Techniques Machine Learning.
- Preethi et al. (2025). Optimizing Polynomial and Regularization Techniques for
  Enhanced Housing Price Prediction Accuracy. SN Computer Science, 6(2).
  doi:10.1007/s42979-024-03578-7.
- Nokeri, T. C. (2021). Data Science Revealed. Apress. doi:10.1007/978-1-4842-6870-4.
- Whieldon, L. and Ashqar, H. I. (2022). Predicting residential property value: a
  comparison of multiple regression techniques. SN Business & Economics, 2(11).
  doi:10.1007/s43546-022-00358-4.

