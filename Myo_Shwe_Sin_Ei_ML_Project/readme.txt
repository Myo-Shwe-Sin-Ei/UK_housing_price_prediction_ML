Author - Myo Shwe Sin Ei
INM431 Machine Learning - ML Project

UK Housing Price Prediction
Ridge Regression vs Random Forest

*** The Random Forest Model was 523MB and couldn't be uploaded. I have added the Models zip file on Google Drive at this link. ***

Google Drive Folder Link to Models Zip File and Video + Transcript

https://drive.google.com/drive/u/0/folders/1kjFccTnob7npqGu22vJh76JlK3De7KFB

Project Overview
This project compares Ridge Regression and Random Forest models for predicting UK housing prices using an enhanced version of public UK housing transaction data from HM Land Registry on Kaggle.

Dataset

https://www.kaggle.com/datasets/burhanimtengwa/uk-housing-cleaned/data?select=property_data_clean.csv

File: 'uk_housing_clean.csv'
Samples: 249,766 property transactions
Features: Property_Type, Old_New, Duration, Town_City, District, County, PPD_Category_Type, Year, Month
Target Variable: Price (log-transformed)

Files
'main.m' - Main training script with EDA, preprocessing, model training and evaluation
'run_test_demo.m' - Demo script for loading trained models and making predictions
'uk_housing_clean.csv' - Dataset
'readme.txt' - Instruction document
'/models' - Folder of trained models, parameters and indices for training, validation and testing sets
'/results' - Folder of cross-validation and test set performance results comparisons

How to Run
Testing models - run_test_demo.m

Run 'run_test_demo.m' in MATLAB
This will:
1. Load the trained models and preprocessing info
2. Make predictions on the test set
2. Display performance metrics
4. Generate visualization figures

Library, directory dependencies and required software versions
1. MATLAB R2020a or later
2. Statistics and Machine Learning Toolbox