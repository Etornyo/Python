import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score

import lightgbm
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from CatBoost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')
np.random.seed(4590)

print('Finished imports...')

train_client = pd.read_csv('client_train.csv')
test_client = pd.read_csv('client_test.csv')
train_invoice = pd.read_csv('invoice_train.csv')
test_invoice = pd.read_csv('invoice_test.csv')
print('Finished data read...')

# Rest of your preprocessing code...

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store predictions and errors
predictions = []
errors = []

# Iterate over folds
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_client)):
    # Extract training and validation data for this fold
    X_train_fold, X_valid_fold = train_client.iloc[train_idx], train.iloc[valid_idx]
    y_train_fold, y_valid_fold = target[train_idx], target[valid_idx]

    # Define and train your model
    model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        subsample=0.9,
        # colsample_bylevel=0.9,
        random_seed=42,
        verbose=50  # Set to 'Verbose' for training logs
    )
    model.fit(X_train_fold, y_train_fold)

    # Make predictions on the validation set
    fold_preds = model.predict(X_valid_fold)
    
    # Evaluate model and store predictions and errors
    rmse = np.sqrt(mean_squared_error(y_valid_fold, fold_preds))
    errors.append(rmse)
    predictions.append(fold_preds)

    # Optionally, you can also make predictions on the test set for each fold and save them for ensembling later

# Calculate average error across folds
mean_error = np.mean(errors)
print("Mean RMSE:", mean_error)

# Optionally, you can also ensemble the predictions from different folds here

# After ensembling (if applicable), you can make predictions on the test set using the ensemble model
# And write the submission file

