import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from lightgbm import LGBMRegressor
from Catboost import CatBoostRegressor

def process_prediction(pred):
    final_pred = np.zeros_like(pred)  # Initialize final predictions array with zeros

    # Apply comparisons element-wise
    final_pred[pred < -0.5] = -1.0
    final_pred[(pred >= -0.5) & (pred <= 0.5)] = 0.0
    final_pred[pred > 0.5] = 1.0

    return final_pred

# Load the data
train = pd.read_csv('Train.csv').dropna(axis=0)  # Read in train, ignoring one row with missing data
test = pd.read_csv('Test.csv').fillna('')  # Read in test

# Split data into training and validation sets
df_train, df_valid = train_test_split(train, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform text data into TF-IDF features
X_train = vectorizer.fit_transform(df_train['safe_text'])
X_valid = vectorizer.transform(df_valid['safe_text'])
X_test = vectorizer.transform(test['safe_text'])

# Define labels
y_train = df_train['label']
y_valid = df_valid['label']

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=2)

# Initialize lists to store RMSE values and total predictions
errlgb = []
y_pred_totlgb = []

# Initialize lists to store CatBoost and LightGBM predictions
catboost_preds = []
lgb_preds = []

for fold_, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    X_train_fold, X_valid_fold = X_train[train_idx], X_train[valid_idx]
    y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    # Train CatBoost model
    catboost_model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8, eval_metric='RMSE', random_seed=42)
    catboost_model.fit(X_train_fold, y_train_fold)
    
    # Make predictions with CatBoost
    catboost_fold_preds = catboost_model.predict(X_test)
    catboost_preds.append(catboost_fold_preds)
    
    # Train LightGBM model
    lgb_model = LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=2000, objective='regression', metric='rmse', random_state=42)
    lgb_model.fit(X_train_fold, y_train_fold)
    
    # Make predictions with LightGBM
    lgb_fold_preds = lgb_model.predict(X_test)
    lgb_preds.append(lgb_fold_preds)

    # Evaluate CatBoost RMSE
    catboost_rmse = np.sqrt(mean_squared_error(y_valid_fold, catboost_model.predict(X_valid_fold)))
    print(f"Fold {fold_ + 1} CatBoost RMSE:", catboost_rmse)
    
    # Evaluate LightGBM RMSE
    lgb_rmse = np.sqrt(mean_squared_error(y_valid_fold, lgb_model.predict(X_valid_fold)))
    print(f"Fold {fold_ + 1} LightGBM RMSE:", lgb_rmse)
    
# Combine CatBoost and LightGBM predictions using simple averaging
ensemble_preds = (np.mean(catboost_preds, axis=0) + np.mean(lgb_preds, axis=0)) / 2

# Save the predictions to a DataFrame
sub_ensemble = pd.DataFrame({
        'tweet_id': test['tweet_id'],
        'label': ensemble_preds
    })

# Write the DataFrame to a CSV file
sub_ensemble.to_csv(f'ensemble_submission.csv', index=False)
sub_ensemble.head()
