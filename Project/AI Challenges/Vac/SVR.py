import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR

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
kf = KFold(n_splits=15, shuffle=True, random_state=2)

# Initialize lists to store RMSE values and total predictions
err_svr = []
y_pred_tot_svr = []

for fold_, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    X_train_fold, X_valid_fold = X_train[train_idx], X_train[valid_idx]
    y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    # Train SVR model
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train_fold, y_train_fold)
    
    # Make predictions with SVR
    svr_fold_preds = svr_model.predict(X_test)
    y_pred_tot_svr.append(svr_fold_preds)

    # Evaluate SVR RMSE
    svr_rmse = np.sqrt(mean_squared_error(y_valid_fold, svr_model.predict(X_valid_fold)))
    print(f"Fold {fold_ + 1} SVR RMSE:", svr_rmse)
    err_svr.append(svr_rmse)
    
# Combine SVR predictions
ensemble_preds_svr = np.mean(y_pred_tot_svr, 0)

# Save the predictions to a DataFrame
sub_svr = pd.DataFrame({
        'tweet_id': test['tweet_id'],
        'label': ensemble_preds_svr
    })

# Write the DataFrame to a CSV file
sub_svr.to_csv(f'svr_submission.csv', index=False)
sub_svr.head()
