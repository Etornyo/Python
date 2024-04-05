import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Load the data
train = pd.read_csv('Train.csv').dropna(axis=0) # Read in train, ignoring one row with missing data
test = pd.read_csv('Test.csv').fillna('') # Read in test

# Split data into training and validation sets
df_train, df_valid = train_test_split(train, test_size=0.2, random_state=42)
print(df_valid.shape, df_train.shape)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform text data into TF-IDF features
X_train = vectorizer.fit_transform(df_train['safe_text'])
X_valid = vectorizer.transform(df_valid['safe_text'])
X_test = vectorizer.transform(test['safe_text'])

# Define labels
y_train = df_train['label']
y_valid = df_valid['label']

# Train LightGBM model
lgbm_classifier = LGBMClassifier(n_estimators=500, random_state=42)
lgbm_classifier.fit(X_train, y_train)

# Predict on the validation set
valid_preds_lgbm = lgbm_classifier.predict(X_valid)

# Calculate accuracy on the validation set
accuracy_lgbm = accuracy_score(y_valid, valid_preds_lgbm)
print("Validation Accuracy (LightGBM):", accuracy_lgbm)

# Train XGBoost model
xgb_classifier = XGBClassifier(n_estimators=500, random_state=42)

# Ensure labels are encoded as integers starting from 0
label_mapping = {label: idx for idx, label in enumerate(df_train['label'].unique())}
y_train = df_train['label'].map(label_mapping)
y_valid = df_valid['label'].map(label_mapping)

xgb_classifier.fit(X_train, y_train)

# Predict on the validation set
valid_preds_xgb = xgb_classifier.predict(X_valid)

# Calculate accuracy on the validation set
accuracy_xgb = accuracy_score(y_valid, valid_preds_xgb)
print("Validation Accuracy (XGBoost):", accuracy_xgb)

# Ensemble predictions from LightGBM and XGBoost
ensemble_preds_proba = (lgbm_classifier.predict_proba(X_test) + xgb_classifier.predict_proba(X_test)) / 2
ensemble_preds = np.argmax(ensemble_preds_proba, axis=1)

# Make a submission dataframe for the ensemble
sub_ensemble = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': ensemble_preds
})
sub_ensemble.to_csv('ensemble_submission.csv', index=False)
sub_ensemble.head()
