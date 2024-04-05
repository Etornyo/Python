import numpy as np
import pandas as pd
from sklearn.model_selection import KFold #Addition
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


# Addition
kf=KFold(n_splits=20,shuffle=True,random_state=2)

for fold,(train_idx,valid_idx) in enumerate(kf.split(X_train)):
    X_train_fold, X_valid_fold = X_train[train_idx], X_train[valid_idx]
    y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)
    lgbm_classifier.fit(X_train_fold, y_train_fold)
    
    valid_preds_fold = lgbm_classifier.predict(X_valid_fold)
    accuracy_fold = accuracy_score(y_valid_fold, valid_preds_fold)
    print(f"Fold {fold+1} Accuracy (LightGBM):", accuracy_fold)



# Train LightGBM model
lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)
lgbm_classifier.fit(X_train, y_train)

# Predict on the validation set
valid_preds_lgbm = lgbm_classifier.predict(X_valid)

# Calculate accuracy on the validation set
accuracy_lgbm = accuracy_score(y_valid, valid_preds_lgbm)
print("Validation Accuracy (LightGBM):", accuracy_lgbm)

# Make predictions on the test set for LightGBM
test_preds_lgbm = lgbm_classifier.predict(X_test)

# Make a submission dataframe for LightGBM
sub_lgbm = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': test_preds_lgbm
})
sub_lgbm.to_csv('lgbm_submission.csv', index=False)
sub_lgbm.head()

