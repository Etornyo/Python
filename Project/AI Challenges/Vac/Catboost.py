import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

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

# Train CatBoost model
catboost_classifier = CatBoostClassifier(iterations=1000, random_state=42, verbose=100)
catboost_classifier.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)

# Predict on the validation set
valid_preds_catboost = catboost_classifier.predict(X_valid)

# Calculate accuracy on the validation set
accuracy_catboost = accuracy_score(y_valid, valid_preds_catboost)
print("Validation Accuracy (CatBoost):", accuracy_catboost)

# Make predictions on the test set for CatBoost
test_preds_catboost = catboost_classifier.predict(X_test)

# Make a submission dataframe for CatBoost
sub_catboost = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': test_preds_catboost.reshape(-1)
})
sub_catboost.to_csv('catboost_submission.csv', index=False)
sub_catboost.head()
