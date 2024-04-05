import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Load the data
train = pd.read_csv('Train.csv').dropna(axis=0) # Read in train, ignoring one row with missing data
test = pd.read_csv('Test.csv').fillna('') # Read in test

# Split data into training and validation sets
df_valid = train.sample(1000)
df_train = train.loc[~train.tweet_id.isin(df_valid.tweet_id.values)]
print(df_valid.shape, df_train.shape)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=300, random_state=42)

# Create a pipeline combining the vectorizer and classifier
pipeline = Pipeline([('count_vectorizer', vectorizer), ('rf', rf_classifier)])

# Fit the pipeline on the training data
pipeline.fit(df_train['safe_text'], df_train['label'])

# Predict on the validation set
valid_preds = pipeline.predict(df_valid['safe_text'])

# Calculate accuracy on the validation set
accuracy = accuracy_score(df_valid['label'], valid_preds)
print("Validation Accuracy:", accuracy)

# Make predictions on the test set
test_preds = pipeline.predict(test['safe_text'])

# Make a submission dataframe
sub = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': test_preds
})
sub.to_csv('random_forest_submission.csv', index=False)
sub.head()
