import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, cv

# Load the data
train = pd.read_csv('Train.csv').dropna(axis=0) # Read in train, ignoring one row with missing data
test = pd.read_csv('Test.csv').fillna('') # Read in test

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

# Convert data to CatBoost Pool format
train_pool = Pool(X_train, label=y_train)
valid_pool = Pool(X_valid, label=y_valid)

# Define hyperparameters grid for CatBoost
params = {
    'iterations': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
'loss_function': 'Logloss'
}

# Perform cross-validation with CatBoost
cv_results = cv(pool=train_pool, params=params, fold_count=3, early_stopping_rounds=10, shuffle=True, partition_random_seed=42, verbose=100)

# Get best iteration
best_iteration = int(np.argmax(cv_results['test-Accuracy-mean']))

# Train CatBoost model with best hyperparameters
best_params = params.copy()
best_params['iterations'] = best_iteration
best_model = CatBoostClassifier(**best_params, random_state=42)
best_model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=10, verbose=100)

# Predict on the validation set
valid_preds_catboost = best_model.predict(X_valid)

# Calculate accuracy on the validation set
accuracy_catboost = accuracy_score(y_valid, valid_preds_catboost)
print("Validation Accuracy (CatBoost):", accuracy_catboost)

# Make predictions on the test set for CatBoost
test_preds_catboost = best_model.predict(X_test)

# Make a submission dataframe for CatBoost
sub_catboost = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': test_preds_catboost.reshape(-1)
})
sub_catboost.to_csv('catboost_submission_cv.csv', index=False)
sub_catboost.head()
