
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
train = pd.read_csv('Train.csv').dropna(axis=0)
test = pd.read_csv('Test.csv').fillna('')
test['label'] = 0

# Feature engineering
# (you may need to preprocess text features into numerical features for XGBoost)

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(train.drop(columns=['label']), train['label'], test_size=0.1, random_state=42)

# Define XGBoost dataset
train_data = xgb.DMatrix(data=X_train, label=y_train)
valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    # Add more parameters as needed
}

# Train the model
model = xgb.train(params, train_data, num_boost_round=1000, evals=[(train_data, 'train'), (valid_data, 'valid')], early_stopping_rounds=100, verbose_eval=100)

# Predictions
y_pred = model.predict(xgb.DMatrix(test.drop(columns=['label'])))

# Make a submission dataframe
sub = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': y_pred
})
sub.to_csv('xgboost1_submission.csv', index=False)
