import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from datetime import timedelta

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Convert the 'Date' column to datetime object
train_data['Date'] = pd.to_datetime(train_data['Date'])

# Create new columns for the next 10 days
for i in range(1, 11):
    c = f'Day {i}'
    train_data[c] = train_data['Date'] + pd.Timedelta(days=i)

# Define the features and target variable
features = ['Place_ID', 'Date', 'Place_ID X Date', 'Place_ID X Date_prev_1', 'Place_ID X Date_prev_2',
            'Place_ID X Date_prev_3', 'Place_ID X Date_prev_4', 'Place_ID X Date_prev_5',
            'Place_ID X Date_prev_6', 'Place_ID X Date_prev_7', 'Place_ID X Date_prev_8',
            'Place_ID X Date_prev_9', 'Place_ID X Date_prev_10', 'Place_ID X Date_prev_10',
            'Place_ID X Date_next_1', 'Place_ID X Date_next_2', 'Place_ID X Date_next_3',
            'Place_ID X Date_next_4', 'Place_ID X Date_next_5', 'Place_ID X Date_next_6',
            'Place_ID X Date_next_7', 'Place_ID X Date_next_8', 'Place_ID X Date_next_9',
            'Place_ID X Date_next_10', 'Place_ID X Date_next_10', 'day', 'month', 'dayofweek',
            'weekofyear', 'days_in_month', 'is_month_start', 'is_month_end', 'dayofyear',
            'is_weekend', 'fortnight', 'which_fortnight']

# Add cyclic dateparts
for i in range(1, 45):
    c = 'Date' + str(i + 1)
    train_data[c] = train_data['Date'] + pd.Timedelta(days=i)
    # Add cyclic datepart
    # _ = add_cyclic_datepart(train_data, c, prefix=c)

# Add lag features
for i in range(1, 11):
    tmp = train_data.sort_values(by='Date').groupby('Place_ID')[features].shift(i).sort_index()
    tmp_diff_prev = train_data[features] - tmp
    tmp.columns = [c + f'_prev_{i}' for c in tmp.columns]
    tmp_diff_prev.columns = [c + f'_prev_diff_{i}' for c in tmp_diff_prev.columns]
    train_data = pd.concat([train_data, tmp, tmp_diff_prev], axis=1)

    tmp = train_data.sort_values(by='Date').groupby('Place_ID')[features].shift(-i).sort_index()
    tmp_diff_next = train_data[features] - tmp
    tmp.columns = [c + f'_next_{i}' for c in tmp.columns]
    tmp_diff_next.columns = [c + f'_next_diff_{i}' for c in tmp_diff_next.columns]
    train_data = pd.concat([train_data, tmp, tmp_diff_next], axis=1)

# Add date features
for attr in ['day', 'month', 'dayofweek', 'weekofyear', 'days_in_month', 'is_month_start', 'is_month_end',
             'dayofyear']:
    train_data[attr] = getattr(train_data['Date'].dt, attr)
train_data['is_weekend'] = (train_data['dayofweek'] >= 5) * 1
train_data['fortnight'] = train_data['day'] % 15
train_data['which_fortnight'] = train_data['day'] // 15
# add_cyclic_datepart(train_data, "Date", prefix="Current_Date_")

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_data[features], train_data['target'],
                                                    test_size=0.2, random_state=42)

# Define parameter dictionary for the LightGBM model
param = {
    'num_leaves': 100,  # Set the number of leaves
    'objective': 'regression',
    'boosting': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': -1,
    'min_data_in_leaf': 40,
    'feature_fraction': 0.35,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'random_state': 6,
    'verbosity': -1,
    'metric': 'rmse',
    'num_iterations': 2200
}

# Model training (Example with LightGBM)
model = LGBMRegressor(**param)
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))
print("Root Mean Squared Error:", rmse)

# Model prediction on the test dataset
test_predictions = model.predict(test_data[features])

# Prepare submission file
# Assuming 'Place_ID' and 'Date' columns exist in the test dataset
submission = pd.DataFrame({
    'Place_ID X Date': test_data['Place_ID'] + ' X ' + test_data['Date'].astype(str),
    'target': test_predictions
})
submission.to_csv('submission_lgbm_modified_2.csv', index=False)