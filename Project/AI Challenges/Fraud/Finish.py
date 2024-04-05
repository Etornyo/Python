import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')
np.random.seed(4590)

print('Finished imports...')

# Load data
train_client = pd.read_csv('client_train.csv')
test_client = pd.read_csv('client_test.csv')
train_invoice = pd.read_csv('invoice_train.csv')
test_invoice = pd.read_csv('invoice_test.csv')

print('Finished data read...')

# Encode categorical variables
d = {"ELEC": 0, "GAZ": 1}
train_invoice['counter_type'] = train_invoice['counter_type'].map(d)
test_invoice['counter_type'] = test_invoice['counter_type'].map(d)

# Convert object columns to category type
train_client['client_catg'] = train_client['client_catg'].astype('category')
train_client['disrict'] = train_client['disrict'].astype('category')
test_client['client_catg'] = test_client['client_catg'].astype('category')
test_client['disrict'] = test_client['disrict'].astype('category')

# Extract date features
for df in [train_invoice, test_invoice]:
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month

# Aggregate invoice data at the client level
aggs = {
    'consommation_level_1': ['sum', 'max', 'min', 'mean', 'std'],
    'consommation_level_2': ['sum', 'max', 'min', 'mean', 'std'],
    'consommation_level_3': ['sum', 'max', 'min', 'mean', 'std'],
    'consommation_level_4': ['sum', 'max', 'min', 'mean', 'std'],
    'month': ['mean', 'max', 'min', 'std'],
    'year': ['nunique', 'max', 'min', 'mean'],
    'months_number': ['max', 'min', 'mean', 'sum'],
    'reading_remarque': ['max', 'min', 'mean', 'std', 'sum'],
    'counter_coefficient': ['max', 'min', 'mean'],
    'counter_number': ['nunique'],
    'counter_type': ['nunique', 'mean', 'sum'],
    'counter_statue': ['nunique'],
    'tarif_type': ['nunique', 'max', 'min'],
    'counter_code': ['nunique', 'max', 'mean', 'min'],
    'old_index': ['nunique', 'mean', 'std']
}

agg_train = train_invoice.groupby('client_id').agg(aggs)
agg_train.columns = ['_'.join(col).strip() for col in agg_train.columns.values]
agg_train.reset_index(inplace=True)

agg_test = test_invoice.groupby('client_id').agg(aggs)
agg_test.columns = ['_'.join(col).strip() for col in agg_test.columns.values]
agg_test.reset_index(inplace=True)

# Merge aggregated invoice data with client data
train = pd.merge(train_client, agg_train, on='client_id', how='left')
test = pd.merge(test_client, agg_test, on='client_id', how='left')

# Drop unnecessary columns
cols_to_drop = ['client_id', 'old_index_std', 'reading_remarque_std', 'month_std',
                'consommation_level_1_std', 'consommation_level_2_std', 'consommation_level_3_std', 'consommation_level_4_std']
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)


# if 'creation_date' in df.columns:
#     df['creation_date'] = pd.to_datetime(df['creation_date'])
# else:
#     print("The 'creation_date' column does not exist.")



# Convert creation_date to datetime and extract additional features
for df in [train, test]:
    df['creation_date'] = pd.to_datetime(df['creation_date'])
    df['year'] = df['creation_date'].dt.year
    df['month'] = df['creation_date'].dt.month
    df['month_diff'] = ((datetime.datetime.today() - df['creation_date']).dt.days) // 30

# Define features and target variable
X = train.drop('target', axis=1)
y = train['target']

# Define K-Fold cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

# Initialize lists to store predictions
preds_catboost = np.zeros((len(test),))
preds_xgboost = np.zeros((len(test),))

# Iterate through each fold
for i, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f'Fold {i + 1}/{n_splits}')

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    train['disrict'] = train['disrict'].astype ( 'category' )
    test['disrict'] = test['disrict'].astype ( 'category' )

    cat_features = ['disrict','client_catg']

    # CatBoost model
    cat_model = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.055,
                                   subsample=0.9, colsample_bylevel=0.9, random_seed=2024,
                                   verbose=50, cat_features=cat_features)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose_eval=50)

    preds_catboost += cat_model.predict_proba(test)[:, 1] / n_splits

    # XGBoost model
    xgb_model = xgb.XGBClassifier(n_estimators=1000, max_depth=9, learning_rate=0.05,
                                  subsample=0.9, colsample_bytree=0.9, random_state=2024,
                                  verbosity=1)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=50, feature_name=X_train.columns.tolist(), cat_features=cat_features)

    preds_xgboost += xgb_model.predict_proba(test)[:, 1] / n_splits

# Calculate final predictions
final_preds = (preds_catboost + preds_xgboost) / 2

# Write submission file
submission = pd.DataFrame({
    "client_id": test_client["client_id"],
    "target": final_preds
})
submission.to_csv('submission_fraudze.csv', index=False)
