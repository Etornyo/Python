import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from joblib import Parallel, delayed
import os

warnings.simplefilter('ignore')

# Define file paths
DATA_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# Function to read data in parallel
def read_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

# Read data in parallel
client_train, invoice_train, client_test, invoice_test = Parallel(n_jobs=-1)(
    delayed(read_data)(file_path) for file_path in [
        os.path.join(DATA_DIR, 'client_train.csv'),
        os.path.join(DATA_DIR, 'invoice_train.csv'),
        os.path.join(DATA_DIR, 'client_test.csv'),
        os.path.join(DATA_DIR, 'invoice_test.csv')
    ]
)

# Visualize fraudulent activities if 'target' column exists
if 'target' in client_train.columns:
    fraud_activities = client_train.groupby(['target'])['client_id'].count()
    plt.bar(x=fraud_activities.index, height=fraud_activities.values, tick_label=[0, 1])
    plt.title('Fraud - Target Distribution')
    plt.show()

# Visualize client distribution across districts and regions
for col in ['disrict', 'region']:  # Check for correct column names
    if col in client_train.columns:
        region = client_train.groupby([col])['client_id'].count()
        plt.bar(x=region.index, height=region.values)
        plt.title(col + ' distribution')
        plt.show()

# Convert the column invoice_date to date time format on both the invoice train and invoice test
for df in [invoice_train, invoice_test]:
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])

# Encode labels in categorical column
d = {"ELEC": 0, "GAZ": 1}
invoice_train['counter_type'] = invoice_train['counter_type'].map(d)
invoice_test['counter_type'] = invoice_test['counter_type'].map(d)

# Convert categorical columns to int for model
client_train['client_catg'] = client_train['client_catg'].astype(int)
client_train['disrict'] = client_train['disrict'].astype(int)

client_test['client_catg'] = client_test['client_catg'].astype(int)
client_test['disrict'] = client_test['disrict'].astype(int)

def aggregate_by_client_id(invoice_data):
    aggs = {}
    aggs['consommation_level_1'] = ['mean']
    aggs['consommation_level_2'] = ['mean']
    aggs['consommation_level_3'] = ['mean']
    aggs['consommation_level_4'] = ['mean']

    agg_trans = invoice_data.groupby(['client_id']).agg(aggs)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (invoice_data.groupby('client_id')
            .size()
            .reset_index(name='{}transactions_count'.format('1')))
    return pd.merge(df, agg_trans, on='client_id', how='left')

# Group invoice data by client_id
agg_train = aggregate_by_client_id(invoice_train)

print(agg_train.shape)
agg_train.head()

# Merge aggregate data with client dataset
train = pd.merge(client_train, agg_train, on='client_id', how='left')

# Aggregate test set
agg_test = aggregate_by_client_id(invoice_test)
test = pd.merge(client_test, agg_test, on='client_id', how='left')

train.shape, test.shape

# Drop redundant columns
sub_client_id = test['client_id']
drop_columns = ['client_id', 'creation_date']

for col in drop_columns:
    if col in train.columns:
        train.drop([col], axis=1, inplace=True)
    if col in test.columns:
        test.drop([col], axis=1, inplace=True)


x_train = train.drop(columns=['target'])
y_train = train['target']

# XGBoost Model
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(x_train, y_train)

# LightGBM Model
lgb_model = LGBMClassifier(boosting_type='gbdt', num_iterations=500)
lgb_model.fit(x_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Predictions from each model
xgb_preds = xgb_model.predict(test)
lgb_preds = lgb_model.predict(test)
rf_preds = rf_model.predict(test)

# Ensemble predictions (simple voting)
ensemble_preds = pd.DataFrame({
    'xgb': xgb_preds,
    'lgb': lgb_preds,
    'rf': rf_preds
})

# Take the mode across predictions
final_preds = ensemble_preds.mode(axis=1)[0]

# Submission dataframe
submission = pd.DataFrame(
    {
        'client_id': sub_client_id,
        'target': final_preds
    }
)

submission.head()

# Write submission to CSV
submission.to_csv('Submission_ensemble.csv', index=False)
