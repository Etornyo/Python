import pandas as pd
import matplotlib.pyplot as plt
import lightgbm
from lightgbm import LGBMRegressor

import warnings
warnings.simplefilter('ignore')

client_train = pd.read_csv(f'client_train.csv', low_memory=False)
invoice_train = pd.read_csv(f'invoice_train.csv', low_memory=False)

client_test = pd.read_csv(f'client_test.csv', low_memory=False)
invoice_test = pd.read_csv(f'invoice_test.csv', low_memory=False)
# sample_submission = pd.read_csv(f'SampleSubmission.csv', low_memory=False)

#compare size of the various datasets
print(client_train.shape, invoice_train.shape, client_test.shape, invoice_train.shape)


invoice_train.head()

client_train.head()


invoice_train.describe()

client_train.describe()

invoice_train.info()

client_train.info()


#Getting unique values on the invoice train data
for col in invoice_train.columns:
    print(f"{col} - {invoice_train[col].nunique()}")

#Getting unique values on the invoice train data
for col in client_train.columns:
    print(f"{col} - {client_train[col].nunique()}")

#check for missing values
invoice_train.isnull().sum()

client_train.isnull().sum()


#Visualize fraudulent activities
fraudactivities = client_train.groupby(['target'])['client_id'].count()
plt.bar(x=fraudactivities.index, height=fraudactivities.values, tick_label = [0,1])
plt.title('Fraud - Target Distribution')
plt.show()


#Visualize client distribution across districts and regions
for col in ['disrict','region']:
    region = client_train.groupby([col])['client_id'].count()
    plt.bar(x=region.index, height=region.values)
    plt.title(col+' distribution')
    plt.show()


#convert the column invoice_date to date time format on both the invoice train and invoice test
for df in [invoice_train,invoice_test]:
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])

#encode labels in categorical column
d={"ELEC":0,"GAZ":1}
invoice_train['counter_type']=invoice_train['counter_type'].map(d)
invoice_test['counter_type']=invoice_test['counter_type'].map(d)

#convert categorical columns to int for model
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


#group invoice data by client_id
agg_train = aggregate_by_client_id(invoice_train)

print(agg_train.shape)
agg_train.head()

#merge aggregate data with client dataset
train = pd.merge(client_train,agg_train, on='client_id', how='left')

#aggregate test set
agg_test = aggregate_by_client_id(invoice_test)
test = pd.merge(client_test,agg_test, on='client_id', how='left')


train.shape, test.shape

#drop redundant columns
sub_client_id = test['client_id']
drop_columns = ['client_id', 'creation_date']

for col in drop_columns:
    if col in train.columns:
        train.drop([col], axis=1, inplace=True)
    if col in test.columns:
        test.drop([col], axis=1, inplace=True)


x_train = train.drop(columns=['target'])
y_train = train['target']

'''param = {
    'n_estimators': 500,
    'max_depth': 10,
    'boosting_type':'gbdt',
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}'''

param = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 6,
    'num_iterations': 2200
}

model = LGBMRegressor(**param)
model.fit(X_train, y_train)

predictions = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))
print("Root Mean Squared Error:", rmse)


x_train.disrict.unique()

test.columns

preds = model.predict(test)
preds = pd.DataFrame(preds, columns=['target'])
preds.head()


submission = pd.DataFrame(
    {
        'client_id': sub_client_id,
        'target': preds['target']
    }
)

submission.head()

submission.to_csv(f'submission_no.csv', index=False)


