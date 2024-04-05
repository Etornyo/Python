import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
import warnings

warnings.simplefilter('ignore')

# Define file paths
DATA = '/content'
TRAIN = f'{DATA}/train'
TEST = f'{DATA}/test'
OUTPUT = f'{DATA}/output'

# Define file names
train_ZIP = "train.zip"
test_ZIP = "test.zip"
# non_sample_submit = "non_sample_submitsamplesubmission.csv"

#  Function to download and extract files
# def download_and_extract_files(paths):
#     for pth in paths:
#         if not os.path.exists(pth):
#             os.mkdir(pth)

# Function to read data
def read_data(train_path, test_path, output_path):
    client_train = pd.read_csv('client_train.csv')
    invoice_train = pd.read_csv('invoice_train.csv')
    client_test = pd.read_csv('client_test.csv')
    invoice_test = pd.read_csv('invoice_test.csv')
    # non_sample_submit = pd.read_csv('non_sampleSubmission.csv')     , non_sample_submit    , non_sample_submit
    return client_train, invoice_train, client_test, invoice_test

# Download and extract files
# download_and_extract_files([TRAIN, TEST, OUTPUT])

# Read data
client_train, invoice_train, client_test, invoice_test = read_data(TRAIN, TEST, OUTPUT)

# Collection of Data
print(client_train.shape, invoice_train.shape, client_test.shape, invoice_test.shape)

# Manipulation of the Dataset Available
invoice_train.head()
client_train.head()
invoice_train.info()
client_train.info()
invoice_train.describe()
client_train.describe()

fraudulence = client_train.groupby(['target'])['client_id'].count()
plt.bar(x=fraudulence.index, height=fraudulence.values, tick_label=[0, 1])
plt.title('Activity Distribution')
plt.show()

# Client_train
for col in client_train.columns:
    print(f"{col} - {client_train[col].nunique()}")

client_train.isnull().sum()

for col in ['disrict', 'region']:
    region = client_train.groupby([col])['client_id'].count()
    plt.bar(x=region.index, height=region.values)
    plt.title(col + 'Activity distribution')
    plt.show()

client_train['client_catalogue'] = client_train['client_catalogue'].astype(int)
client_train['disrict'] = client_train['disrict'].astype(int)
client_test['client_catalogue'] = client_test['client_catalogue'].astype(int)
client_test['disrict'] = client_test['disrict'].astype(int)

# Invoice_train
for col in invoice_train.columns:
    print(f"{col} - {invoice_train[col].nunique()}")

invoice_train.isnull().sum()

for df in [invoice_train, invoice_test]:
    df['date_of_invoice'] = pd.to_datetime(df['date_of_invoice'])

d = {"Element": 0, "Group": 1}
invoice_train['count'] = invoice_train['count'].map(d)
invoice_test['count'] = invoice_test['count'].map(d)

def agg_using_client_id(invoice_data):
    aggs = {}
    aggs['consommation_lvl_1'] = ['mean']
    aggs['consommation_lvl_2'] = ['mean']
    aggs['consommation_lvl_3'] = ['mean']
    aggs['consommation_lvl_4'] = ['mean']

    agg_vec = invoice_data.groupby(['client_id']).agg(aggs)
    agg_vec.columns = ['_'.join(col).strip() for col in agg_vec.columns.values]
    agg_vec.reset_index(inplace=True)

    df = (invoice_data.groupby('client_id')
          .size()
          .reset_index(name='{}vecagg_vecactions_count'.format('1')))
    return pd.merge(df, agg_vec, on='client_id', how='left')

agg_train = agg_using_client_id(invoice_train)
print(agg_train.shape)
agg_train.head()

train = pd.merge(client_train, agg_train, on='client_id', how='left')
x_train = train.drop(columns=['target'])
x_train.disrict.unique()
y_train = train['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

agg_test = agg_using_client_id(invoice_test)
test = pd.merge(client_test, agg_test, on='client_id', how='left')
train.shape, test.shape

sub_client_id = test['client_id']
drop_columns = ['client_id', 'date']

for col in drop_columns:
    if col in train.columns:
        train.drop([col], axis=1, inplace=True)
    if col in test.columns:
        test.drop([col], axis=1, inplace=True)

test.columns
# Predict probabilities instead of classes
probabilities = model.predict_proba(test)[:, 1]  # Probability of the positive class (fraud)
auc_score = roc_auc_score(sub_client_id, probabilities)
print("AUC Score:", auc_score)

# predictions = model.predict(test)
# predictions = pd.DataFrame(predictions, columns=['target'])
# predictions.head()

# submit = pd.DataFrame(
#     {
#         'client_id': sub_client_id,
#         'target': predictions['target']
#     }
# )
# submit.head()
# submit.to_csv(f'{OUTPUT}/submit.csv', index=False)
