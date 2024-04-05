   #Starter 
DATA = '/content'
TRAIN = f'{DATA}/train'
TEST = f'{DATA}/test'
OUTPUT= f'{DATA}/output'

#IMPORTS
import pandas as dan
import matplotlib.pyplot as ean
import lightgbm
from lightgbm import LGBMClassifier
import os.path
from os import path
import warnings
warnings.simplefilter('ignore')
import requests, os

#Files 
train_ZIP= "train.zip"
test_ZIP= "test.zip"
non_sample_submit= "non_sample_submitsamplesubmission.csv"



# Download and extracting files
for pth in [TRAIN, TEST, OUTPUT]:
  if path.exists(pth) == False:
    os.mkdir(pth)




#READ DATA 
path = '/Users/manuel/Desktop/Zindi'
client_train = dan.read_csv(path+'client_train.csv')
invoice_train = dan.read_csv(path+'invoice_train.csv')

client_test = dan.read_csv(path+'client_test.csv')
invoice_test = dan.read_csv(path+'invoice_test.csv')

non_sample_submit = dan.read_csv(path+'non_sampleSubmission.csv')




#COLLECTION OF DATA 
#Contrasting the various datasets 
print(client_train.shape, invoice_train.shape, client_test.shape, invoice_test.shape)




# MANUPULATION OF THE DATASET AVAILABLE 
# Selecting and printing the first row of the dataset
invoice_train.head()
client_train.head()

#Get a specific row of interest in the dataset
invoice_train.info()
client_train.info()

#Breakdown the dataset into blocks
invoice_train.describe()
client_train.describe()

#Snapshot of fraud
fraudulence = client_train.groupby(['target'])['client_id'].count()
ean.bar(x=fraudulence.index, height=fraudulence.values, tick_label = [0,1])
ean.title('Activity Distribution')
ean.show()




#Client_train
for col in client_train.columns:
    print(f"{col} - {client_train[col].nunique()}")

#Checking for omissions
client_train.isnull().sum()

#Chart 
for col in ['disrict','region']:
    region = client_train.groupby([col])['client_id'].count()
    ean.bar(x=region.index, height=region.values)
    ean.title(col+'Activity distribution')
    ean.show()

#Integer conversion
client_train['client_catalogue'] = client_train['client_catalogue'].astype(int)
client_train['disrict'] = client_train['disrict'].astype(int)
client_test['client_catalogue'] = client_test['client_catalogue'].astype(int)
client_test['disrict'] = client_test['disrict'].astype(int)




#Invoice_train
for col in invoice_train.columns:
    print(f"{col} - {invoice_train[col].nunique()}")

#Checking for omissions
invoice_train.isnull().sum()

#invoice date to date-time format 
for df in [invoice_train,invoice_test]:
    df['date_of_invoice'] = dan.to_datetime(df['date_of_invoice'])

#Chart
d={"Element":0,"Group":1}
invoice_train['count']=invoice_train['count'].map(d)
invoice_test['count']=invoice_test['count'].map(d)

#grouping invoice by client_id and print 
def agg_using_client_id(invoice_data):
    aggs = {}
    aggs['consommation_lvl_1'] = ['average']
    aggs['consommation_lvl_2'] = ['average']
    aggs['consommation_lvl_3'] = ['average']
    aggs['consommation_lvl_4'] = ['average']

    agg_vec = invoice_data.groupby(['client_id']).agg(aggs)
    agg_vec.columns = ['_'.join(col).strip() for col in agg_vec.columns.values]
    agg_vec.reset_index(inplace=True)

    df = (invoice_data.groupby('client_id')
            .size()
            .reset_index(name='{}vecagg_vecactions_count'.format('1')))
    return dan.merge(df, agg_vec, on='client_id', how='left')
agg_train = agg_using_client_id(invoice_train)
print(agg_train.shape)
agg_train.head()

#merge aggregate data with client dataset
train = dan.merge(client_train,agg_train, on='client_id', how='left')
x_train = train.drop(columns=['target'])
x_train.disrict.unique()
y_train = train['target']
model = LGBMClassifier(boosting_type='gbdt',num_iteration=500)
model.fit(x_train, y_train)

agg_test = agg_using_client_id(invoice_test)
test = dan.merge(client_test,agg_test, on='client_id', how='left')
train.shape, test.shape

#delete redundant columns
sub_client_id = test['client_id']
drop_columns = ['client_id', 'date']

for col in drop_columns:
    if col in train.columns:
        train.drop([col], axis=1, inplace=True)
    if col in test.columns:
        test.drop([col], axis=1, inplace=True)




#PREDICTIONS 
test.columns
predictions = model.predict(test)
predictions = dan.DataFrame(predictions, columns=['target'])
predictions.head()

#Data submissions
submit = dan.DataFrame(
    {
        'client_id': sub_client_id,
        'target': predictions['target']
    }
)
submit.head()
submit.to_csv(f'{OUTPUT}/submit.csv',index=False)
