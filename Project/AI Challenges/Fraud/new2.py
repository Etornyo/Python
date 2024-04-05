import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn import preprocessing
import time

from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from catboost import CatBoostClassifier




#Reduce Memory Usage
def reduce_memory_usage(df):
  for col in df.columns:
    col_type = df[col].dtype.name
    if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
      if (col_type != 'object'):
        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_type)[:3] == 'int':
          if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
          elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
          elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
          elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
            df[col] = df[col].astype(np.int64)

        else:
          if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            df[col] = df[col].astype(np.float16)
          elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)
          else:
            pass
    else:
      df[col] = df[col].astype('category')
    
  return df


class CFG:
  seed = 47
  
  
testInvoices = pd.read_csv(f'invoice_test.csv',low_memory=False)
trainInvoices = pd.read_csv(f'invoice_train.csv',low_memory=False)
testClient = pd.read_csv(f'client_test.csv',low_memory=False)
trainClient = pd.read_csv(f'client_train.csv',low_memory=False)

print("Train invoices: ", trainInvoices.shape)
print("Train clients: ", trainClient.shape, '\n')
print("Test invoices: ", testInvoices.shape)
print("Test clients: ", testClient.shape)

print('Number of missing rows in Train invoices:',trainInvoices.isna().sum().sum())
print('Number of missing rows in Test invoices:',testInvoices.isna().sum().sum(),'\n')
print('Number of missing rows in Train clients:',trainClient.isna().sum().sum())
print('Number of missing rows in Test clients:',testClient.isna().sum().sum())

print('Number of unique values in invoice_train:')
for col in trainInvoices.columns:
    print(f"{col}            <=========>                {trainInvoices[col].nunique()}")
    
trainClient.head()

trainInvoices.head()

trainClient['creation_year'] = pd.to_datetime(trainClient['creation_date'], dayfirst=True).dt.year
years=set(trainClient['creation_year'])


counterType = trainInvoices['counter_type'].tolist()
counterTypeElec = counterType.count('ELEC')*100 / len(counterType)
counterTypeGaz = counterType.count('GAZ')*100 / len(counterType)
plt.figure(figsize=(6,6))
plt.pie([counterTypeElec, counterTypeGaz], labels = ['Electricity','Gas'],autopct='%1.1f%%')
plt.title("Proportion of Counter type (Electricity to Gas)")
plt.show()

creationYear = trainClient.groupby(['creation_year'])['client_id'].count()
plt.figure(figsize=(12,3))
plt.plot(creationYear)
plt.title('Number of Clients by Creation Year')
plt.xlabel('Creation Year')
plt.ylabel('Number of Clients')
plt.xticks(range(min(creationYear.index), max(creationYear.index)+1, 5), rotation=45)
plt.show()

targetType = trainClient['target'].tolist()
targetType_0 = targetType.count(0)*100 / len(counterType)
targetType_1 = targetType.count(1)*100 / len(counterType)
plt.figure(figsize=(6,6))
plt.pie([targetType_0, targetType_1], labels = ['Not Fraud','Fraudulent'],autopct='%1.1f%%')
plt.title("Proportion of Fraudulent and Non-Fraudulent Consumption")
plt.show()

regionFraudPercent = trainClient.groupby("region")["target"].mean().sort_values() * 100
fig, ax = plt.subplots(figsize=(12, 3))
regionFraudPercent.plot(kind="bar", ax=ax, color="blue")
ax.set_xlabel("Region")
ax.set_ylabel("Percentage of Frauds")
ax.set_title("Percentage of Fraudulent Consumption by Region")

plt.show()

districtFraudPercent = trainClient.groupby("disrict")["target"].mean() * 100
fig, ax = plt.subplots(figsize=(12, 3))
districtFraudPercent.plot(kind="bar", ax=ax, color="blue")
ax.set_xlabel("Disrict")
ax.set_ylabel("Percentage of Frauds")
ax.set_title("Percentage of Fraudulent Consumption by disrict")
plt.show()

trainClient['client_catg'] = trainClient['client_catg'].astype(int)
clientCatfraudPercent = trainClient.groupby("client_catg")["target"].mean().sort_values() * 100
fig, ax = plt.subplots(figsize=(12, 3))
clientCatfraudPercent.plot(kind="bar", ax=ax)
ax.set_xlabel("Client Category")
ax.set_ylabel("Percentage of Frauds")
ax.set_title("Percentage of Fraudulent Consumption by Client Category")
plt.show()

mergedData = pd.merge(trainClient, trainInvoices, on="client_id")

fraudulentData = mergedData[mergedData["target"] == 1]
nonFraudulentData = mergedData[mergedData["target"] == 0]

consumptionData = fraudulentData.groupby("invoice_date").agg({"consommation_level_1": "sum", "consommation_level_2": "sum", "consommation_level_3": "sum", "consommation_level_4": "sum"}).reset_index()
consumptionData["invoice_date"] = pd.to_datetime(consumptionData["invoice_date"])
consumptionData.head()

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
for i, year in enumerate(invoiceYears[-2:]):
  ax = axes[i]
  yearlyConsumption = consumptionData[consumptionData["invoice_date"].dt.year == year]
  if not yearlyConsumption.empty:
    ax.plot(yearlyConsumption["invoice_date"], yearlyConsumption["consommation_level_1"], label="Level 1")
    ax.plot(yearlyConsumption["invoice_date"], yearlyConsumption["consommation_level_2"], label="Level 2")
    ax.plot(yearlyConsumption["invoice_date"], yearlyConsumption["consommation_level_3"], label="Level 3")
    ax.plot(yearlyConsumption["invoice_date"], yearlyConsumption["consommation_level_4"], label="Level 4")
    ax.set_ylabel("Consumption")
    ax.set_title(f"Consumption Levels for Clients with Frauds in {year}")
    ax.legend()
ax.set_xlabel("Invoice Date")
plt.show()


def compareCustomers(year, counterType):
  startDate = f'{year}-01-01'
  endDate = f'{year}-12-31'

  fraudulentData['invoice_date'] = pd.to_datetime(fraudulentData['invoice_date'])
  nonFraudulentData['invoice_date'] = pd.to_datetime(nonFraudulentData['invoice_date'])

  # Select invoices of a fraudulent and a non-fraudulent customer in 2019
  fraudulentClient = fraudulentData[fraudulentData['invoice_date'].dt.year == year].sample(n=1)
  nonFraudulentClient = nonFraudulentData[nonFraudulentData['invoice_date'].dt.year == year].sample(n=1)

  # Calculate the average consumption level 1 for each customer
  fraudAvgConsumpLevel1 = fraudulentData[fraudulentData['client_id'] == fraudulentClient['client_id'].values[0]]['consommation_level_1'].mean()
  nonFraudAvgConsumpLevel1 = nonFraudulentData[nonFraudulentData['client_id'] == nonFraudulentClient['client_id'].values[0]]['consommation_level_1'].mean()

  fraudDataFiltered = fraudulentData[(fraudulentData['counter_type'] == counterType) & (fraudulentData['invoice_date'] >= startDate) & (fraudulentData['invoice_date'] <= endDate)]
  nonFraudDataFiltered = nonFraudulentData[(nonFraudulentData['counter_type'] == counterType) & (nonFraudulentData['invoice_date'] >= startDate) & (nonFraudulentData['invoice_date'] <= endDate)]

# Calculate the average consumption level 1 for each month in the selected date range for each customer
fraudulentMonthly = fraudDataFiltered.groupby(pd.Grouper(key='invoice_date', freq='M')).agg({'consommation_level_1': 'mean'}).reset_index()
fraudulentMonthly['Monthly AVG'] = 'Fraudulent'
nonFraudulentMonthly = nonFraudDataFiltered.groupby(pd.Grouper(key='invoice_date', freq='M')).agg({'consommation_level_1': 'mean'}).reset_index()
nonFraudulentMonthly['Monthly AVG'] = 'Non-Fraudulent'

# Merge the two data frames on the month column
mergedMonthly = pd.concat([nonFraudulentMonthly,fraudulentMonthly])

# Plot the consumption discrepancy between the two customers
fig = px.line(mergedMonthly, x='invoice_date', y='consommation_level_1', color='Monthly AVG',
                line_dash='Monthly AVG', markers=True, template='plotly_white')
fig.update_layout(title=f'Consumption Discrepancy Comparison Between Two Selected Customers ({year})')
fig.show()
  
compareCustomers(year = 2019,
                 counterType = "ELEC")