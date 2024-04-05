import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping

from catboost import CatBoostRegressor

from math import sqrt
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
print ('Finished imports...')

# Reading data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')


train.shape, test.shape

train.head(2)

test.head(2)

# Times for the dataset
train['Date'] = pd.to_datetime(train['Date'])
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day
train['Dayofweek'] = train['Date'].dt.dayofweek
train['DayOfyear'] = train['Date'].dt.dayofyear
train['DayOfyear'] = train['Date'].dt.dayofyear

test['Date'] = pd.to_datetime(test['Date'])
test['Month'] = test['Date'].dt.month
test['Day'] = test['Date'].dt.day
test['Dayofweek'] = test['Date'].dt.dayofweek
test['DayOfyear'] = test['Date'].dt.dayofyear
test['DayOfyear'] = test['Date'].dt.dayofyear

train.head(2)

# Drop unnecessary features
train.drop(['Place_ID X Date','Date','Place_ID','target_min','target_max','target_variance','target_count'], axis=1, inplace=True)
test.drop(['Place_ID X Date','Date','Place_ID'], axis=1, inplace=True)


train.shape, test.shape


features = ['relative_humidity_2m_above_ground',
                'v_component_of_wind_10m_above_ground',
                'L3_CLOUD_surface_albedo',
                'L3_CO_CO_column_number_density',
                'temperature_2m_above_ground',
                'L3_CLOUD_cloud_optical_depth',
                'L3_O3_O3_column_number_density'
                ]

for feat in features:
    i = 0
    for i in range(3):
        train[feat+'_lag'+str(i+1)] = train[feat].shift(i+1)
        test[feat+'_lag'+str(i+1)] = test[feat].shift(i+1)


X = train.drop(labels=['target'], axis=1)
y = train['target'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

X.head(5)

errlgb = []
y_pred_totlgb = []
Xtest = test

fold = KFold ( n_splits=10, shuffle=True, random_state=2 )

for train_index, test_index in fold.split ( X ):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
y_train, y_test = y[train_index], y[test_index]

cb_model = CatBoostRegressor ( iterations=8000,
learning_rate = 0.050,
depth = 8,
eval_metric = 'RMSE',
random_seed = 42,
bagging_temperature = 0.2,
od_type = 'Iter',
metric_period = 50,
od_wait = 300)
cb_model.fit ( X, y,
use_best_model = True,
verbose = 50)

y_pred = cb_model.predict ( X_test )

print ( "RMSE: ", np.sqrt ( mean_squared_error ( y_test, y_pred ) ) )

errlgb.append ( np.sqrt ( mean_squared_error ( y_test, y_pred ) ) )

p = cb_model.predict ( Xtest )

y_pred_totlgb.append ( p )