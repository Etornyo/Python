import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from fastai.tabular import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold, StratifiedKFold
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import datetime
from tqdm import tqdm_notebook


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))




# DATA_PATH = 'input/'
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sample_sub = pd.read_csv('SampleSubmission (1).csv')


train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
test['Date'] = pd.to_datetime(test['Date'], format='%Y-%m-%d')
ID_COL, TARGET_COL = 'Place_ID X Date', 'target'
df = pd.concat([train, test]).reset_index(drop=True)
features = [c for c in df.columns if c not in ['Date', 'target_count', 'target_min', 'Place_ID X Date', 'target_variance', 'Place_ID', 'target_max', 'target']]
simple_feats = [c for c in features if ('angle' not in c) & ('height' not in c) & ('altittude' not in c)]
len(simple_feats)

df['placeID_freq'] = df['Place_ID'].map ( df['Place_ID'].value_counts () )
for i in range ( 1, 20 ):
    df[f'prev_target_{i}'] = df.sort_values ( by='Date' )[TARGET_COL].fillna ( method='ffill' ).shift (
        i ).sort_index ()
    df[f'next_target_{i}'] = df.sort_values ( by='Date' )[TARGET_COL].fillna ( method='bfill' ).shift (
        -i ).sort_index ()

for i in tqdm_notebook ( range ( 1, 15 ) ):
    df[f'magic_{i}'] = df.sort_values ( by='Date' )[TARGET_COL].shift ( i ).expanding ().mean ().fillna (
        method='ffill' ).sort_index ()
    df[f'magic2_{i}'] = df.sort_values ( by='Date' )[TARGET_COL].shift ( -i ).expanding ().mean ().fillna (
        method='bfill' ).sort_index ()

for i in tqdm_notebook ( range ( 1, 45 ) ):
    c = 'Date' + str ( i + 1 )
    df[c] = df['Date'] + datetime.timedelta ( days=i )
    _ = add_cyclic_datepart ( df, c, prefix=c )

for i in tqdm_notebook ( range ( 1, 11 ) ):
    tmp = df.sort_values ( by='Date' ).groupby ( 'Place_ID' )[simple_feats].shift ( i ).sort_index ()
    tmp_diff_prev = df[simple_feats] - tmp
    tmp.columns = [c + f'_prev_{i}' for c in tmp.columns]
    tmp_diff_prev.columns = [c + f'_prev_diff_{i}' for c in tmp_diff_prev.columns]
    df = pd.concat ( [df, tmp, tmp_diff_prev], axis=1 )

    tmp = df.sort_values ( by='Date' ).groupby ( 'Place_ID' )[simple_feats].shift ( -i ).sort_index ()
    tmp_diff_next = df[simple_feats] - tmp
    tmp.columns = [c + f'_next_{i}' for c in tmp.columns]
    tmp_diff_next.columns = [c + f'_next_diff_{i}' for c in tmp_diff_next.columns]
    df = pd.concat ( [df, tmp, tmp_diff_next], axis=1 )

for attr in ['day', 'month', 'week', 'dayofweek', 'weekofyear', 'days_in_month', 'is_month_start', 'is_month_end',
             'dayofyear']:
    df[attr] = getattr ( df['Date'].dt, attr )
df['is_weekend'] = (df['dayofweek'] >= 5) * 1
df['fortnight'] = df['day'] % 15
df['which_fortnight'] = df['day'] // 15
add_cyclic_datepart ( df, "Date", prefix="Current_Date_" )

features = [c for c in df.columns if c not in ['Date', 'target_count', 'target_min', 'Place_ID X Date', 'target_variance', 'Place_ID',
                                               'target_max', 'target',  'month_year_cos','month_year_sin','day_year_cos','day_year_sin']]
train = df[:train.shape[0]].reset_index(drop=True)
test = df[train.shape[0]:].reset_index(drop=True)
target = train[TARGET_COL]
len(features)


param = {'num_leaves': 100,
         'min_data_in_leaf': 40,
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "feature_fraction": 0.35,
         "metric": 'auc',
         "lambda_l1": 1,
         "lambda_l2": 1,
         "random_state": 6,
         "verbosity": -1,
          'metric' : 'rmse',
          'num_iterations': 2200}


target_cols = ['target']
oofs_df = pd.DataFrame()
preds_df = pd.DataFrame()
for t_col in target_cols:
    oofs_df[t_col] = np.zeros(len(train))
    preds_df[t_col] = np.zeros(len(test))

max_iter = 10
folds = StratifiedKFold ( n_splits=max_iter, random_state=1901 )

for fold_, (trn_idx, val_idx) in enumerate (
        folds.split ( train.values, pd.qcut ( target, 10, labels=False, duplicates='drop' ) ) ):
    print ( "\nfold n°{}".format ( fold_ ) )
    X_trn, X_val, X_test = train.iloc[trn_idx][features], train.iloc[val_idx][features], test[features]
    for t_col in target_cols:
        target = train[t_col]
        print ( f"\n\n**** {t_col} ****\n" )
        y_trn, y_val = target.iloc[trn_idx], target.iloc[val_idx]
        trn_data = lgb.Dataset ( X_trn, y_trn )
        val_data = lgb.Dataset ( X_val, y_val )

        clf = lgb.train ( param, trn_data, valid_sets=[trn_data, val_data], verbose_eval=50, early_stopping_rounds=200 )

        oofs_df[t_col][val_idx] = clf.predict ( X_val, num_iteration=clf.best_iteration )
        current_test_pred = clf.predict ( X_test, num_iteration=clf.best_iteration )
        current_test_pred[current_test_pred < 0] = 0
        preds_df[t_col] += current_test_pred / folds.n_splits



_ = plt.figure(figsize=(10, 10))
fi = pd.Series(index=features, data=clf.feature_importance())
_ = fi.sort_values()[-20:].plot(kind='barh')



rmse(target.values, oofs_df['target'].values)

predictions_test = preds_df['target']
predictions_test[predictions_test < 0] = 0

SUB_FILE_NAME = 'preds_lgbm_v3.csv'
sub_df = pd.DataFrame()
sub_df[ID_COL] = test[ID_COL]
sub_df[TARGET_COL] = predictions_test
sub_df.to_csv(SUB_FILE_NAME, index=False)
sub_df.head(10)


sub_df[TARGET_COL].describe()