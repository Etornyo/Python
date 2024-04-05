import pandas as pd
from fastai.text import *

# Load the data
train = pd.read_csv('Train.csv').dropna(0) # Read in train, ignoring one row with missing data
test = pd.read_csv('Test.csv').fillna('') # Read in test
test['label'] = 0 # We'll fill this in with predictions later

# Build the databunch, and keep 1000 rows for validation
df_valid = train.sample(1000)
df_train = train.loc[~train.tweet_id.isin(df_valid.tweet_id.values)]
data_clas = TextClasDataBunch.from_df(path=Path(''), train_df=df_train,
                                      valid_df=df_valid,
                                      test_df=test,
                                      label_cols='label',
                                      text_cols='safe_text')

# Learner
clas = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3, metrics=[rmse])

# Train the model
clas.fit_one_cycle(20)

# Get predictions
preds, _ = clas.get_preds(DatasetType.Test)

# Make a submission dataframe
sub = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': [p[0].item() for p in preds.numpy()]
})
sub.to_csv('first_try_fastai_20_epochs.csv', index=False)
