import numpy as np
import pandas as pd
import os
from fastai.text import *

# Load the data
train = pd.read_csv('Train.csv').dropna(axis=0) # Read in train, ignoring one row with missing data
test = pd.read_csv('Test.csv').fillna('') # Read in test

# Build the databunch, and keep 1000 rows for validation
df_valid = train.sample(1000)
df_train = train.loc[~train.tweet_id.isin(df_valid.tweet_id.values)]
print(df_valid.shape, df_train.shape)

# Create TextClasDataBunch
data_clas = TextClasDataBunch.from_df(path=Path(''), train_df=df_train,
                                      valid_df=df_valid, test_df=test,
                                      label_cols='label', text_cols='safe_text')

# Create a learner object (renamed from 'class' to 'learner')
learner = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3, metrics=[accuracy])

# Train the model
learner.fit_one_cycle(20)

# Get predictions
preds, _ = learner.get_preds(DatasetType.Test)

# Convert probabilities to class labels
predicted_labels = preds.argmax(dim=-1).numpy()

# Make a submission dataframe
sub = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': predicted_labels
})
sub.to_csv('first_try_fastai_20_epochs.csv', index=False)
sub.head()
