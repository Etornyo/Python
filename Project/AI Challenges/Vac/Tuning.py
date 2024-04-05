# Creating a databunch for the language model
data_lm = TextLMDataBunch.from_df(path='', train_df=df_train,
                                  valid_df=df_valid,
                                  text_cols='safe_text')

# And the learner
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

# Training the language model
learn.fit_one_cycle(2, 1e-2)
learn.unfreeze()
learn.fit_one_cycle(3, 1e-3)
learn.save_encoder('ft_enc')

# Creating the classifier learner
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, metrics=[rmse])
learn.load_encoder('ft_enc')

# Training the classifier
learn.fit_one_cycle(2, 1e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(3, slice(5e-3/2., 5e-3))
learn.unfreeze()
learn.fit_one_cycle(8, slice(2e-3/100, 2e-3))

# Save predictions
preds, _ = learn.get_preds(DatasetType.Test)
sub = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': [p[0].item() for p in preds.numpy()]
})
sub.to_csv('fastai_2nd_try_lm.csv', index=False)
