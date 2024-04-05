import pandas as pd
from simpletransformers.classification import ClassificationModel

train = pd.read_csv('Train.csv').dropna(0)
test = pd.read_csv('Test.csv').fillna('')
test['label'] = 0

df_valid = train.sample(1000)
df_train = train.loc[~train.tweet_id.isin(df_valid.tweet_id.values)]

args = {"reprocess_input_data": True, "overwrite_output_dir": True,
        'fp16': False,
        "num_train_epochs": 3,
        "learning_rate": 1e-4,
        "max_seq_length": 128,
        'regression': True}

df_train = df_train[['safe_text', 'label']]
df_valid = df_valid[['safe_text', 'label']]

model = ClassificationModel(
    "distilbert", "distilbert-base-uncased", num_labels=1, args=args
)

model.train_model(df_train)

result, model_outputs, _ = model.eval_model(df_valid)

sub = pd.DataFrame({
    'tweet_id': test['tweet_id'],
    'label': model.predict(test['safe_text'].values)[0]
})
sub.to_csv('transformer_1.csv', index=False)
