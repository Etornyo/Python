import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from catboost import CatBoostClassifier

# Load the data
train = pd.read_csv('Train.csv').dropna(axis=0) # Read in train, ignoring one row with missing data
test = pd.read_csv('Test.csv').fillna('') # Read in test

# Split data into training and validation sets
df_train, df_valid = train_test_split(train, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform text data into TF-IDF features
X_train = vectorizer.fit_transform(df_train['safe_text'])
X_valid = vectorizer.transform(df_valid['safe_text'])
X_test = vectorizer.transform(test['safe_text'])

# Define labels
y_train = df_train['label']
y_valid = df_valid['label']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    X_train_fold, X_valid_fold = X_train[train_idx], X_train[valid_idx]
    y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    
    # Model
    catboost_classifier = CatBoostClassifier(iterations=8000,
                                learning_rate=0.05,
                                depth=8,
                                eval_metric='RMSE',
                                random_seed = 42,
                                bagging_temperature = 0.2,
                                od_type='Iter',
                                metric_period = 50,
                                od_wait=300)
    
    catboost_classifier.fit( X_train_fold, y_train_fold,
                            use_best_model=True,
                            verbose=5       
    )
    
    y_pred = catboost_classifier.predict(X_test) 

    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    
    errlgb.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    p = cb_model.predict(Xtest)
    
    y_pred_totlgb.append(p)

# Perform grid search

# Define evaluation set
eval_set = [(X_valid_fold, y_valid_fold)]

# Define parameter grid for grid search
param_grid = {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6],
        'l2_leaf_reg': [1, 3]
    }

    # Initialize CatBoost classifier

'''accuracy'''
grid_search = GridSearchCV(estimator=catboost_classifier, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train_fold, y_train_fold, eval_set=eval_set)

# Print best parameters and corresponding accuracy for each fold
print(f"Fold {fold_ + 1} Best Parameters:", grid_search.best_params_)
print(f"Fold {fold_ + 1} Best rmse:",np.sqrt(grid_search.best_score_))
# print(f"Fold {fold_ + 1} Best Accuracy:", grid_search.best_score_)

# Use best model for predictions
best_model = grid_search.best_estimator_
valid_preds_catboost = best_model.predict(X_valid_fold)

# Calculate accuracy on the validation set for each fold
# accuracy_catboost = accuracy_score(y_valid_fold, valid_preds_catboost)
rmse_catboost = np.sqrt(mean_squared_error(y_valid_fold,valid_preds_catboost))
print(f"Fold {fold_ + 1} Validation RMSE (CatBoost):",rmse_catboost )
# accuracy_catboost

# Make predictions on the test set for each fold
test_preds_catboost = best_model.predict(X_test)

# Make a submission dataframe for CatBoost for each fold (optional)
sub_catboost = pd.DataFrame({
        'tweet_id': test['tweet_id'],
        'label': test_preds_catboost.reshape(-1)
    })
sub_catboost.to_csv(f'catboost_submission_gridsearch_fold_{fold_ + 1}.csv', index=False)
sub_catboost.head()
