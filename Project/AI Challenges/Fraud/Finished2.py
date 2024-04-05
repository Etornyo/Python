import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV

import warnings
import logging
warnings.filterwarnings('ignore')
np.random.seed(4590)


def hyperparameter_tuning(train, target):
    try:
        logger.info ( 'Performing hyperparameter tuning...' )

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [0.0, 0.1, 0.5]
        }

        # Create the LightGBM classifier
        model = LGBMClassifier ( boosting_type='gbdt' )

        # Create GridSearchCV object
        grid_search = GridSearchCV ( model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1 )

        # Perform grid search
        grid_search.fit ( train, target )

        logger.info ( 'Hyperparameter tuning completed.' )

        # Print the best parameters
        logger.info ( 'Best parameters found:' )
        logger.info ( grid_search.best_params_ )

        # Return the best model
        return grid_search.best_estimator_
    except Exception as e:
        logger.error ( f'Error performing hyperparameter tuning: {str ( e )}' )
        raise


def main():
    try:
        # Load data
        train_client, test_client, train_invoice, test_invoice = load_data ()

        # Preprocess data
        train_client, test_client, train_invoice, test_invoice = preprocess_data ( train_client, test_client,
                                                                                   train_invoice, test_invoice )

        # Feature engineering
        feature_engineering ( train_invoice, test_invoice )

        # Train-test split
        train = train_client  # Update with processed train data
        target = train['target']
        train.drop ( 'target', axis=1, inplace=True )
        test = test_client  # Update with processed test data

        # Hyperparameter tuning
        best_model = hyperparameter_tuning ( train, target )

        # Train model
        model = best_model.fit ( train, target )

        # Make predictions
        predictions = predict ( model, test )

        # Save predictions
        submission = pd.DataFrame ( {
            "client_id": test_client["client_id"],
            "target": predictions[:, 1]
        } )
        submission.to_csv ( 'submission.csv', index=False )

        logger.info ( 'Script executed successfully.' )
    except Exception as e:
        logger.error ( f'Script execution failed: {str ( e )}' )


if __name__ == "__main__":
    main ()
