"""This module contains LightGBM model for PV forecasting."""
import argparse
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from typing import Union

from src_pandas.utils import get_default_params


class LGBMModel:
    def __init__(self, params: dict):
        """Initialize LightGBM model with given parameters."""
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit LightGBM model.
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features, influences when training stops
            y_val (pd.Series, optional): Validation target
        """
        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dtrain]
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=valid_sets,
            num_boost_round=1000,
        )
        return self

    def predict(self, X):
        """Predict using the fitted LightGBM model."""
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class LGBMTune:
    def __init__(self, params: dict, X_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame]):
        """Initialize with dataset and search space."""
        self.params = params
        self.best_params = None
        self.X_train = X_train
        self.y_train = y_train

    def _create_hyperspace(self, trial):
        """Define hyperparameter search space for Optuna tuning."""
        basic_params = self.params['basic_params']
        int_params = self.params['int_params']
        float_params = self.params['float_params']

        basic_param_space = {
            param: details['value'] for param, details in basic_params.items()
        }
        int_param_space = {
            param: trial.suggest_int(
                name=int_params[param]['name'], 
                low=int_params[param]['min'], 
                high=int_params[param]['max'],
                step=int_params[param].get('step', 1)
            ) for param in int_params
        }

        float_param_space = {
            param: trial.suggest_float(
                name=float_params[param]['name'],
                low=float_params[param]['min'],
                high=float_params[param]['max'],
                step=float_params[param].get('step', 0.01)
            ) for param in float_params
        }

        hyper_space = {**basic_param_space, **int_param_space, **float_param_space}
        return hyper_space

    def _lgbm_objective(self, trial):
        """Objective function for Optuna to minimize.

        Args:
            trial (optuna.trial.Trial): Optuna trial object
        
        Returns:
            float: Mean RMSE across folds
        """
        # Create hyperparameter space
        hyper_space = self._create_hyperspace(trial)

        tscv = TimeSeriesSplit(n_splits=5)
        rmse_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = LGBMModel(hyper_space)
            model.fit(X_train_cv, y_train_cv, X_val_cv, y_val_cv)

            y_pred = model.predict(X_val_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    def optimize(self):
        n_trials = self.params["n_trials"]
        study = optuna.create_study(direction="minimize")
        study.optimize(self._lgbm_objective, n_trials=n_trials)
        self.best_params = study.best_trial.params
        return self.best_params

    def retrain_model_with_best_params(self):
        if self.best_params is None:
            raise ValueError("Best parameters not found. Run optimize() first.")
        model = LGBMModel(self.best_params)
        model.fit(self.X_train, self.y_train)
        return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="LightGBM model for PV forecasting.")
    argparser.add_argument(
        "-execute_optimize",
        type=bool,
        default=False,
        help="Whether to run hyperparameter optimization with Optuna."
    )
    args = argparser.parse_args()
    execute_optimize = args.execute_optimize

    # Load params
    with open("src_pandas/params_modules/data_param.yaml", 'r') as file:
        data_params = yaml.safe_load(file)
    with open("src_pandas/params_modules/model_param.yaml", 'r') as file:
        model_params = yaml.safe_load(file)
    datetime_column = data_params['datetime_column']
    target_column = data_params['target_column']
    feature_table_path = data_params['feature_table_save_path']
    train_start_date = data_params['train_start_date']
    train_end_date = data_params['train_end_date']
    test_start_date = data_params['test_start_date']
    test_end_date = data_params['test_end_date']
    lgbm_params = model_params['LGBM']
    
    # Load data
    df = pd.read_csv(feature_table_path, index_col=0)  # Pay attention to exclude time column

    train_df = df[(df.index >= train_start_date) & (df.index <= train_end_date)]
    test_df = df[(df.index >= test_start_date) & (df.index <= test_end_date)]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    if execute_optimize:
        # Tune
        tuner = LGBMTune(params=lgbm_params, X_train=X_train, y_train=y_train)
        best_params = tuner.optimize()
        print("Best Params:", best_params)

        # Retrain final model
        final_model = tuner.retrain_model_with_best_params()
    else:
        # Train directly without tuning
        final_model = LGBMModel(get_default_params(lgbm_params))
        final_model.fit(X_train, y_train)

    # Evaluate
    y_pred = final_model.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Final LGBM RMSE: {rmse}")

    # Plot the predictions vs actuals
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()

  # Plot the feature importance
    lgb.plot_importance(
        final_model.model,       
        max_num_features=10,     # top 10 features
        importance_type='split', # or 'gain'
        figsize=(8, 6),       
        grid=True
    )
    plt.show()


# TODO: 1. add sampler
#       2. add workflow under Tune class