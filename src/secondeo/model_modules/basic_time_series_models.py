"""This module contains basic time series models for PV forecasting."""
import argparse
import optuna
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

from src_pandas.utils import merge_dfs_on_datetime


class ARIMAModel:
    def __init__(self, params: dict):
        """Initialize ARIMA model with given order (p,d,q)."""

        self.order = (params['p'], params['d'], params['q'])
        self.model = None
        self.model_fit = None

    def fit(self, y_train: pd.DataFrame):
        """Fit the ARIMA model to the time series data.
        Args:
            y_train (pd.DataFrame): Time series data to fit the model.
        """
        self.model = ARIMA(y_train, order=self.order)
        self.model_fit = self.model.fit()
        return self

    def predict(self, steps: int):
        """Make predictions using the fitted ARIMA model.
        Args:
            steps (int): Number of steps to forecast, i.e. the size of the test set.
        """
        forecast = self.model_fit.forecast(steps=steps)
        return forecast
    

class ARIMATune:
    def __init__(self, y_train: pd.DataFrame, y_test: pd.DataFrame, params: dict):
        """Initialize with time series data."""
        self.y_train = y_train
        self.y_test = y_test
        self.params = params
        self.best_params = None

    def arima_objective(self, trial):
        p = trial.suggest_int(
            'p', 
            self.params['p']['min'], 
            self.params['p']['max']
        )  # autoregressive order
        d = trial.suggest_int(
            'd', 
            self.params['d']['min'], 
            self.params['d']['max']
        )
        q = trial.suggest_int(
            'q', 
            self.params['q']['min'], 
            self.params['q']['max']
        )  # moving average order
        try:
            model = ARIMAModel(params={
                'p': p,
                'd': d,
                'q': q
            })
            model.fit(self.y_train)
            y_pred = model.predict(steps=len(self.y_test))
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            return rmse
        except:
            return np.inf

    def optimize(self):
        n_trials = self.params['n_trials']
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.arima_objective(trial), n_trials=n_trials)
        self.best_params = study.best_trial.params
        return self.best_params

    def retrained_model_with_best_params(self):
        if self.best_params is None:
            raise ValueError("Best parameters not found. Run optimize() first.")
        model = ARIMAModel(params=self.best_params)
        model.fit(self.y_train)
        return model
    

class SARIMAXModel:
    def __init__(self, params: dict):
        """Initialize SARIMAX model with given order and seasonal_order."""
        self.order = (params['p'], params['d'], params['q'])
        self.seasonal_order = (params['P'], params['D'], params['Q'], params['s'])
        self.model = None
        self.model_fit = None

    def fit(self, y_train: pd.DataFrame, X_train: pd.DataFrame):
        """Fit the SARIMAX model to the time series data with exogenous variables.
        Args:
            y_train (pd.DataFrame): Time series data to fit the model.
            X_train (pd.DataFrame): Exogenous variables.
        """
        self.model = SARIMAX(y_train,
                             exog=X_train,
                             order=self.order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        self.model_fit = self.model.fit(disp=False)
        return self

    def predict(self, X_test: pd.DataFrame):
        """Make predictions using the fitted SARIMAX model.
        Args:
            X_test (pd.DataFrame): Exogenous variables for prediction.
        """
        forecast = self.model_fit.forecast(steps=len(X_test), exog=X_test)
        return forecast
    

class SARIMAXTune:
    def __init__(self, y_train: pd.DataFrame, y_test: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, params: dict):
        """Initialize with time series data and exogenous variables.
        Args:
            y_train (pd.DataFrame): Time series data for training.
            y_test (pd.DataFrame): Time series data for testing.
            X_train (pd.DataFrame): Exogenous variables for training.
            X_test (pd.DataFrame): Exogenous variables for testing.
            params (dict): Hyperparameter search space.
        """
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.params = params
        self.best_params = None

    def sarimax_objective(self, trial):
        # Regular SARIMA hyperparameters
        p = trial.suggest_int('p', self.params['p']['min'], self.params['p']['max'])  # autoregressive order
        d = trial.suggest_int('d', self.params['d']['min'], self.params['d']['max'])  # differencing order
        q = trial.suggest_int('q', self.params['q']['min'], self.params['q']['max'])  # moving average order
        P = trial.suggest_int('P', self.params['P']['min'], self.params['P']['max'])  # seasonal autoregressive order
        D = trial.suggest_int('D', self.params['D']['min'], self.params['D']['max'])  # seasonal differencing order
        Q = trial.suggest_int('Q', self.params['Q']['min'], self.params['Q']['max'])  # seasonal moving average order

        # Seasonal period as fixed hyperparameter
        s = self.params['s']  # e.g., around daily for 15-min resolution
        
        try:
            model = SARIMAXModel(params={
                'p': p,
                'd': d,
                'q': q,
                'P': P,
                'D': D,
                'Q': Q,
                's': s
            })
            results = model.fit(y_train=self.y_train, X_train=self.X_train)
            y_pred = results.predict(exog=self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            return rmse
        except:
            return np.inf

    def optimize(self):
        n_trials = self.params['n_trials']
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.sarimax_objective(trial), n_trials=n_trials)
        self.best_params = study.best_trial
        return self.best_params

    def retrained_model_with_best_params(self):
        """Retrain the SARIMAX model using the best hyperparameters found."""
        if self.best_params is not None:
            model = SARIMAXModel(params={
                'p': self.best_params['p'],
                'd': self.best_params['d'],
                'q': self.best_params['q'],
                'P': self.best_params['P'],
                'D': self.best_params['D'],
                'Q': self.best_params['Q'],
                's': self.best_params['s']
            })
            model.fit(y_train=self.y_train, X_train=self.X_train)
            return model
        return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Basic time series models for PV forecasting.")
    argparser.add_argument(
        "-basic_model", 
        type=str,
        choices=['ARIMA', 'SARIMA', 'SARIMAX'],
        default='SARIMAX',
        help="Basic time series forecasting models to use."
    )
    argparser.add_argument(
        "-execute_optimize",
        type=bool,  
        default=False,
        help="Whether to run hyperparameter optimization with Optuna."
    )
    args = argparser.parse_args()
    basic_model = args.basic_model
    execute_optimize = args.execute_optimize

    # Load params
    with open("src_pandas/params_modules/data_param.yaml", 'r') as file:
        data_params = yaml.safe_load(file)
    with open("src_pandas/params_modules/model_param.yaml", 'r') as file:
        model_params = yaml.safe_load(file)

    weather_params = data_params['weather_data']
    pv_params = data_params['pv_data']
    target_column = data_params['target_column']
    datetime_column = data_params['datetime_column']

    # Load data (weather + pv)
    weather_df = pd.read_csv(weather_params['input_path'])
    pv_df = pd.read_csv(pv_params['input_path'])

    # Merge weather + pv on datetime
    df = merge_dfs_on_datetime([weather_df, pv_df], datetime_col=datetime_column, how='inner')
    df[datetime_column] = pd.to_datetime(df[datetime_column])  # TODO: IMPORTANT! But can be fixed in data generation
    print(f"Merged data: {df.head()}")

    # Example Optuna study (1 month) ------------------- # TODO: WRITE FUNCTION FOR THIS
    y = df["pv_feed_in"].iloc[:96*300]
    X = df[["ghi","cloud_cover","temperature"]].iloc[:96*300]

    train_size = int(len(y)*0.8)
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]

    if basic_model == 'ARIMA':
        arima_param = model_params['ARIMA']
        if not execute_optimize:
            model = ARIMAModel(
                params={"p": arima_param['p']["default"], 
                        "d": arima_param['d']["default"], 
                        "q": arima_param['q']["default"]}
            )
            fitted_model = model.fit(y_train)
            y_pred = model.predict(steps=len(y_test))
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"ARIMA RMSE without tuning (test set): {rmse}")
        else:
            tuner = ARIMATune(y_train=y_train, y_test=y_test, params=arima_param)
            best_params = tuner.optimize()
            print("Best ARIMA params:", best_params)
            retrained_model = tuner.retrained_model_with_best_params()
            y_pred = retrained_model.predict(steps=len(y_test))
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"ARIMA RMSE after retraining with best params (test set): {rmse}")
    
    elif basic_model == 'SARIMAX':
        sarimax_param = model_params['SARIMAX']
        if not execute_optimize:
            model = SARIMAXModel(
                params={
                    "p": sarimax_param['p']["default"], 
                    "d": sarimax_param['d']["default"], 
                    "q": sarimax_param['q']["default"],
                    "P": sarimax_param['P']["default"], 
                    "D": sarimax_param['D']["default"], 
                    "Q": sarimax_param['Q']["default"],
                    "s": sarimax_param['s']  # fixed seasonal period
                }
            )
            fitted_model = model.fit(y_train, X_train)
            y_pred = fitted_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"SARIMAX RMSE without tuning (test set): {rmse}")
        else:
            tuner = SARIMAXTune(y_train=y_train, y_test=y_test, X_train=X_train, X_test=X_test, params=sarimax_param)
            best_params = tuner.optimize()
            print("Best SARIMAX params:", best_params)
            retrained_model = tuner.retrained_model_with_best_params()
            y_pred = retrained_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"SARIMAX RMSE after retraining with best params (test set): {rmse}")


