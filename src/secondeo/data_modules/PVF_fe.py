"""This module contains feature engineering functions for PV forecasting."""
import argparse
import pandas as pd
import numpy as np
import yaml

from src_pandas.utils import (
    merge_dfs_on_datetime,
    remove_nulls
)


def create_lagged_feature(
    df: pd.DataFrame, 
    datetime_col: str, 
    target_col: str, 
    min_lag: int,
    max_lag: int
) -> pd.DataFrame:
    """
    Create lagged (or forward if negative) feature for a time series target column.
    
    Parameters:
        df (pd.DataFrame): the input DataFrame
        datetime_col (str): name of datetime column
        target_col (str): name of target column
        min_lag (int): minimum lag to create (positive for lag, negative for forward), considering the availablity of the latest data
        max_lag (int): maximum lag to create (positive for lag, negative for forward)

    Returns:
        df_lags (pd.DataFrame): DataFrame with new lag feature
    """
    # Sort by datetime to ensure correct lagging
    df = df.sort_values(datetime_col).copy()

    # Create lag features
    for lag in range(min_lag, max_lag + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # # Drop rows with NaN values introduced by shifting
    # df = df.dropna().reset_index(drop=True)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    window: int, 
    stats: list[str] = ['mean', 'std']
):
    """
    Add rolling window features for a time series target column.
    
    Args:
        df (pd.DataFrame): pandas DataFrame
        datetime_col (str): name of datetime column
        target_col (str): name of target column
        window (int): lagged window size for rolling calculations
        stats (list[str]): list of statistics to compute, options include 'mean', 'std'
    
    Returns:
        df (pd.DataFrame): DataFrame with new rolling features
    """
    # Sort by datetime to ensure correct rolling calculations
    df = df.sort_values(datetime_col).copy()

    # Create rolling features
    rolling_window = df[target_col].rolling(window=window)

    if 'mean' in stats:
        df[f'{target_col}_rolling_mean_{window}'] = rolling_window.mean()
    
    if 'std' in stats:
        df[f'{target_col}_rolling_std_{window}'] = rolling_window.std()
    
    return df


def add_temporal_features(
        df: pd.DataFrame, 
        datetime_col: str
) -> pd.DataFrame:
    """
    Add temporal features useful for PV forecasting.
    Features added:
    - month-of-year sin/cos (annual cycle)
    - hour-of-day sin/cos (daily cycle)
    
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column
        datetime_col (str): Name of the datetime column

    Returns:
        pd.DataFrame: DataFrame with added temporal features
    """
    df = df.copy()

    # Ensure datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Month-of-year cycle (smooth annual variation)
    month_of_year = df[datetime_col].dt.month
    df['month_of_year_sin'] = np.sin(2 * np.pi * month_of_year / 12)
    df['month_of_year_cos'] = np.cos(2 * np.pi * month_of_year / 12)

    # Hour-of-day cycle (smooth daily variation), but in 15-minute resolution
    hour_of_day = df[datetime_col].dt.hour + df[datetime_col].dt.minute / 60
    df['hour_of_day_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering for PV forecasting.")
    parser.add_argument(
        '-dataset', 
        type=str, 
        choices=['pv', 'weather'],  # TODO: alarm: only 'pv' implemented now, do we need weather processing and feature engineering?
        default='pv',
        help='Dataset to use for feature engineering.'
    )
    args = parser.parse_args()
    dataset = args.dataset

    # Load params
    with open("src_pandas/params_modules/data_param.yaml", 'r') as file:
        data_params = yaml.safe_load(file)
    
    weather_params = data_params['weather_data']
    pv_params = data_params['pv_data']
    test_start_date = data_params['test_start_date']
    test_end_date = data_params['test_end_date']
    datetime_column = data_params['datetime_column']
    target_column = data_params['target_column']
    min_lag = data_params['min_lag']
    max_lag = data_params['max_lag']
    rolling_window = data_params['rolling_window']
    fe_save_path = data_params["feature_table_save_path"]

    # Load historical data and weather data (don't merge now)
    pv_df = pd.read_csv(pv_params['input_path'])
    weather_df = pd.read_csv(weather_params['input_path'])  # TODO: name load_file function or pay attention to the datetime format

    # Create lagged features
    pv_df = create_lagged_feature(
        df=pv_df, 
        datetime_col=datetime_column, 
        target_col=target_column, 
        min_lag=min_lag, 
        max_lag=max_lag
    )

    # Add rolling features
    pv_df = add_rolling_features(
        df=pv_df, 
        datetime_col=datetime_column, 
        target_col=target_column, 
        window=rolling_window
    )

    # Add temporal features
    pv_df = add_temporal_features(df=pv_df, datetime_col=datetime_column)

    pv_df = remove_nulls(df=pv_df)

    # Save to CSV
    pv_df.to_csv(pv_params['save_path'], index=False)
    print(f"PV feed-in engineered features saved to {pv_params['save_path']}")

    # Clean weather data
    weather_df[datetime_column] = pd.to_datetime(weather_df[datetime_column])  # TODO: IMPORTANT!
    # Select certain columns
    weather_df = weather_df[[datetime_column] + weather_params['feature_cols']]
    
    # Merge weather data to featured pv data
    df = merge_dfs_on_datetime([pv_df, weather_df], datetime_col=datetime_column, how='inner')
    df = remove_nulls(df=df)
    df.to_csv(fe_save_path, index=False)
    print(f"Feature table (weather + pv) saved to {fe_save_path}")

