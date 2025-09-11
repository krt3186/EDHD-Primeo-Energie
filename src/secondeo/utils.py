# merge(); datetime handling; quality check; column selection
# Note: pd.merge_asof() is useful for time series joins with tolerance
"""This module contains utility functions for PV forecasting."""
import pandas as pd
import numpy as np

from functools import reduce


def check_missing_timestamps(
    df: pd.DataFrame, 
    datetime_col: str, 
    freq: str
) -> pd.DataFrame:
    """
    Check for missing timestamps in a DataFrame and return a DataFrame of missing timestamps.
    
    Args:
        df (pd.DataFrame): pandas DataFrame with datetime column
        datetime_col (str): name of datetime column
        freq (str): expected frequency (e.g., '15min', 'H', 'D')
    
    Returns:
        pd.DataFrame: DataFrame with missing timestamps
    """
    df = df.sort_values(datetime_col).copy()
    full_range = pd.date_range(start=df[datetime_col].min(), end=df[datetime_col].max(), freq=freq)
    missing_timestamps = full_range.difference(df[datetime_col])
    
    return pd.DataFrame(missing_timestamps, columns=[datetime_col])


def remove_nulls(
    df: pd.DataFrame, 
) -> pd.DataFrame:
    """
    Remove rows with null values from the DataFrame.
    """
    return df.dropna().reset_index(drop=True)


def merge_dfs_on_datetime(
    dfs: list[pd.DataFrame],
    datetime_col: str, 
    how: str = 'inner', 
) -> pd.DataFrame:
    """
    Merge two DataFrames on a datetime column using an asof merge.
    
    Args:
        dfs (list[pd.DataFrame]): list of DataFrames to merge
        datetime_col (str): name of datetime column
        how (str): type of merge - 'inner', 'left', 'right', 'outer'
    
    Returns:
        pd.DataFrame: merged DataFrame
    """
    if how not in ['inner', 'left', 'right', 'outer']:
        raise ValueError("Invalid merge type. Choose from 'inner', 'left', 'right', 'outer'.")
    
    # Use reduce() to iteratively merge all DataFrames in the list
    merged_df = reduce(
        lambda left, right: pd.merge(
            left, right, 
            on=datetime_col, 
            how=how,
            suffixes=('', '_dup')  # to handle overlapping columns
        ),
        dfs
    )
    # Drop duplicate columns created by suffixes
    dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
    merged_df = merged_df.drop(columns=dup_cols)

    return merged_df


def get_default_params(params: dict) -> dict:
    """
    Extract default parameters for model initialization.

    Args:
        params (dict): dictionary of parameters
    
    Returns:
        dict: dictionary of default parameters
    """
    basic_params = params['basic_params'] if 'basic_params' in params else {}
    int_params = params.get('int_params', {})
    float_params = params.get('float_params', {})

    basic_param_space = {key: value['value'] for key, value in basic_params.items()}
    int_param_space = {key: value['default'] for key, value in int_params.items()}
    float_param_space = {key: value['default'] for key, value in float_params.items()}
    default_params = {**basic_param_space, **int_param_space, **float_param_space}

    return default_params
