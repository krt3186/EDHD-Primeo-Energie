"""This module explores feature importance for PV forecasting."""

# Note: Usually irradiance + clouds + temperature are the most important features for PV forecasting.
# However, the importance of features can vary based on the specific dataset and model used.
import argparse
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_regression

from src_pandas.utils import merge_dfs_on_datetime

def compute_correlations(df: pd.DataFrame, target_col: str, datetime_col: str, method: str = 'pearson') -> pd.Series:
    """
    Compute correlation of all features in df with target_col.
    
    Args:
        df (pd.DataFrame): pandas DataFrame
        target_col (str): name of target variable
        datetime_col (str): name of datetime column
        method (str): 'pearson' (linear) or 'spearman' (monotonic)

    Returns:
        pd.Series: sorted by correlation strength
    """
    # Remove datetime column if present
    df = df.drop(columns=[datetime_col], errors='ignore')
    # Compute correlations
    corrs = df.corr(method=method)[target_col].sort_values(ascending=False)
    return corrs


def compute_mutual_information(df: pd.DataFrame, target_col: str, datetime_col: str, features=None, random_state=42):
    """
    Compute mutual information (shared information) between features and a target variable.

    Args:
        df (pd.DataFrame): pandas DataFrame containing features and target
        target_col (str): name of the target variable
        datetime_col (str): name of datetime column
        features (list): list of feature names (default: all except target)
        random_state (int): reproducibility

    Returns:
        pd.Series: mutual information scores sorted in descending order
    """
    if features is None:
        features = [col for col in df.columns if col != target_col and col != datetime_col]

    X = df[features]
    y = df[target_col]
    
    mi = mutual_info_regression(X, y, random_state=random_state)
    mi_series = pd.Series(mi, index=features).sort_values(ascending=False)
    
    return mi_series


if __name__ == "__main__":
    # Note: for simplicity, only weather data is considered here.
    argparser = argparse.ArgumentParser(description="Feature importance analysis for PV forecasting.")
    argparser.add_argument(
        "-correlation_method", 
        type=str,
        choices=['pearson', 'spearman'],
        default='pearson',
        help="Correlation method to use."
    )
    args = argparser.parse_args()
    correlation_method = args.correlation_method

    # Load params
    with open("src_pandas/params_modules/data_param.yaml", 'r') as file:
        data_params = yaml.safe_load(file)
    
    weather_params = data_params['weather_data']
    pv_params = data_params['pv_data']
    target_column = data_params['target_column']
    datetime_column = data_params['datetime_column']

    # Load data (weather + pv)
    weather_df = pd.read_csv(weather_params['input_path'])
    pv_df = pd.read_csv(pv_params['input_path'])

    # # Preprocess datetime
    # weather_df[datetime_column] = pd.to_datetime(weather_df[datetime_column])
    # weather_df = weather_df.sort_values(datetime_column).reset_index(drop=True)

    # Merge weather + pv on datetime
    df = merge_dfs_on_datetime([weather_df, pv_df], datetime_col=datetime_column, how='inner')

    # Compute correlations
    corrs = compute_correlations(
        df=df, 
        target_col=target_column, 
        datetime_col=datetime_column, 
        method=correlation_method
    )
    print(f"Feature correlations with {target_column} ({correlation_method}):\n", corrs, "\n")

    # Compute mutual information
    mi_scores = compute_mutual_information(
        df=df,
        target_col=target_column,
        datetime_col=datetime_column
    )
    print(f"Mutual information scores with {target_column}:\n", mi_scores, "\n")

