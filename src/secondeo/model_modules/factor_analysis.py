"""This module contains factor analysis model."""
import pandas as pd
import numpy as np
import yaml
from scipy.ndimage import uniform_filter1d
from sklearn.isotonic import IsotonicRegression


# The main idea:
# 1. Try to quantify the capacity factor (CF) of the PV system using historical data.
# 2. Try to estimate the effect of weather on the PV output.
# 3. Consider periodic patterns (daily, seasonal) in the PV output.
# Combine these factors

def create_empirical_clearsky(
        df: pd.DataFrame, 
        datetime_col: str, 
        value_col: str, 
        clearsky_col: str = "G_cs",
        time_interval: int = 15,
        percentile: float = 98,
        minute_smooth: int = 2, 
        doy_smooth: int = 3):
    """
    Estimate empirical clear-sky irradiance from historical GHI/feed-in data.
    
    Args:
        df (pd.DataFrame): containing a datetime index and a column for GHI or feed-in data.
        datetime_col (str): The name of the datetime column.
        value_col (str): The name of the column to use for estimating clear-sky.
        time_interval (int): Time interval in minutes (default is 15).
        percentile (float): Upper percentile to approximate clear-sky.
        minute_smooth (int): Intraday smoothing window (in minutes).
        doy_smooth (int): Annual smoothing window (in days).

    Returns:
        clearsky_df (pd.DataFrame): index aligned with original df.index, columns ['G_cs'].
    """

    # Ensure timestamp is datetime and set as index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df[datetime_col]))

    # Get day of year and minute of day
    df = df.copy()
    df["doy"] = df.index.dayofyear
    df["minute"] = df.index.hour * 60 + df.index.minute

    # Calculate the clearsky matrix
    grouped = df.groupby(["doy", "minute"])[value_col].apply(
        lambda x: np.nanpercentile(x, percentile)
    )
    clearsky_matrix = grouped.unstack(level="minute").reindex(range(1, 367)).fillna(0)
    print("Clearsky matrix shape:", clearsky_matrix.shape)

    # Smooth the clearsky matrix
    # Minute direction
    clearsky_matrix = uniform_filter1d(clearsky_matrix, size=minute_smooth, axis=1, mode="nearest")
    # Annual direction
    clearsky_matrix = uniform_filter1d(clearsky_matrix, size=doy_smooth, axis=0, mode="nearest")

    print("Clearsky matrix after smoothing shape:", clearsky_matrix.shape)

    # Map back to original time series
    doy = df["doy"].values
    minute = df["minute"].values
    
    G_cs_vals = clearsky_matrix[doy-1, minute//time_interval]

    # Add the clearsky values to the dataframe
    df[clearsky_col] = G_cs_vals

    return df


def compute_capacity_factor(
        df_with_clearsky: pd.DataFrame, 
        value_col: str, 
        clearsky_col: str,
        proxy_quantile: float = 0.95):
    """
    Compute the capacity factor (CF) as the ratio of actual to clear-sky values.
    
    Args:
        df_with_clearsky (pd.DataFrame): DataFrame containing actual and clear-sky columns.
        value_col (str): The name of the actual value column, e.g. GHI or feed-in.
        clearsky_col (str): The name of the clear-sky value column.
        proxy_quantile (float): Quantile to use for daily proxy CF (default is 0.95).

    Returns:
        cf_series (pd.Series): Series of capacity factor values.
    """
    # Clearsky normalization
    df = df_with_clearsky.copy()
    df["CF"] = df[value_col] / (df[clearsky_col] + 1e-6)  # Avoid division by zero
    df["CF"] = df["CF"].clip(upper=1.0)  # Cap CF at 1.0, realistic max

    # Select daily proxy
    daily_proxy = (
        df.groupby(df.index.date)["CF"]
        .quantile(proxy_quantile)
        .rename("proxy")
        .to_frame()
    )
    daily_proxy.index = pd.to_datetime(daily_proxy.index)
    # daily_proxy['proxy_smooth'] = daily_proxy['proxy'].rolling(window=7, center=True, min_periods=1).mean()
    summer_months = daily_proxy.index.month.isin([6,7,8,9, 10])
    summer_proxy = daily_proxy['proxy'].where(summer_months)

    # Smooth and interpolate the proxy to get a capacity trend
    summer_proxy_smooth = summer_proxy.rolling(window=30, center=True, min_periods=1).max()
    daily_proxy['capacity_trend'] = np.interp(
        np.arange(len(daily_proxy)),
        summer_proxy_smooth.dropna().index.astype(int),
        summer_proxy_smooth.dropna().values
    )

    # Go back to original df index
    df['capacity_est'] = np.interp(
        df.index.astype('int64') // 10**9,                     
        pd.to_datetime(daily_proxy.index).astype('int64') // 10**9,  
        daily_proxy['capacity_trend'].values
    )

    return daily_proxy, df



if __name__ == "__main__":

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
    lstm_params = model_params['LSTM']

    # Load data
    df = pd.read_csv(feature_table_path)  # Pay attention to include time column ------------
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df = df.set_index(datetime_column)

    train_df = df[(df.index >= train_start_date) & (df.index <= train_end_date)]
    test_df = df[(df.index >= test_start_date) & (df.index <= test_end_date)]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Compute empirical clearsky
    clearsky_df = create_empirical_clearsky(
        train_df, 
        datetime_col=datetime_column, 
        value_col=target_column, 
        time_interval=15,
        clearsky_col="G_cs",
        percentile=98,
        minute_smooth=2, 
        doy_smooth=3
    )
    print("Clearsky df head:\n", clearsky_df)

    # Compute capacity factor
    daily_cf = compute_capacity_factor(
        df_with_clearsky=clearsky_df,
        value_col=target_column,
        clearsky_col="G_cs"
    )

    print("Daily capacity factor head:\n", daily_cf)

    # plot proxy to see the trend
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(daily_cf.index, daily_cf[1]["capacity_trend"], label="Daily Proxy CF (95th percentile)")
    plt.xlabel("Date")
    plt.ylabel("Capacity Factor")
    plt.title("Estimated Daily Capacity Factor from Historical Data")
    plt.legend()
    plt.grid()
    plt.show()

