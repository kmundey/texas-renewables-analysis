import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clean_data(era5, ercot):
    """
    Aggregates regional features to totals, merges and cleans ERA5 and ERCOT data.

    Parameters:
        - era5 (pd.DataFrame): ERA5 regional weather data with datetime index
        - ercot (pd.DataFrame): ERCOT wind generation data with datetime index

    Returns: cleaned DataFrames (era5_dropped, ercot)
    """
    # Average regional features
    era5['temp_2m'] = era5[['temp_2m_south', 'temp_2m_north', 'temp_2m_east', 'temp_2m_west']].mean(axis=1)
    era5['wind_u_100m'] = era5[['wind_u_100m_south', 'wind_u_100m_north', 'wind_u_100m_east', 'wind_u_100m_west']].mean(axis=1)
    era5['wind_v_100m'] = era5[['wind_v_100m_south', 'wind_v_100m_north', 'wind_v_100m_east', 'wind_v_100m_west']].mean(axis=1)

    # Convert index to datetime
    era5['time'] = pd.to_datetime(era5['time'])
    era5.set_index('time', inplace=True)

    ercot['timestamp'] = pd.to_datetime(ercot['timestamp'])
    ercot.set_index('timestamp', inplace=True)

    # Check the date ranges
    common_datetime_range = era5.index.intersection(ercot.index)
    print(f'Common Date/Time Range: {common_datetime_range.min()} to {common_datetime_range.max()}\n')

    # Check what dates/times are missing
    ercot_notin_era = era5.index.difference(ercot.index)
    print(f'ERCOT dates not in ERA5 data: {ercot_notin_era}\n')
    
    # Drop rows in ERA5 data that are missing in ERCOT data
    era5_dropped = era5.drop(index=ercot_notin_era, errors='ignore')

    # Check that the dataframes match now
    print(f'Checking -- ERCOT dates not in ERA5 data: {era5_dropped.index.difference(ercot.index)}\n')

    return era5_dropped, ercot

def combine_dfs(era5_dropped, ercot):
    """
    Combines cleaned ERA5 weather data and ERCOT wind generation data into one DataFrame

    Parameters:
        - era5_dropped (pd.DataFrame): cleaned ERA5 data
        - ercot (pd.DataFrame): cleaned ERCOT data

    Returns: (pd.DataFrame): combined DataFrame of weather features and wind generation.
    """
    # Keep only the relevant system-wide features
    era5_small = era5_dropped[['temp_2m', 'wind_u_100m', 'wind_v_100m']]

    df = pd.concat([era5_small, ercot['wind_system']], axis=1)

    # Rename column
    df = df.rename(columns={'wind_system': 'wind_generation'})

    print(f'Dataframe Shape: {df.shape}')
    df.head()

    return df

def add_interaction_features(df):
    """
    Adds engineered features to the dataset to improve model performance.

    Features include:
    - Time-based sine/cosine features (hour of day, day of year)
    - Lag features for wind generation
    - Rolling mean and standard deviation over 3, 6, and 24-hour windows

    Parameters: df (pd.DataFrame): combined weather and wind generation data

    Returns: (pd.DataFrame): DataFrame that includes additional interaction features
    """
    # Time of day features
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Time of year features
    df['dayofyear'] = df.index.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    # Drop temporary columns
    df.drop(['hour', 'dayofyear'], axis=1, inplace=True)

    print("Added Feature: Time of Day and Time of Year")
    print("Description:   Since wind patterns often follow dinural cycles, it's best to have time as a function of sin/cos to avoid discontinuities\n")


    # Wind Generation Lag
    df['wind_gen_lag1'] = df['wind_generation'].shift(1)
    df['wind_gen_lag2'] = df['wind_generation'].shift(2)

    print("Added Feature: Wind Generation Lag")
    print("Description:   This helps the CNN 'remember' recent values\n")


    # Rolling means & standard deviations
    df['wind_gen_roll3'] = df['wind_generation'].rolling(window=3).mean()
    df['wind_gen_std3'] = df['wind_generation'].rolling(window=3).std()

    df['wind_gen_roll6'] = df['wind_generation'].rolling(window=6).mean()
    df['wind_gen_std6'] = df['wind_generation'].rolling(window=6).std()

    df['wind_gen_roll24'] = df['wind_generation'].rolling(window=24).mean()
    df['wind_gen_std24'] = df['wind_generation'].rolling(window=24).std()

    print("Added Feature: Rolling Means/Standard Deviations of Wind Generation")
    print("Description:   This helps smooth noisy data and capture trends\n")

    df.head()

    return df


def normalize(df):
    """
    Normalizes feature and target variables using Min-Max scaling.

    Parameters: df (pd.DataFrame): all features and wind generation target.

    Returns: X_scaled, y_scaled, scaler_y (tuple): scaled X and y (np.array), plus the MinMaxScaler() for y
    """

    # Make sure there's no missing data and data are ordered
    if df.isna().any().any():
        df = df.dropna().reset_index(drop=True)
        print('Dropped some missing values')
        print(f'Shape: {df.shape}\n')
    df = df.sort_index()

    feature_cols = ['temp_2m', 'wind_u_100m', 'wind_v_100m', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'wind_gen_lag1', 'wind_gen_lag2']

    # Initialize scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Scale the data
    X_scaled = scaler_X.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[['wind_generation']])
    
    print(f'X_scaled shape: {X_scaled.shape}')
    print(f'y_scaled shape: {y_scaled.shape}\n')

    return X_scaled, y_scaled, scaler_y


def create_time_windows(X_scaled, y_scaled):
    """
    Converts time series data into sliding windows (24 hours) for supervised learning.

    Parameters:
        - X_scaled (np.ndarray): Scaled feature array
        - y_scaled (np.ndarray): Scaled target array

    Returns: X_window, y_window (tuple): contains sliding windows of input and output data
    """
    # Look at 24 hours of data (window_size), forecasting the next hour (forecast_horizon)
    def create_sliding_windows(X, y, window_size=24, forecast_horizon=1):
        """
        Generates sliding windows of time series data. Each returned window contains a sequence 
        of `window_size` time steps from the input `X`, and the corresponding target value from 
        `y` that is `forecast_horizon` steps ahead.

        Parameters:
            - X (np.ndarray): Input feature array of shape (n_samples, n_features).
            - y (np.ndarray): Target array of shape (n_samples,).
            - window_size (int): Number of past time steps to include in each input sample.
            - forecast_horizon (int): Number of steps ahead to forecast.

        Returns: tuple of windows
            - X_windows (np.array): Shape (n_windows, window_size, n_features)
            - y_windows (np.array): Shape (n_windows,)
        """
        X_windows, y_windows = [], []

        # Iterate through the data to build each sliding window
        for i in range(window_size, len(X) - forecast_horizon + 1):
            X_windows.append(X[i - window_size:i])                    # Extract the sequence of features for this window
            y_windows.append(y[i + forecast_horizon - 1])             # Get the corresponding target value `forecast_horizon` steps ahead
        return np.array(X_windows), np.array(y_windows)

    # Convert into an array
    X_window, y_window = create_sliding_windows(X_scaled, y_scaled, window_size=24, forecast_horizon=1)

    print('Window Shapes:')
    print(f'X: {X_window.shape}, y: {y_window.shape}\n')

    return X_window, y_window