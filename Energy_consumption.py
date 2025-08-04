import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

#import warnings
#warnings.filterwarnings("ignore")


def load_and_clean_data(filepath):
    """
    Load power dataset and clean missing values and outliers.
    """
    df = pd.read_csv(filepath, sep=';', parse_dates={'datetime': ['Date', 'Time']},
                     na_values='?', infer_datetime_format=True, low_memory=False)
    df.set_index('datetime', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = interpolate_short_gaps(df)
    df = remove_outliers(df)
    return df


def interpolate_short_gaps(df, max_gap_minutes=60):
    """
    Missing Value Interpolation for period of less than one hour
    """
    threshold = pd.Timedelta(minutes=max_gap_minutes)
    for col in df.columns:
        mask = df[col].isna()
        gap_groups = (mask != mask.shift()).cumsum()
        nan_groups = mask[mask].groupby(gap_groups[mask])

        for _, indices in nan_groups.groups.items():
            ts_index = df.loc[indices].index
            gap_duration = ts_index[-1] - ts_index[0] + pd.Timedelta(minutes=1)
            if gap_duration <= threshold:
                df.loc[ts_index, col] = df[col].interpolate(method='time')[ts_index]
    return df


def remove_outliers(df, z_thresh=3):
    """
    Removes outliers using z-score method and interpolates them again.
    """
    for col in df.columns:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        df.loc[np.abs(z_scores) > z_thresh, col] = np.nan
    return df.interpolate(method='time')


def aggregate_temporally(df):
    """
    Aggregates data into hourly, daily, and weekly views.
    """
    agg_rules = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
    }
    return df.resample('H').mean(), df.resample('D').agg(agg_rules), df.resample('W').agg(agg_rules)


def plot_global_power(df_hourly, df_daily, df_weekly):
    """
    Plots aggregated global active power for different frequencies.
    """
    plt.figure(figsize=(12, 4))
    df_hourly['Global_active_power'].plot(title='Hourly Global Active Power', color='blue')
    plt.grid(), plt.ylabel('KW'), plt.show()

    df_daily['Global_active_power'].plot(title='Daily Global Active Power', color='green')
    plt.grid(), plt.ylabel('KW'), plt.show()

    df_weekly['Global_active_power'].plot(title='Weekly Global Active Power', color='red')
    plt.grid(), plt.ylabel('KW'), plt.show()


def plot_decomposition(df_daily, df_weekly):
    """
    Decomposes and plots seasonal components for daily and weekly data
    for the period 2009-01 to 2009-12
    """
    ts = df_daily['Global_active_power']['2009-01':'2009-12'].asfreq('D')
    ts1 = df_weekly['Global_active_power']['2009-01':'2009-12'].asfreq('W')

    result_daily = seasonal_decompose(ts, model='additive', period=7)
    result_weekly = seasonal_decompose(ts1, model='additive', period=4)

    result_daily.plot()
    plt.suptitle("Decomposition (Daily - 2009)", fontsize=14)
    plt.tight_layout(), plt.show()

    result_weekly.plot()
    plt.suptitle("Decomposition (Weekly - 2009)", fontsize=14)
    plt.tight_layout(), plt.show()

    plot_acf(ts.dropna(), lags=60)    # ACF for daily
    plt.title("ACF - Daily Global Active Power")
    plt.grid(), plt.show()

    plot_acf(ts1.dropna(), lags=30)     # ACF for weekly
    plt.title("ACF - Weekly Global Active Power")
    plt.grid(), plt.show()

def plot_rolling_stats(ts_daily, ts_weekly):
    """
    Plots rolling statistics to detect structural shifts.
    """
    rolling_mean_d = ts_daily.rolling(window=30).mean()
    rolling_std_d = ts_daily.rolling(window=30).std()

    plt.figure(figsize=(12, 5))
    plt.plot(ts_daily, label='Original', alpha=0.5)
    plt.plot(rolling_mean_d, label='30 days Rolling Mean', color='blue')
    plt.plot(rolling_std_d, label='30 days Rolling Std', color='red')
    plt.title("Structural Shifts - Daily"), plt.legend(), plt.grid(), plt.show()

    rolling_mean_w = ts_weekly.rolling(window=4).mean()
    rolling_std_w = ts_weekly.rolling(window=4).std()

    plt.figure(figsize=(12, 5))
    plt.plot(ts_weekly, label='Original', alpha=0.5)
    plt.plot(rolling_mean_w, label='4 weeks Rolling Mean', color='blue')
    plt.plot(rolling_std_w, label='4 weeks Rolling Std', color='red')
    plt.title("Structural Shifts - Weekly"), plt.legend(), plt.grid(), plt.show()


def engineer_features(df_daily, weather_path):
    """
    Adds time-based, lag, rolling, and weather features to the daily dataset.
    """
    with open(weather_path, 'r') as f:
        data = json.load(f)
    weather_df = pd.DataFrame(data['daily'])
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df.rename(columns={'time': 'date'}, inplace=True)
    weather_df.set_index('date', inplace=True)

    df_daily['day_of_week'] = df_daily.index.dayofweek # Time-Based Features   # 0 = Monday, 6 = Sunday
    df_daily['month'] = df_daily.index.month            # Time-Based Features   # 1 to 12
    df_daily['day'] = df_daily.index.day                 # Time-Based Features    # 1 to 31
    df_daily['lag_1'] = df_daily['Global_active_power'].shift(1)   # Lag Features
    df_daily['lag_7'] = df_daily['Global_active_power'].shift(7)   # Lag Features
    df_daily['rolling_mean_7'] = df_daily['Global_active_power'].rolling(7).mean() # Rolling Statistics
    df_daily['rolling_std_7'] = df_daily['Global_active_power'].rolling(7).std()    # Rolling Statistics
    df_daily['weekday_month_interaction'] = df_daily['day_of_week'] * df_daily['month']   # Interaction Terms  

    holiday_flags = holidays.France()   # Add Holiday Flags 
    df_daily['is_holiday'] = df_daily.index.to_series().apply(lambda x: 1 if x in holiday_flags else 0)
       # Merge weathor by date
    df_daily = df_daily.merge(weather_df, left_index=True, right_index=True, how='left')
    return df_daily


def train_models(df_daily, features_df):
    """
    Trains SARIMA, Prophet, XGBoost, and LightGBM models.
    Returns forecasts and actual values.
    """
    train = df_daily[:'2010-08-31']
    test = df_daily['2010-09-01':]
    y_train, y_test = train['Global_active_power'], test['Global_active_power']

    sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit()
    sarima_forecast = sarima.predict(start=y_test.index[0], end=y_test.index[-1])

    prophet_df = y_train.reset_index().rename(columns={'datetime': 'ds', 'Global_active_power': 'y'})
    prophet_model = Prophet(interval_width=0.9)
    prophet_model.fit(prophet_df)
    future = pd.DataFrame({'ds': y_test.index})
    prophet_forecast_df = prophet_model.predict(future).set_index('ds')
    prophet_forecast = prophet_forecast_df['yhat']
    prophet_lower = prophet_forecast_df['yhat_lower']
    prophet_upper = prophet_forecast_df['yhat_upper']

    x_train = features_df.loc[train.index].drop(columns='Global_active_power')
    x_test = features_df.loc[test.index].drop(columns='Global_active_power')
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1).fit(x_train, y_train)
    lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1).fit(x_train, y_train)

    xgb_forecast = pd.Series(xgb_model.predict(x_test), index=y_test.index)
    lgbm_forecast = pd.Series(lgbm_model.predict(x_test), index=y_test.index)

    return y_test, sarima_forecast, prophet_forecast, prophet_lower, prophet_upper, xgb_forecast, lgbm_forecast


def evaluate_models(y_test, sarima, prophet, xgb, lgbm, prophet_lower, prophet_upper):
    """
    Evaluates all models using MAE/RMSE and plots residuals and forecast.
    """
    def eval_metric(true, pred, name):
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        print(f"{name:<10} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    print("\n*** Model Evaluation ***")
    eval_metric(y_test, sarima, "SARIMA")
    eval_metric(y_test, prophet, "Prophet")
    coverage = ((y_test >= prophet_lower) & (y_test <= prophet_upper)).mean()
    print(f"Prophet 90% CI Coverage: {coverage * 100:.2f}%")
    eval_metric(y_test, xgb, "XGBoost")
    eval_metric(y_test, lgbm, "LightGBM")

    residuals = {
        "SARIMA": y_test - sarima,
        "Prophet": y_test - prophet,
        "XGBoost": y_test - xgb,
        "LightGBM": y_test - lgbm,
    }

    plt.figure(figsize=(12, 6))
    for i, (name, resid) in enumerate(residuals.items(), 1):
        plt.subplot(2, 2, i)
        plt.plot(resid)
        plt.axhline(0, linestyle='--', color='gray')
        plt.title(f"Residuals: {name}")
    plt.tight_layout(), plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(y_test, label="Actual", color='black')
    plt.plot(sarima, label="SARIMA", linestyle='--')
    plt.plot(prophet, label="Prophet", linestyle=':')
    plt.fill_between(prophet.index, prophet_lower, prophet_upper, alpha=0.2, label="Prophet 90% CI")
    plt.plot(xgb, label="XGBoost", linestyle='-.')
    plt.plot(lgbm, label="LightGBM", linestyle='--')
    plt.title("Forecast Comparison")
    plt.legend(), plt.grid(), plt.tight_layout(), plt.show()


def main():
    raw_path = r"D:\power project\household_power_consumption.txt"
    weather_path = r"D:\power project\weather.json"
    df_raw = load_and_clean_data(raw_path)
    df_hourly, df_daily, df_weekly = aggregate_temporally(df_raw)
    df_daily = engineer_features(df_daily, weather_path)

    plot_global_power(df_hourly, df_daily, df_weekly)
    plot_decomposition(df_daily, df_weekly)
    plot_rolling_stats(df_daily['Global_active_power'], df_weekly['Global_active_power'])

    y_test, sarima, prophet, lower, upper, xgb, lgbm = train_models(df_daily, df_daily)
    evaluate_models(y_test, sarima, prophet, xgb, lgbm, lower, upper)


if __name__ == "__main__":
    main()
