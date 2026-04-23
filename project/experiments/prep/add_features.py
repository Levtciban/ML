import pandas as pd
import numpy as np

def get_features(data, is_train = False):
    data["datetime"] = pd.to_datetime(data["datetime"])

    data['zero_windspeed'] = (data['windspeed'] == 0).astype(int)
    data['month'] = data['datetime'].dt.month
    data['hour'] = data['datetime'].dt.hour
    data['year'] = data['datetime'].dt.year - 2011
    data['service_day'] = data['datetime'].dt.dayofyear + data['year'] * 365
    data['service_month'] = data['datetime'].dt.month + data['year'] * 12
    data['dayofweek'] = data['datetime'].dt.weekday

    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
    data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    data['is_night'] = ((data['hour'] >= 0) & (data['hour'] <= 4)).astype(int)
    data['morning_rush'] = (data['hour'] == 9).astype(int)
    data['evening_rush'] = ((data['hour'] >= 7) & (data['hour'] <= 9)).astype(int)
    
    data['temp_atemp_diff'] = data['temp'] - data['atemp']
    data['windspeed_temp'] = data["windspeed"] * data['temp']
    data['windspeed_humidity'] = data["windspeed"] * data['humidity']

    data['hour_workingday'] = data['hour'] * data['workingday']
    data['hour_sin_workingday'] = data['hour_sin'] * data['workingday']
    data['hour_cos_workingday'] = data['hour_cos'] * data['workingday']
    data['windspeed_sq'] = data['windspeed'] ** 2
    data['temp_sq'] = data['temp'] ** 2
    
    # data = data.drop('datetime', axis=1)
    # data = data.drop('temp', axis=1)
    

    # временные
    data['hour'] = data['datetime'].dt.hour
    data['day'] = data['datetime'].dt.day
    data['month'] = data['datetime'].dt.month
    data['year'] = data['datetime'].dt.year
    data['dayofweek'] = data['datetime'].dt.dayofweek
    data['dayofyear'] = data['datetime'].dt.dayofyear
    data['weekofyear'] = data['datetime'].dt.isocalendar().week.astype(int)
    data['quarter'] = data['datetime'].dt.quarter
    data['day_from_start'] = (data['datetime'] - pd.Timestamp('2011-01-01')).dt.days
    data = data.drop("datetime", axis=1)

    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    data['is_night'] = ((data['hour'] >= 0) & (data['hour'] <= 5)).astype(int)

    # погодные
    data['windspeed_sq'] = data['windspeed'] ** 2
    data['temp_hour_interaction'] = data['temp'] * data['hour']
    data['temp_atemp_diff'] = data['temp'] - data['atemp']
    data['humidity_windspeed'] = data['humidity'] * data['windspeed']
    data['windspeed_temp'] = data["windspeed"] * data['temp']
    
    data["is_weather_1"] = (data["weather"] == 1).astype(int)
    data["is_weather_2"] = (data["weather"] == 2).astype(int)
    data["is_weather_3"] = (data["weather"] >= 3).astype(int)
    data = data.drop("weather", axis=1)

    data["is_season_1"] = (data["season"] == 1).astype(int)
    data["is_season_2"] = (data["season"] == 2).astype(int)
    data["is_season_3"] = (data["season"] == 3).astype(int)
    data["is_season_4"] = (data["season"] == 4).astype(int)
    data = data.drop("season", axis=1)
    
    #циклические
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
    data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    data["morning_peak"] = ((data["hour"] >= 7) & (data["hour"] <= 9)).astype(int)
    data["evening_peak"] = ((data["hour"] >= 17) & (data["hour"] <= 19)).astype(int)
    data["holiday_or_weekend"] = (data["holiday"] | (1 - data["workingday"])).astype(int)

    if is_train:
        data = data.drop("count", axis=1)
        data = data.drop("casual", axis=1)
        data = data.drop("registered", axis=1)
    # data = data.drop('datetime', axis=1)
    return data

def get_target(data):
    return np.log1p(data["count"])

train = pd.read_csv(r"csvs\train.csv")
test = pd.read_csv(r"csvs\test.csv")

X_train = get_features(train, True)
y_train = get_target(train)
X_test = get_features(test)

X_train.to_csv("../csvs/X_train.csv", sep=',', index=False)
y_train.to_csv("../csvs/y_train.csv", sep=',', index=False)
X_test.to_csv(r"csvs\X_test.csv", sep=',', index=False)