import pandas as pd
import numpy as np

def add_features_v1(df):
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year - 2011

    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['hour_workingday'] = df['hour'] * df['workingday']

    df = df.drop('datetime', axis=1)

    return df

train = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv')
test = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv')

train = add_features_v1(train)
test = add_features_v1(test)

train.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_fev1.csv', index=False)
test.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test_fev1.csv', index=False)