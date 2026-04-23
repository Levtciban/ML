import pandas as pd
import numpy as np

def add_time(df):
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    df = df.drop('datetime', axis=1)

    return df

train = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv')
test = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv')

train = add_time(train)
test = add_time(test)

train.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_base.csv', index=False)
test.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test_base.csv', index=False)
