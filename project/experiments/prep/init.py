import pandas as pd
import numpy as np

train = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\raw\train.csv')
test = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\raw\test.csv')

count = np.log1p(train['count'])
casual = np.log1p(train['casual'])
registered = np.log1p(train['registered'])

count.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_count.csv', index=False)
registered.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_registered.csv', index=False)
casual.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_casual.csv', index=False)

train = train.drop('registered', axis=1)
train = train.drop('casual', axis=1)
train = train.drop('count', axis=1)

test.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv', index=False)
train.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv', index=False)