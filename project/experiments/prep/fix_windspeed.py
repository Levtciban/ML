import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv')
X_test = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv')

X_train['datetime'] = pd.to_datetime(X_train['datetime'])
X_test['datetime'] = pd.to_datetime(X_test['datetime'])
X_test['wind_was_zero'] = (X_test['windspeed'] == 0).astype(int)
X_train['wind_was_zero'] = (X_train['windspeed'] == 0).astype(int)
X_train['hour'] = X_train['datetime'].dt.hour
X_test['hour'] = X_test['datetime'].dt.hour
X_train['month'] = X_train['datetime'].dt.month
X_test['month'] = X_test['datetime'].dt.month

train_known = X_train[X_train['windspeed'] != 0]
train_unknown = X_train[X_train['windspeed'] == 0]

test_known = X_test[X_test['windspeed'] != 0]
test_unknown = X_test[X_test['windspeed'] == 0]

wind_train = pd.concat([train_known, test_known], axis=0)

y_wind = wind_train['windspeed']

features = ['season', 'weather', 'humidity', 'temp', 'atemp', 'hour', 'month']

X_wind = wind_train[features]

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_wind, y_wind)

preds_train = model.predict(train_unknown[features])
X_train.loc[train_unknown.index, 'windspeed'] = preds_train

preds_test = model.predict(test_unknown[features])
X_test.loc[test_unknown.index, 'windspeed'] = preds_test

X_train.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_wind_fixed.csv', index=False)
X_test.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test_wind_fixed.csv', index=False)
