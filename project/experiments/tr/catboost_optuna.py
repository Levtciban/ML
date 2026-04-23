import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

X_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_fev2.csv'
y_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_count.csv'
X_train_aux = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv')

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()
X_train_aux['datetime'] = pd.to_datetime(X_train_aux['datetime'])

cat_features = ['season', 'holiday', 'workingday', 'weather', 'month', 'hour', 'weekday', 'year', 'is_weekend', 'is_night', 'morning_rush', 'evening_rush']
for col in cat_features:
    X_train[col] = X_train[col].astype(int)

train_mask = X_train_aux['datetime'].dt.day <= 15
val_mask = (X_train_aux['datetime'].dt.day >= 16) & (X_train_aux['datetime'].dt.day <= 19)

X_tr, y_tr = X_train[train_mask], y_train[train_mask]
X_val, y_val = X_train[val_mask], y_train[val_mask]

def objective(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.07, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 20.0),
        'random_strength': trial.suggest_float('random_strength', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        
        'depth': 6, 
        'subsample': 0.8,
        
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 0,
        'thread_count': -1
    }

    model = CatBoostRegressor(**params, cat_features=cat_features)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print('Best RMSE:', study.best_value)
print('Best params:', study.best_params)