import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Пути к файлам
X_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_fev2.csv'
y_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_count.csv'
X_train_aux_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv'

# 2. Загрузка данных
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()
X_train_aux = pd.read_csv(X_train_aux_path)

X_train_aux['datetime'] = pd.to_datetime(X_train_aux['datetime'])
train_day = X_train_aux['datetime'].dt.day

# 3. Hold-out split
train_mask = train_day <= 15
val_mask = (train_day >= 16) & (train_day <= 19)

X_tr = X_train.loc[train_mask].reset_index(drop=True)
y_tr = y_train.loc[train_mask].reset_index(drop=True)

X_val = X_train.loc[val_mask].reset_index(drop=True)
y_val = y_train.loc[val_mask].reset_index(drop=True)

# 4. Objective
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 20.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# 5. Запуск Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('\n=== BEST XGBOOST RESULT ===')
print('Best RMSE:', study.best_value)
print('Best params:')
for k, v in study.best_params.items():
    print(f'{k}: {v}')