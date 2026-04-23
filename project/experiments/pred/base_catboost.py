import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Пути к файлам
X_train_base_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_fev2.csv'
X_test_base_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test_fev2.csv'
y_train_path      = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_count.csv'

X_train_aux_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv'
X_test_aux_path   = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv'

pred_dir  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\preds'
pred_path = os.path.join(pred_dir, 'fe_v3_catboost.csv')

os.makedirs(pred_dir, exist_ok=True)

# Загрузка данных
X_train = pd.read_csv(X_train_base_path)
X_test = pd.read_csv(X_test_base_path)
y_train = pd.read_csv(y_train_path).squeeze()

X_train_aux = pd.read_csv(X_train_aux_path)
X_test_aux = pd.read_csv(X_test_aux_path)

# datetime только для разбиения и submission
X_train_aux['datetime'] = pd.to_datetime(X_train_aux['datetime'])
train_day = X_train_aux['datetime'].dt.day

# Hold-out split
# train: 1-15
# val:   16-19
train_mask = train_day <= 15
val_mask = (train_day >= 16) & (train_day <= 19)

X_tr = X_train.loc[train_mask].reset_index(drop=True)
y_tr = y_train.loc[train_mask].reset_index(drop=True)

X_val = X_train.loc[val_mask].reset_index(drop=True)
y_val = y_train.loc[val_mask].reset_index(drop=True)

print(f"Train size: {len(X_tr)}")
print(f"Val size:   {len(X_val)}")

# CatBoost модель
model = CatBoostRegressor(
    iterations=800,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
    subsample=0.7,
    thread_count=-1
)

# Валидация
model.fit(X_tr, y_tr)
val_pred = model.predict(X_val)

rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Hold-out RMSE (= RMSLE): {rmse_val:.6f}")

# Обучение на всём train
model_full = CatBoostRegressor(
    iterations=800,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
    subsample=0.8,
    thread_count=-1
)

model_full.fit(X_train, y_train)

# Предсказание на X_test
test_pred_log = model_full.predict(X_test)
test_pred = np.round(np.expm1(test_pred_log)).astype(int)
test_pred = np.maximum(test_pred, 0)

# Сохранение
submission = pd.DataFrame({
    'datetime': pd.to_datetime(X_test_aux['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': test_pred
})

submission.to_csv(pred_path, index=False)

print(f"\nФайл с предсказаниями сохранён в:\n{pred_path}")
print(submission.head())