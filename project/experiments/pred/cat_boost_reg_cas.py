import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

X_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_fev2.csv'
X_test_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test_fev2.csv'

y_casual_path     = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_casual.csv'
y_registered_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_registered.csv'
y_count_path      = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_count.csv'

X_train_aux_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv'
X_test_aux_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv'

pred_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\preds\catboost_best_cas_reg.csv'

X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)

y_cas = pd.read_csv(y_casual_path).squeeze()
y_reg = pd.read_csv(y_registered_path).squeeze()
y_cnt = pd.read_csv(y_count_path).squeeze()

X_train_aux = pd.read_csv(X_train_aux_path)
X_test_aux  = pd.read_csv(X_test_aux_path)

X_train_aux['datetime'] = pd.to_datetime(X_train_aux['datetime'])
train_day = X_train_aux['datetime'].dt.day

cat_features = [
    'season', 'holiday', 'workingday', 'weather', 
    'month', 'hour', 'weekday', 'year',
    'is_weekend', 'is_night', 'morning_rush', 'evening_rush'
]

for col in cat_features:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

train_mask = train_day <= 15
val_mask = (train_day >= 16) & (train_day <= 19)

X_tr  = X_train.loc[train_mask].reset_index(drop=True)
X_val = X_train.loc[val_mask].reset_index(drop=True)

y_cas_tr  = y_cas.loc[train_mask].reset_index(drop=True)
y_reg_tr  = y_reg.loc[train_mask].reset_index(drop=True)

y_cas_val = y_cas.loc[val_mask].reset_index(drop=True)
y_reg_val = y_reg.loc[val_mask].reset_index(drop=True)
y_cnt_val = y_cnt.loc[val_mask].reset_index(drop=True)

model_cas = CatBoostRegressor(
    iterations=1000, learning_rate=0.03, depth=5, 
    l2_leaf_reg=5, random_seed=42, verbose=0, 
    subsample=0.8, cat_features=cat_features, thread_count=-1
)

model_reg = CatBoostRegressor(
    iterations=1000, learning_rate=0.03, depth=6, 
    l2_leaf_reg=3, random_seed=42, verbose=0, 
    subsample=0.8, cat_features=cat_features, thread_count=-1
)

print("Обучение Casual...")
model_cas.fit(X_tr, y_cas_tr)
val_pred_cas = np.maximum(np.expm1(model_cas.predict(X_val)), 0)

print("Обучение Registered...")
model_reg.fit(X_tr, y_reg_tr)
val_pred_reg = np.maximum(np.expm1(model_reg.predict(X_val)), 0)

val_pred_cnt = val_pred_cas + val_pred_reg
val_pred_cnt_log = np.log1p(val_pred_cnt)

rmse_val = np.sqrt(mean_squared_error(y_cnt_val, val_pred_cnt_log))
print(f"\nHold-out RMSE (Casual + Registered): {rmse_val:.6f}")

print("\nОбучение финальных моделей на всём train...")
model_cas_full = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=5, random_seed=42, verbose=0, subsample=0.8, cat_features=cat_features, thread_count=-1)
model_reg_full = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=3, random_seed=42, verbose=0, subsample=0.8, cat_features=cat_features, thread_count=-1)

model_cas_full.fit(X_train, y_cas)
model_reg_full.fit(X_train, y_reg)

test_pred_cas = np.maximum(np.expm1(model_cas_full.predict(X_test)), 0)
test_pred_reg = np.maximum(np.expm1(model_reg_full.predict(X_test)), 0)

test_pred_cnt = np.round(test_pred_cas + test_pred_reg).astype(int)

submission = pd.DataFrame({
    'datetime': pd.to_datetime(X_test_aux['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': test_pred_cnt
})

submission.to_csv(pred_path, index=False)
print(f"Файл сохранён: {pred_path}")