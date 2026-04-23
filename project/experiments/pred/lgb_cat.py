import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

X_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_fev2.csv'
X_test_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test_fev2.csv'
y_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_count.csv'

X_train_aux_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv'
X_test_aux_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv'

pred_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\preds\lgbm_with_cats.csv'

X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).squeeze()

X_train_aux = pd.read_csv(X_train_aux_path)
X_test_aux  = pd.read_csv(X_test_aux_path)

X_train_aux['datetime'] = pd.to_datetime(X_train_aux['datetime'])
train_day = X_train_aux['datetime'].dt.day

cat_features = [
    'season', 'holiday', 'workingday', 'weather', 
    'month', 'hour', 'weekday', 'is_weekend', 
    'is_night', 'morning_rush', 'evening_rush', 'year'
]

for col in cat_features:
    X_train[col] = X_train[col].astype('category')
    X_test[col]  = X_test[col].astype('category')

train_mask = train_day <= 15
val_mask = (train_day >= 16) & (train_day <= 19)

X_tr  = X_train.loc[train_mask].reset_index(drop=True)
y_tr  = y_train.loc[train_mask].reset_index(drop=True)
X_val = X_train.loc[val_mask].reset_index(drop=True)
y_val = y_train.loc[val_mask].reset_index(drop=True)

model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)

print("Обучение LightGBM с типами category...")
model.fit(X_tr, y_tr)
val_pred = model.predict(X_val)

rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Hold-out RMSE: {rmse_val:.6f}")

model_full = LGBMRegressor(
    n_estimators=800, learning_rate=0.03, max_depth=6, 
    num_leaves=31, min_child_samples=20, subsample=0.8, 
    colsample_bytree=0.8, random_state=42, verbosity=-1
)
model_full.fit(X_train, y_train)

test_pred_log = model_full.predict(X_test)
test_pred = np.maximum(np.round(np.expm1(test_pred_log)).astype(int), 0)

submission = pd.DataFrame({
    'datetime': pd.to_datetime(X_test_aux['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': test_pred
})
submission.to_csv(pred_path, index=False)
print(f"Сохранено в {pred_path}")