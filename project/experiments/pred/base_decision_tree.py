import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

X_train_base_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_base.csv'
X_test_base_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test_base.csv'
y_train_path      = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_count.csv'

X_train_aux_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train.csv'
X_test_aux_path   = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv'

pred_dir  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\preds'
pred_path = os.path.join(pred_dir, 'baseline_tree.csv')

os.makedirs(pred_dir, exist_ok=True)

X_train = pd.read_csv(X_train_base_path)
X_test = pd.read_csv(X_test_base_path)
y_train = pd.read_csv(y_train_path).squeeze()

X_train_aux = pd.read_csv(X_train_aux_path)
X_test_aux = pd.read_csv(X_test_aux_path)

X_train_aux['datetime'] = pd.to_datetime(X_train_aux['datetime'])
X_test_aux['datetime'] = pd.to_datetime(X_test_aux['datetime'])

train_day = X_train_aux['datetime'].dt.day
test_day = X_test_aux['datetime'].dt.day 

# 5. Hold-out split
# train: дни 1-15
# val:   дни 16-19
train_mask = train_day <= 15
val_mask = (train_day >= 16) & (train_day <= 19)

X_tr = X_train.loc[train_mask].reset_index(drop=True)
y_tr = y_train.loc[train_mask].reset_index(drop=True)

X_val = X_train.loc[val_mask].reset_index(drop=True)
y_val = y_train.loc[val_mask].reset_index(drop=True)

print(f"Train size: {len(X_tr)}")
print(f"Val size:   {len(X_val)}")

# 6. Baseline-модель
model = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Валидация
print(X_tr.columns)
model.fit(X_tr, y_tr)
val_pred = model.predict(X_val)

rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Hold-out RMSE (= RMSLE): {rmse_val:.6f}")

# Обучение на всём train
model_full = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model_full.fit(X_train, y_train)

# Предсказание на X_test
test_pred_log = model_full.predict(X_test)

# Обратное преобразование из log1p
test_pred = np.round(np.expm1(test_pred_log)).astype(int)
test_pred = np.maximum(test_pred, 0)

# Сохранение предсказаний
submission = pd.DataFrame({
    'datetime': X_test_aux['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': test_pred
})

submission.to_csv(pred_path, index=False)
