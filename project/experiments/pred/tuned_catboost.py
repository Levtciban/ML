import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

X_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_train_fev2.csv'
X_test_path  = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test_fev2.csv'
y_train_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\y_count.csv'
X_test_aux   = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\prep\X_test.csv')

pred_path = r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\preds\catboost_optuna_safe.csv'

X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).squeeze()

cat_features = ['season', 'holiday', 'workingday', 'weather', 'month', 'hour', 'weekday', 'year', 'is_weekend', 'is_night', 'morning_rush', 'evening_rush']
for col in cat_features:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

final_params = {
    'iterations': 1000,
    'depth': 6,           
    'subsample': 0.8,   
    'learning_rate': 0.0638171654639343,
    'l2_leaf_reg': 11.399882902786985,
    'random_strength': 9.952622633958235,
    'bagging_temperature': 0.21702302145316427,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 100,
    'thread_count': -1,
    'cat_features': cat_features
}

print("Обучение финальной модели CatBoost...")
model = CatBoostRegressor(**final_params)
model.fit(X_train, y_train)

print("Предсказание...")
test_pred_log = model.predict(X_test)
test_pred = np.maximum(np.round(np.expm1(test_pred_log)).astype(int), 0)

submission = pd.DataFrame({
    'datetime': pd.to_datetime(X_test_aux['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': test_pred
})

submission.to_csv(pred_path, index=False)
print(f"ГОТОВО! Файл сохранен: {pred_path}")