import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

X_train = pd.read_csv(r'csvs\X_train.csv')
y_train = pd.read_csv(r'csvs\y_train.csv').squeeze()

train_raw = pd.read_csv(r'csvs\train.csv')
train_raw["datetime"] = pd.to_datetime(train_raw["datetime"])

assert len(X_train) == len(train_raw) == len(y_train)

day = train_raw["datetime"].dt.day

train_mask = day <= 15
val_mask = (day >= 16) & (day <= 19)

X_tr = X_train.loc[train_mask].reset_index(drop=True)
y_tr = y_train.loc[train_mask].reset_index(drop=True)

X_val = X_train.loc[val_mask].reset_index(drop=True)
y_val = y_train.loc[val_mask].reset_index(drop=True)

print(f"Train size: {len(X_tr)}")
print(f"Val size:   {len(X_val)}")

rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=40,
    n_jobs=-1,
    verbose=0
)

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.5,
    colsample_bytree=0.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

lgb = LGBMRegressor(
    verbosity=-1
)

catb = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
    subsample=0.4,
    od_type='Iter',
    od_wait=50
)

dtr = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

models = [
    ("Decision Tree", dtr),
    ("Random Forest", rf),
    ("XGBoost", xgb),
    ("LightGBM", lgb),
    ("CatBoost", catb),
]

results = []

for name, model in models:
    print(f"\nОбучение модели: {name}")
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    results.append((name, rmse))

    print(f"{name}: RMSE = {rmse:.5f}")

results_df = pd.DataFrame(results, columns=["Model", "RMSE"])
results_df = results_df.sort_values("RMSE").reset_index(drop=True)

print("\nИтоговые результаты:")
print(results_df.to_string(index=False))

results_df.to_csv(r"csvs\validation_results.csv", index=False)