import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=40,
    n_jobs=-1,
    verbose=0
)

xgb = XGBRegressor(n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.5,
    colsample_bytree=0.5,
    random_state=42,
    n_jobs=-1)

lgb = LGBMRegressor(verbosity=-1)

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

X_train = pd.read_csv(r'csvs\X_train.csv')
y_train = pd.read_csv(r'csvs\y_train.csv').squeeze()

models = [rf, xgb, lgb, catb, dtr]
weights = [0, 0, 0, 1, 0]
model_names = ["random_forest", "xgb", "lgb", "catboost", "decision_tree"]
pred_train = np.zeros(len(y_train))
s = sum(weights)

for i, mod in enumerate(models):
    print(f"обучение {model_names[i]}")
    mod.fit(X_train, y_train)
    pred_train = pred_train + mod.predict(X_train) * weights[i] / s

rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))

print(f"RMSE на train: {rmse_train:.5f}")

joblib.dump(models, r'files\models.joblib')
joblib.dump(model_names, r'files\model_names.joblib')