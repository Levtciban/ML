import pandas as pd
import numpy as np
import joblib as jl

import sys
sys.stdout = open('importances.txt', 'w', encoding='utf-8')

models = jl.load(r'files\models.joblib')
model_names = jl.load(r'files\model_names.joblib')
X_train = pd.read_csv(r"csvs\X_train.csv")

for model, name in zip(models, model_names):
    print(f"\n{'='*50}")
    print(f"Модель: {name}")
    print('='*50)

    importance = pd.Series(index=X_train.columns, data=0.0, dtype=float)

    if hasattr(model, 'get_booster'):  
        gain_dict = model.get_booster().get_score(importance_type='gain')
        importance = importance.add(pd.Series(gain_dict), fill_value=0)
    elif hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=X_train.columns)

    importance_sorted = importance.sort_values(ascending=False)

    top5 = importance_sorted.head(5)
    print("\nТоп-5 важных признаков:")
    for i, (feat, score) in enumerate(top5.items(), 1):
        print(f"{i}. {feat}: {score:.4f}")

    bottom5 = importance_sorted.tail(5)
    print("\nТоп-5 неважных признаков:")
    for i, (feat, score) in enumerate(bottom5[::-1].items(), 1): 
        print(f"{i}. {feat}: {score:.4f}")