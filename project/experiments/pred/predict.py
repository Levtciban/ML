import numpy as np
import pandas as pd
import joblib as jl

weights = [0, 0, 0, 1, 0]
s = sum(weights)
models = jl.load(r'files\models.joblib')

X_test = pd.read_csv(r'csvs\X_test.csv')
test = pd.read_csv(r'csvs\test.csv')

predictions = np.zeros(len(X_test))
for i, mod in enumerate(models):
    predictions = predictions + mod.predict(X_test) * weights[i] / s
    
predictions = (np.round(np.expm1(predictions))).astype(int)

submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count': predictions
})

submission.to_csv(r'csvs\bike_predictions.csv',
                  index=False,
                  date_format='%Y-%m-%d %H:%M:%S')



