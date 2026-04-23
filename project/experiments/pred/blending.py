import pandas as pd
import numpy as np

best_single = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\preds\fev2fx_catboost.csv')
best_ensemble = pd.read_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\preds\ensemble_blend.csv')

final_count = (best_single['count'] * 0.7) + (best_ensemble['count'] * 0.3)

final_count = np.round(final_count).astype(int)

submission = pd.DataFrame({
    'datetime': best_single['datetime'],
    'count': final_count
})

submission.to_csv(r'C:\Users\Home\Downloads\bike-sharing-demand\csvs\preds\pipipi.csv', index=False)

print("Финальный гибрид создан!")