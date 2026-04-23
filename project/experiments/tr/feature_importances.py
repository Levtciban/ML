import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor

# =========================
# 1. Пути к данным
# =========================
ROOT = r'C:\Users\Home\Downloads\bike-sharing-demand'

X_train_path = os.path.join(ROOT, r'csvs\prep\X_train_fev2.csv')
y_train_path = os.path.join(ROOT, r'csvs\prep\y_count.csv')

save_dir = os.path.join(ROOT, r'figures\models')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'top10_feature_importance.pdf')

# 2. Загрузка данных
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()

# 3. Категориальные признаки
cat_features = [
    'season', 'holiday', 'workingday', 'weather',
    'month', 'hour', 'weekday', 'year',
    'is_weekend', 'is_night', 'morning_rush', 'evening_rush'
]

for col in cat_features:
    X_train[col] = X_train[col].astype(int)

# 4. Лучшая модель CatBoost
model = CatBoostRegressor(
    iterations=800,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
    subsample=0.8,
    cat_features=cat_features
)

model.fit(X_train, y_train)

# 5. Важность признаков
importances = pd.Series(
    model.get_feature_importance(),
    index=X_train.columns
).sort_values(ascending=False)

top10 = importances.head(10).sort_values(ascending=True)

print("Топ-10 важных признаков:")
print(top10[::-1])

# 6. График
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(top10.index, top10.values, color='#2874A6', edgecolor='black')
ax.set_xlabel('Важность признака')
ax.set_ylabel('Признак')
ax.set_title('Топ-10 наиболее важных признаков (CatBoost)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f'\nГрафик сохранён: {save_path}')