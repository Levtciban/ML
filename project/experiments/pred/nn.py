import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

test = pd.read_csv(r'csvs\test.csv')
X = pd.read_csv(r'csvs\X_train.csv')
y = pd.read_csv(r'csvs\y_train.csv')
X_test = pd.read_csv(r'csvs\X_test.csv')

scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='swish', input_shape=[input_shape]),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='swish'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='swish'),
        layers.Dense(1)
    ])
    return model

model = build_model(X_full_scaled.shape[1])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) 
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("Начинаем финальное обучение на 100% данных...")
history = model.fit(
    X_full_scaled, y,
    epochs=80,    
    batch_size=32,
    verbose=1
)

predictions = model.predict(X_test_scaled).flatten()

predictions = np.expm1(predictions)
predictions = np.round(predictions).astype(int)

predictions = np.maximum(predictions, 0)

submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count': predictions
})
submission.to_csv('submission_final.csv', index=False)
print("Финальный файл submission_final.csv создан!")

# График теперь покажет только Training Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Final Model Training (Full Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()