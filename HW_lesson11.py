import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures.csv')
df_two = pd.read_csv('data/figures_new.csv')
print(df.head())
print()
print(df_two.head())

encoder = LabelEncoder()
encoder_two = LabelEncoder()

df['label_enc'] = encoder.fit_transform(df['label'])
df_two['label_enc'] = encoder_two.fit_transform(df_two['label'])

X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']
X_two = df_two[['area', 'perimeter', 'corners', 'area_to_perimeter']]
y_two = df_two['label_enc']


model = keras.Sequential([layers.Dense(8, activation='relu', input_shape=(3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(3, activation='softmax')])

model_two = keras.Sequential([layers.Dense(16, activation='relu', input_shape=(4,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(5, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_two.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=200, verbose=0)
history_two = model_two.fit(X_two, y_two, epochs=300, verbose=0)


plt.plot(history.history['loss'], label = 'Втрата (loss)')
plt.plot(history.history['accuracy'], label = 'Точність (accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Процес навчання, стара модель')
plt.legend()
plt.show(block=False)


plt.plot(history_two.history['loss'], label = 'Втрата (Loss)')
plt.plot(history_two.history['accuracy'], label = 'Точність (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Процес навчання, нова модель')
plt.legend()
plt.show()

test = np.array([[18, 16, 0]])
test_two = np.array([[18, 16, 0, 1.125]])

pred = model.predict(test, verbose=0)
pred_two = model_two.predict(test_two, verbose=0)

print(f'\nРезультати старої моделі:')
print(f'Імовірність по кожному класу: {pred[0]}')
print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])[0]}')

print(f'\nРезультати нової моделі:')
print(f'Імовірність по кожному класу: {pred_two[0]}')
print(f'Модель визначила: {encoder_two.inverse_transform([np.argmax(pred_two)])[0]}')