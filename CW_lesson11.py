import pandas as pd #для таблиць
import numpy as np
import tensorflow as tf
from tensorflow import keras #для створення шарів
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder #перетворює текстові мітки в числа
import matplotlib.pyplot as plt #бібліотека для побудови графіків

df = pd.read_csv('data/figures.csv')
print(df.head())
#
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label']) #ще один стовпчик аби перевести в 0 та 1
#обирали елементи для навчання
X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

#створення моделі
model = keras.Sequential([layers.Dense(8, activation='relu', input_shape=(3,)),#число нейронів, далы функція для активації, далі скільки параметрів подається для навчання
                          layers.Dense(8, activation='relu'),
                          layers.Dense(8, activation='softmax')])         #означає що шари розташовані послідовно, один за одним
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#навчання
history = model.fit(X, y, epochs=200, verbose=0) #епоха - кількість проходів моделі по даним, чим більше епох, тим розумніша модель, вербоз - щоб не засоряти консоль непотрібною інформацією

#візуалізація навчання, створення графіка
plt.plot(history.history['loss'], label = 'Втрата (loss)')
plt.plot(history.history['accuracy'], label = 'Точність (accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Процес навчання')
plt.legend()
plt.show()

#проводення тестування
test = np.array([18, 16, 0])

#отримуэмо імовірність
pred = model.predict(test)
print(f'Імовірність по кожному класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')