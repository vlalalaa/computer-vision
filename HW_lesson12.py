import tensorflow as tf

from tensorflow.keras import layers, models
import numpy as np

from tensorflow.keras.preprocessing import image

train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
        image_size=(128, 128), batch_size=32, label_mode='categorical')

test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
        image_size=(128, 128), batch_size=32, label_mode='categorical')
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


model = models.Sequential()
model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', input_shape=(128, 128, 3)))#фільтром визначаємо прості ознаки (лінії і контури)
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=50, validation_data=test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print(f'\nTest accuracy: {test_acc}')

class_name = ['apples', 'bananas', 'oranges']
img = image.load_img('images/img10.jpg', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array/255.0

img_array = np.expand_dims(img_array, axis=0)
predictions = model.predict(img_array)

predicted_index = np.argmax(predictions[0])
print(f'Prediction in classes {predictions[0]}')
print(f'Predicted class: {class_name[predicted_index]}')