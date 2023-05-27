import tensorflow as tf
from keras.applications import inception_v3
from sklearn.model_selection import train_test_split
import os
import cv2

path = 'train'
classes = ['Tair', 'Not Tair']
data = []

for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    data.append(img)

x_train, x_test, y_train, y_test = train_test_split(data, test_size=0.2, random_state=42)

# def model_training():
model = inception_v3.InceptionV3(weights='imagenet', include_top=False,
                                     input_shape=(250, 250, 3))
for layer in model.layers:
    layer.trainable = False

inc_model = tf.keras.models.Sequential()

inc_model.add(model)

inc_model.add(tf.keras.layers.Flatten())
inc_model.add(tf.keras.layers.Dense(256, activation='relu'))
inc_model.add(tf.keras.layers.BatchNormalization())
inc_model.add(tf.keras.layers.Dropout(0.4))

inc_model.add(tf.keras.layers.Dense(128, activation='relu'))
inc_model.add(tf.keras.layers.BatchNormalization())
inc_model.add(tf.keras.layers.Dropout(0.4))


inc_model.add(tf.keras.layers.Dense(2, activation='softmax'))

inc_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
inc_model.fit(x = x_train, y=y_train, validation_data= (x_test, y_test), epochs=10)

accuracy = model.evaluate(x_test, y_test)
model.save('googleNetForFace.h5')