import tensorflow as tf
from keras.applications import inception_v3
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

path = 'train'
classes = ['Tair', 'Saleka', 'None']
data = []
labels = []
i = 0
for filename in os.listdir(path):
    label_path = os.path.join(path, filename)
    for img in os.listdir(label_path):
        img_path = os.path.join(label_path, img)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        data.append([img, i])
        # labels.append(i)
    i+=1

random.shuffle(data)
X = []
Y = []
for x, y in data:
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

# data = np.array(data)
# labels = np.array(labels)
# print(data.shape)
# print(labels.shape)
# print(data[0].shape)

# x_train, y_train = train_test_split(data, test_size=0.2, random_state=42)


# print(len(x_train))
# X = np.asarray(X).astype('float32').reshape((-1,1))
# Y = np.asarray(Y).astype('float32').reshape((-1,1))

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


inc_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

inc_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])



inc_model.fit(X, Y, epochs=10, validation_split=0.3)

accuracy = inc_model.evaluate(X, Y)
model.save('googleNetForFace.h5')