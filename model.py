import numpy as np
import tensorflow as tf
from keras import layers
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

path = 'train'
test_path = 'test'
classes = ['Tair', 'Saleka', 'None']
data = []
test = []
test_labels = []
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

j = 0
lab = 0
for file in os.listdir(test_path):
    if j > 2:
        lab = 1
    img_path = os.path.join(test_path, file)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    test.append(img)
    test_labels.append(lab)
    j+=1



random.shuffle(data)
X = []
Y = []
for x, y in data:
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)
test = np.array(test)
test_labels = np.array(test_labels)

# data = np.array(data)
# labels = np.array(labels)
# print(data.shape)
# print(labels.shape)
# print(data[0].shape)

# x_train, y_train = train_test_split(data, test_size=0.2, random_state=42)


# print(len(x_train))
# X = np.asarray(X).astype('float32').reshape((-1,1))
# Y = np.asarray(Y).astype('float32').reshape((-1,1))

# Define the CNN model
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
 
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X, Y, epochs=10, batch_size=16)

# Evaluate the model
test_loss, test_acc = model.evaluate(test, test_labels)
print('Test accuracy:', test_acc)
# pr = test[4]
# pr = pr[None, :]
# accuracy = model.predict(pr)
# print(accuracy)
model.save('cnnForFace.h5')