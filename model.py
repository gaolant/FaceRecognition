import numpy as np
import tensorflow as tf
from keras import layers
import os
import cv2
import numpy as np
 model_test_continue
import matplotlib.pyplot as plt
import random
 face_begin

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
 model_test_continue
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

 model_on_realtime
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

        data.append(img)
        labels.append(i)
    i+=1

print(labels)
# tokenizer=Tokenizer()
# tokenizer.fit_on_texts(labels.Text)
data = np.array(data)
labels = np.array(labels)
print(data.shape)
print(labels.shape)
print(data[0].shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
# y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
face_begin

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
model_test_continue


inc_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])



inc_model.fit(x_train, y_train, batch_size = 32, epochs=10, validation_data=(x_test, y_test),shuffle=True)

accuracy = inc_model.evaluate(x_test, y_test)
model.save('googleNetForFace.h5')
 face_begin

inc_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])



inc_model.fit(X, Y, epochs=10, validation_split=0.3)
 model_test_continue
# accuracy = inc_model.evaluate(x_test, y_test)
model.save('googleNetForFace.h5')

accuracy = model.evaluate(x_test, y_test)
model.save('googleNetForFace.h5')
 face_begin
 face_begin
