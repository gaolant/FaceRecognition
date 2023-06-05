import cv2
import numpy as np
from keras.models import load_model
from model import *


train_path = 'train'
model = load_model('cnnForFace.h5')
# model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])

cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
    # Cut down frame to 250x250px
    fr = frame[120:120+250,200:200+250, :]
    
    fr = np.array(fr)
    fr = fr[None, :]
    pred=model.predict(fr)
    print(classes[int(pred)])
    # Show image back to screen
    cv2.imshow('Image', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()
