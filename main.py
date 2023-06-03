import cv2
import os
from keras.models import load_model


train_path = 'train'
model = load_model('googleNetForFace.h5')
# model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])

cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
    # Cut down frame to 250x250px
    frame = frame[120:120+250,200:200+250, :]
    print(frame.shape)
    pred=model.predict(frame)
    print(pred)
    # Show image back to screen
    #dsad
    cv2.imshow('Image', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()