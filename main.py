import cv2
import os

train_path = 'train'
i = 0

cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()

    # Cut down frame to 250x250px
    frame = frame[120:120+250,200:200+250, :]

    # Collect positives
    if cv2.waitKey(5) & 0XFF == ord('t'):
        # Create the unique file path 
        imgname = os.path.join(train_path, '{}.jpg'.format(i))
        i+=1
        # Write out positive image
        cv2.imwrite(imgname, frame)

    # Show image back to screen
    cv2.imshow('Image', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()