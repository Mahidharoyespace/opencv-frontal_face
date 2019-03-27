import cv2
import numpy as np
#human_cascade = cv2.CascadeClassifier('/home/mahidhar/PycharmProject/untitled2/venv/lib/python3.6/site-packages/cv2/data/haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('/home/mahidhar/PycharmProject/untitled2/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/mahidhar/PycharmProject/untitled2/venv/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #human = human_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    #for (x, y, w, h) in human:
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        count = count + 1
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0*123
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
