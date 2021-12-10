# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:23:26 2021

@author: Swapnil Chatterjee
##FACE DETECTION IN WEBCAM
"""
import cv2
face_cascade = cv2.CascadeClassifier("HaarCascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("HaarCascades/haarcascade_eye.xml")

# Function to detect the eyes and faces of each image(frame) of the video from webcam
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            cv2.rectangle(roi_colour, (x_eye,y_eye), (x_eye+w_eye, y_eye+h_eye), (0, 255, 0), 2)
    return frame     

#Detecting the face recognition in webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(frame_gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()