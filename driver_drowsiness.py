import cv2
import numpy as np
import dlib
from imutils import face_utils
import os

# Check if shape_predictor_68_face_landmarks.dat exists
PREDICTOR_PATH = "E:\python files\pi\Driver-Drowsiness-Detection-master\shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    print("Error: shape_predictor_68_face_landmarks.dat not found!")
    exit()

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not detected!")
    exit()

# Load Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Tracking status
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f) + 1e-6  # Avoid division by zero
    ratio = up / (2.0 * down)
    return 2 if ratio > 0.25 else 1 if 0.21 <= ratio <= 0.25 else 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
        elif left_blink == 1 or right_blink == 1:
            sleep = active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
        else:
            sleep = drowsy = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Drowsiness Detector", frame)
    
    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
