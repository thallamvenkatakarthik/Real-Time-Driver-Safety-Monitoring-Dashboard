import cv2
import numpy as np
import dlib
import os
import serial
import time
from imutils import face_utils
from flask import Flask, render_template, Response, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load Dlib face detector and shape predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError("shape_predictor_68_face_landmarks.dat not found!")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error: Camera not detected!")

# Initialize GPS and Alcohol sensor (Modify COM port accordingly)
try:
    gps_serial = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=1)
except Exception as e:
    gps_serial = None
    print("GPS Module not detected!")

try:
    alcohol_serial = serial.Serial("/dev/ttyUSB1", baudrate=9600, timeout=1)
except Exception as e:
    alcohol_serial = None
    print("Alcohol Sensor not detected!")

# Tracking status
sleep, drowsy, active = 0, 0, 0
status, color = "", (0, 0, 0)

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f) + 1e-6  # Avoid division by zero
    ratio = up / (2.0 * down)
    return 2 if ratio > 0.25 else 1 if 0.21 <= ratio <= 0.25 else 0

def get_speed():
    """Extract speed from GPS module."""
    if gps_serial:
        try:
            line = gps_serial.readline().decode('utf-8', errors='ignore')
            if "$GPRMC" in line:
                data = line.split(",")
                if len(data) > 7 and data[7].isdigit():
                    return float(data[7]) * 1.852  # Convert knots to km/h
        except:
            return 0
    return 0

def detect_alcohol():
    """Detect alcohol presence."""
    if alcohol_serial:
        try:
            return float(alcohol_serial.readline().decode().strip()) > 0.5
        except:
            return False
    return False

def generate_frames():
    global sleep, drowsy, active, status, color
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1; drowsy = active = 0
                if sleep > 6:
                    status, color = "SLEEPING !!!", (255, 0, 0)
            elif left_blink == 1 or right_blink == 1:
                sleep = active = 0; drowsy += 1
                if drowsy > 6:
                    status, color = "Drowsy !", (0, 0, 255)
            else:
                sleep = drowsy = 0; active += 1
                if active > 6:
                    status, color = "Active :)", (0, 255, 0)

            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "camera_status": "Ready" if cap.isOpened() else "Offline",
        "drowsiness_status": status,
        "speed": get_speed(),
        "alcohol_detected": detect_alcohol()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0')


