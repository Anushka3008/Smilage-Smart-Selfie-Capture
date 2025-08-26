import cv2
import time
from collections import deque
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrapper import EmotionDetector, AgeDetector

# Parameters
EMOTION_THRESHOLD = 0.7      # Only consider 'happy' if score > threshold
SMOOTHING_WINDOW = 5         # Number of frames to smooth predictions

# Initialize models
emotion_detector = EmotionDetector()
age_detector = AgeDetector()

# Initialize face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Buffers for smoothing
emotion_buffer = deque(maxlen=SMOOTHING_WINDOW)
age_buffer = deque(maxlen=SMOOTHING_WINDOW)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("(!) Cannot open webcam")
    exit()

# Real-time loop
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Age Detection
        try:
            age, age_conf = age_detector.detect_age(frame, (x, y, w, h))
        except Exception as e:
            age, age_conf = "Error", 0.0

        age_buffer.append(age)
        smoothed_age = max(set(age_buffer), key=age_buffer.count)

        # Emotion Detection
        emotion, score = emotion_detector.detect_emotion(frame)
        if emotion is not None:
            # Apply threshold
            if emotion == "happy" and score < EMOTION_THRESHOLD:
                emotion = "neutral"
            emotion_buffer.append(emotion)
        else:
            emotion_buffer.append("unknown")

        smoothed_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

        # Display results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"{smoothed_age}", (x, y-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"{smoothed_emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Real-time Age & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()