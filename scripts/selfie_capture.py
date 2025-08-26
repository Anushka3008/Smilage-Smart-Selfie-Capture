import cv2
import time
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrapper import EmotionDetector, AgeDetector

# Initialize detectors
emotion_detector = EmotionDetector()
age_detector = AgeDetector()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

smile_captured = False  # Flag to prevent multiple captures per smile
message_timer = 0
message_text = ""
message_color = (0, 255, 255)
MESSAGE_DURATION = 2.0  # seconds

#age_color = (255, 204, 153)     # Sky blue color
age_color = (255, 255, 0)  
smile_color = (0, 255, 0)
not_smile_color = (0, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Age detection
        age, _ = age_detector.detect_age(frame, (x, y, w, h))
        cv2.putText(frame, f"Age: {age}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, age_color, 2)

        # Emotion detection
        emotion, score = emotion_detector.detect_emotion(frame)

        # Display smile message with confidence
        if emotion == "happy":
            cv2.putText(frame, f"Smile Detected ({score:.2f})", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, smile_color, 2)
            
            if not smile_captured:
                timestamp = int(time.time())
                filename = f"captures/selfie_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                message_text = "Selfie Captured"
                message_timer = time.time()
                smile_captured = True
        else:
            cv2.putText(frame, f"Smile: No ({score:.2f})", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, not_smile_color, 2)
            smile_captured = False

        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 204, 153), 2)

    # Semi-transparent popup message
    if message_text and (time.time() - message_timer < MESSAGE_DURATION):
        overlay = frame.copy()
        alpha = 0.6
        text_size = cv2.getTextSize(message_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = int(frame.shape[0] * 0.1)
        cv2.rectangle(overlay, (text_x - 10, text_y - 40), (text_x + text_size[0] + 10, text_y),
                      message_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, message_text, (text_x, text_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Selfie Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()