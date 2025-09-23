import cv2
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrapper import EmotionDetector, AgeDetector

from wrapper import EmotionDetector, AgeDetector
# Initialize detectors
emotion_detector = EmotionDetector()
age_detector = AgeDetector()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Detect age
        age, age_conf = age_detector.detect_age(frame, (x, y, w, h))

        # Detect emotion
        emotion, emotion_score = emotion_detector.detect_emotion(frame[y:y+h, x:x+w])

        # Display results above face rectangle
        text = f"{emotion} ({emotion_score:.2f}), Age: {age} ({age_conf:.2f})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Emotion + Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()