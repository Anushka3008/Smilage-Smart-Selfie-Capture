import cv2
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrapper import EmotionDetector
from preprocessing import FaceDetector, detect_and_preprocess

emotion_detector = EmotionDetector()
face_detector = FaceDetector(method='haar')  # Can also use 'dlib'

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and preprocess
    faces = detect_and_preprocess(frame, face_detector, model='emotion')

    for face_img, (x, y, w, h) in faces:
        emotion, score = emotion_detector.detect_emotion(face_img)
        if emotion:
            cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()