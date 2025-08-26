import cv2
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrapper import AgeDetector
from preprocessing import FaceDetector, detect_and_preprocess

age_detector = AgeDetector()
face_detector = FaceDetector(method='haar')  # Can also use 'dlib'

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and preprocess
    faces = detect_and_preprocess(frame, face_detector, model='age')

    for face_img, (x, y, w, h) in faces:
        age, conf = age_detector.detect_age(frame, (x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{age} ({conf:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()