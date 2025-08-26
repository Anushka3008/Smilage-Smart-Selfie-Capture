import cv2
import numpy as np

class FaceDetector:
    def __init__(self, method='haar'):
        self.method = method
        if self.method == 'haar':
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        elif self.method == 'dlib':
            import dlib
            self.detector = dlib.get_frontal_face_detector()
        else:
            raise ValueError("Invalid detection method. Choose 'haar' or 'dlib'.")

    def detect_faces(self, frame):
        faces = []
        if self.method == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        elif self.method == 'dlib':
            import dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)
            faces = [(r.left(), r.top(), r.width(), r.height()) for r in rects]
        return faces


# Resize and normalize face for model input
def preprocess_face(face_img, target_size=(64,64), model='fer'):
    face_resized = cv2.resize(face_img, target_size)

    if model == 'fer':
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_resized = face_resized / 255.0
    elif model == 'age':
        # Age model expects BGR, mean subtraction done in wrapper
        pass

    return face_resized


# Detect faces and preprocess them for given model
# Returns: list of tuples (preprocessed_face, bounding_box)
def detect_and_preprocess(frame, face_detector, model='fer'):
    faces = face_detector.detect_faces(frame)
    preprocessed_faces = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        preprocessed = preprocess_face(face_img, model=model)
        preprocessed_faces.append((preprocessed, (x, y, w, h)))
    return preprocessed_faces