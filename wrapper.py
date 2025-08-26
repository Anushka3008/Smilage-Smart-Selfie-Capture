import cv2
import numpy as np
from fer import FER

class EmotionDetector:
    def __init__(self):
        try:
            self.detector = FER(mtcnn=True)
        except Exception as e:
            print(f"[ERROR] Failed to initialize FER: {e}")
            self.detector = None

    def detect_emotion(self, frame):
        if self.detector is None or frame is None:
            return None, 0
        try:
            results = self.detector.detect_emotions(frame)
            if results:
                top_emotion, score = self.detector.top_emotion(frame)
                return top_emotion, score
        except Exception as e:
            print(f"[ERROR] Emotion detection failed: {e}")
        return None, 0
    
class AgeDetector:
    def __init__(self, model_path='models/age/age_net.caffemodel',
                 proto_path='models/age/age_deploy.prototxt'):
        try:
            self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load age model: {e}")
            self.net = None

        self.AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def detect_age(self, frame, face_box):
        if self.net is None or frame is None:
            return None, 0
        try:
            x, y, w, h = face_box
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False)
            self.net.setInput(blob)
            preds = self.net.forward()
            i = preds[0].argmax()
            age = self.AGE_BUCKETS[i]
            confidence = preds[0][i]
            return age, confidence
        except Exception as e:
            print(f"[ERROR] Age detection failed: {e}")
        return None, 0