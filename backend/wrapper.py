# backend/wrapper.py

import cv2
import numpy as np
import os
from abc import ABC, abstractmethod
import time

# Optional ONNX import
try:
    import onnxruntime as ort
except Exception:
    ort = None


# -------------------------
# Base classes
# -------------------------
class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, face_img: np.ndarray):
        pass


class BaseEmotionModel(BaseModel):
    @abstractmethod
    def predict(self, face_img: np.ndarray):
        pass


class BaseAgeModel(BaseModel):
    @abstractmethod
    def predict(self, face_img: np.ndarray):
        pass


class BaseGenderModel(BaseModel):
    @abstractmethod
    def predict(self, face_img: np.ndarray):
        pass


# -------------------------
# Emotion model: FER+ ONNX
# -------------------------
class EmotionFERPlus(BaseEmotionModel):
    DEFAULT_EMOTIONS = [
        "neutral", "happiness", "surprise", "sadness",
        "anger", "disgust", "fear", "contempt"
    ]

    def __init__(self, model_path="models/emotion-ferplus.onnx", emotions=None, providers=None):
        super().__init__("FERPlus")
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.emotions = emotions or self.DEFAULT_EMOTIONS
        self.providers = providers
        self.load()

    def load(self):
        if ort is None:
            raise RuntimeError("onnxruntime is required for FERPlus.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"FER model not found: {self.model_path}")
        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers or ["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Failed to load FER ONNX model: {e}")

    def preprocess(self, face_img: np.ndarray):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        
        # MODIFIED: Removed normalization and histogram equalization as a test.
        # The model might perform better with raw pixel values.
        processed_input = resized.astype(np.float32)
        
        return processed_input.reshape(1, 1, 64, 64)

    def predict(self, face_img: np.ndarray):
        try:
            inp = self.preprocess(face_img)
            outputs = self.session.run(None, {self.input_name: inp})
            scores = outputs[0][0]
            
            # Apply softmax to get confidence as a probability
            prob = np.exp(scores) / np.sum(np.exp(scores))
            
            # NEW DEBUG PRINT: This is the most important step.
            # It will show us the confidence score for every emotion.
            print(f"[DEBUG Emotion Scores]: {list(zip(self.emotions, np.round(prob, 2)))}")

            idx = int(np.argmax(prob))
            label = self.emotions[idx]
            confidence = float(prob[idx])
            return label, confidence
        except Exception as e:
            print(f"[ERROR] FER predict failed: {e}")
            return "error", 0.0


# --- The rest of the file remains the same ---
# -------------------------
# Age model: Caffe DNN
# -------------------------
class AgeCaffeNet(BaseAgeModel):
    AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                   '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def __init__(self, proto="models/age_deploy.prototxt", model="models/age_net.caffemodel"):
        super().__init__("CaffeAgeNet")
        self.proto = proto
        self.model = model
        self.net = None
        self.load()

    def load(self):
        if not os.path.exists(self.proto) or not os.path.exists(self.model):
            raise FileNotFoundError(f"Age model/proto not found: {self.proto}, {self.model}")
        self.net = cv2.dnn.readNetFromCaffe(self.proto, self.model)

    def preprocess(self, face_img: np.ndarray):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        return blob

    def predict(self, face_img: np.ndarray):
        try:
            blob = self.preprocess(face_img)
            self.net.setInput(blob)
            preds = self.net.forward()[0]
            idx = int(np.argmax(preds))
            return self.AGE_BUCKETS[idx], float(preds[idx])
        except Exception as e:
            print(f"[ERROR] Age predict failed: {e}")
            return None, 0.0


# -------------------------
# Gender model: Caffe DNN
# -------------------------
class GenderCaffeNet(BaseGenderModel):
    GENDER_LIST = ['Male', 'Female']

    def __init__(self, proto="models/gender_deploy.prototxt", model="models/gender_net.caffemodel"):
        super().__init__("CaffeGenderNet")
        self.proto = proto
        self.model = model
        self.net = None
        self.load()

    def load(self):
        if not os.path.exists(self.proto) or not os.path.exists(self.model):
            raise FileNotFoundError(f"Gender model/proto not found: {self.proto}, {self.model}")
        self.net = cv2.dnn.readNetFromCaffe(self.proto, self.model)

    def preprocess(self, face_img: np.ndarray):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        return blob

    def predict(self, face_img: np.ndarray):
        try:
            blob = self.preprocess(face_img)
            self.net.setInput(blob)
            preds = self.net.forward()[0]
            idx = int(np.argmax(preds))
            return self.GENDER_LIST[idx], float(preds[idx])
        except Exception as e:
            print(f"[ERROR] Gender predict failed: {e}")
            return None, 0.0


# -------------------------
# ModelManager
# -------------------------
class ModelManager:
    def __init__(self):
        self.emotion_models = {}
        self.age_models = {}
        self.gender_models = {}
        self.active_emotion = None
        self.active_age = None
        self.active_gender = None

    # Registration
    def register_emotion_model(self, key: str, model: BaseEmotionModel):
        self.emotion_models[key] = model
        if self.active_emotion is None:
            self.active_emotion = model

    def register_age_model(self, key: str, model: BaseAgeModel):
        self.age_models[key] = model
        if self.active_age is None:
            self.active_age = model

    def register_gender_model(self, key: str, model: BaseGenderModel):
        self.gender_models[key] = model
        if self.active_gender is None:
            self.active_gender = model

    # Switching
    def switch_emotion_model(self, key: str):
        if key in self.emotion_models:
            self.active_emotion = self.emotion_models[key]
            return True
        return False

    def switch_age_model(self, key: str):
        if key in self.age_models:
            self.active_age = self.age_models[key]
            return True
        return False

    def switch_gender_model(self, key: str):
        if key in self.gender_models:
            self.active_gender = self.gender_models[key]
            return True
        return False

    # Predict helpers
    def predict_emotion(self, face_img: np.ndarray):
        if not self.active_emotion:
            return None, 0.0
        return self.active_emotion.predict(face_img)

    def predict_age(self, face_img: np.ndarray):
        if not self.active_age:
            return None, 0.0
        return self.active_age.predict(face_img)

    def predict_gender(self, face_img: np.ndarray):
        if not self.active_gender:
            return None, 0.0
        return self.active_gender.predict(face_img)