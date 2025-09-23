import cv2
import time
import os
import numpy as np
from wrapper import ModelManager, EmotionFERPlus, AgeCaffeNet, GenderCaffeNet

# -------------------------
# Initialize model manager
# -------------------------
mgr = ModelManager()

# Load models (wrapped in try so it won’t crash if missing)
try: mgr.register_emotion_model("ferplus", EmotionFERPlus("models/emotion-ferplus.onnx"))
except: pass
try: mgr.register_age_model("caffe_age", AgeCaffeNet())
except: pass
try: mgr.register_gender_model("caffe_gender", GenderCaffeNet())
except: pass

# -------------------------
# Constants
# -------------------------
smile_threshold = 0.7
blur_threshold = 80.0  # Higher means more blurry
MESSAGE_DURATION = 2.0

# -------------------------
# Helper: Blur detection
# -------------------------
def variance_of_laplacian(image):
    """Return the Laplacian variance (focus measure). Low value => blurry."""
    return cv2.Laplacian(image, cv2.CV_64F).var()


# -------------------------
# Main camera loop (only runs if executed directly)
# -------------------------
def run_camera():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    smile_captured = False
    message_text = ""
    message_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()

            # Age
            age, age_conf = mgr.predict_age(face_img)
            if age:
                cv2.putText(frame, f"Age: {age}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Gender
            gender, gender_conf = mgr.predict_gender(face_img)
            if gender:
                cv2.putText(frame, f"Gender: {gender} ({gender_conf:.2f})", (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Emotion
            emotion, em_conf = mgr.predict_emotion(face_img)
            if emotion:
                cv2.putText(frame, f"Emotion: {emotion}", (x, y + h + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Smile detection
            if emotion == "happy" and em_conf >= smile_threshold:
                cv2.putText(frame, f"Smile Detected ({em_conf:.2f})", (x, y + h + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if not smile_captured:
                    timestamp = int(time.time())
                    filename = f"captures/selfie_{timestamp}.jpg"
                    os.makedirs("captures", exist_ok=True)
                    cv2.imwrite(filename, frame)
                    message_text = "Selfie Captured"
                    message_timer = time.time()
                    smile_captured = True
            else:
                smile_captured = False

            # Blur detection
            blur_score = variance_of_laplacian(face_img)
            if blur_score < blur_threshold:
                cv2.putText(frame, "Blurry", (x, y + h + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 204, 153), 2)

        # Overlay message
        if message_text and (time.time() - message_timer < MESSAGE_DURATION):
            overlay = frame.copy()
            alpha = 0.6
            text_size = cv2.getTextSize(message_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = int(frame.shape[0] * 0.1)
            cv2.rectangle(overlay, (text_x-10, text_y-40), (text_x+text_size[0]+10, text_y), (0,255,255), -1)
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            cv2.putText(frame, message_text, (text_x, text_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        cv2.imshow("Smilage Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------
# Run if called directly
# -------------------------
if __name__ == "__main__":
    run_camera()