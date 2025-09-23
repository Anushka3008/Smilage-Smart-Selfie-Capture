# model_manager_demo.py
"""
Small demo program: captures one frame from webcam (or uses synthetic) and runs predictions.
Saves an annotated image to captures/demo_out.jpg
"""

import cv2, os, time
import numpy as np
from wrapper import ModelManager, EmotionFERPlus, AgeCaffeNet, GenderCaffeNet

def main():
    mgr = ModelManager()
    try:
        mgr.register_emotion_model("ferplus", EmotionFERPlus("models/emotion-ferplus.onnx"))
    except Exception as e:
        print(f"[WARN] EmotionFERPlus not available: {e}")
    try:
        mgr.register_age_model("caffe_age", AgeCaffeNet(proto="models/age_deploy.prototxt", model="models/age_net.caffemodel"))
    except Exception as e:
        print(f"[WARN] AgeCaffeNet not available: {e}")
    try:
        mgr.register_gender_model("caffe_gender", GenderCaffeNet(proto="models/gender_deploy.prototxt", model="models/gender_net.caffemodel"))
    except Exception as e:
        print(f"[WARN] GenderCaffeNet not available: {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARN] No webcam. Using synthetic image.")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[WARN] Failed to capture; using synthetic")
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # For now use full frame as "face"
    face = frame.copy()
    em_label, em_conf = mgr.predict_emotion(face)
    age_label, age_conf = mgr.predict_age(face)
    gender_label, gender_conf = mgr.predict_gender(face)

    # Annotate and save
    text_lines = [
        f"Emotion: {em_label} ({em_conf:.2f})",
        f"Age: {age_label} ({age_conf:.2f})",
        f"Gender: {gender_label} ({gender_conf:.2f})"
    ]
    y0 = 30
    for i, line in enumerate(text_lines):
        y = y0 + i * 30
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    os.makedirs("captures", exist_ok=True)
    out_path = f"captures/demo_out_{int(time.time())}.jpg"
    cv2.imwrite(out_path, frame)
    print(f"[INFO] Saved annotated demo to {out_path}")

if __name__ == "__main__":
    main()