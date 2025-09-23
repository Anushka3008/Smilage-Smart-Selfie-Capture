import cv2
import time
import psutil
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrapper import ModelManager, EmotionFERPlus, AgeCaffeNet, GenderCaffeNet

def benchmark(num_frames=50):
    """
    Benchmark Emotion, Age, and Gender models using webcam feed.
    Measures inference time, confidence scores, CPU and memory usage.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("(!) Could not open webcam.")
        return

    mgr = ModelManager()

    # Register models
    try:
        mgr.register_emotion_model("ferplus", EmotionFERPlus("models/emotion-ferplus.onnx"))
    except Exception as e:
        print(f"[WARN] EmotionFERPlus not registered: {e}")

    try:
        mgr.register_age_model("caffe_age", AgeCaffeNet("models/age_deploy.prototxt", "models/age_net.caffemodel"))
    except Exception as e:
        print(f"[WARN] AgeCaffeNet not registered: {e}")

    try:
        mgr.register_gender_model("caffe_gender", GenderCaffeNet("models/gender_deploy.prototxt", "models/gender_net.caffemodel"))
    except Exception as e:
        print(f"[WARN] GenderCaffeNet not registered: {e}")

    timings = {"emotion": [], "age": [], "gender": []}
    confidences = {"emotion": [], "age": [], "gender": []}
    cpu_usages = []
    memory_usages = []

    frame_count = 0
    print(f"[INFO] Running benchmark for {num_frames} frames...")

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        face_box = (0, 0, frame.shape[1], frame.shape[0])  # full frame for demo

        # Emotion
        start = time.time()
        em_label, em_conf = mgr.predict_emotion(frame)
        timings["emotion"].append((time.time() - start) * 1000)  # ms
        if em_label:
            confidences["emotion"].append(em_conf)

        # Age
        start = time.time()
        age_label, age_conf = mgr.predict_age(frame)
        timings["age"].append((time.time() - start) * 1000)
        if age_label:
            confidences["age"].append(age_conf)

        # Gender
        start = time.time()
        gender_label, gender_conf = mgr.predict_gender(frame)
        timings["gender"].append((time.time() - start) * 1000)
        if gender_label:
            confidences["gender"].append(gender_conf)

        # System usage
        cpu_usages.append(psutil.cpu_percent(interval=None))
        memory_usages.append(psutil.virtual_memory().percent)

        frame_count += 1

    cap.release()

    results = {
        "avg_emotion_time_ms": np.mean(timings["emotion"]),
        "min_emotion_time_ms": np.min(timings["emotion"]),
        "max_emotion_time_ms": np.max(timings["emotion"]),
        "avg_emotion_conf": np.mean(confidences["emotion"]) if confidences["emotion"] else 0,

        "avg_age_time_ms": np.mean(timings["age"]),
        "min_age_time_ms": np.min(timings["age"]),
        "max_age_time_ms": np.max(timings["age"]),
        "avg_age_conf": np.mean(confidences["age"]) if confidences["age"] else 0,

        "avg_gender_time_ms": np.mean(timings["gender"]),
        "min_gender_time_ms": np.min(timings["gender"]),
        "max_gender_time_ms": np.max(timings["gender"]),
        "avg_gender_conf": np.mean(confidences["gender"]) if confidences["gender"] else 0,

        "fps_emotion": 1000.0 / np.mean(timings["emotion"]) if timings["emotion"] else 0,
        "fps_age": 1000.0 / np.mean(timings["age"]) if timings["age"] else 0,
        "fps_gender": 1000.0 / np.mean(timings["gender"]) if timings["gender"] else 0,

        "cpu_usage_avg": np.mean(cpu_usages),
        "memory_usage_avg": np.mean(memory_usages),
    }

    return results

if __name__ == "__main__":
    results = benchmark(num_frames=30)
    print("\n[INFO] Benchmark completed.")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")