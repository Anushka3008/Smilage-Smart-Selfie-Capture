import cv2
import time
import psutil
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrapper import EmotionDetector, AgeDetector

def benchmark(num_frames=100):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("(!) Could not open webcam.")
        return

    emotion_detector = EmotionDetector()
    age_detector = AgeDetector()

    timings = []
    emotion_scores = []
    cpu_usages = []
    memory_usages = []

    frame_count = 0
    print(f"[INFO] Running benchmark for {num_frames} frames...")

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Emotion detection
        start = time.time()
        emotion, score = emotion_detector.detect_emotion(frame)
        emotion_time = (time.time() - start) * 1000  # ms
        timings.append(emotion_time)
        if emotion:
            emotion_scores.append(score)

        # Age detection (full frame or first detected face)
        h, w = frame.shape[:2]
        face_box = (0, 0, w, h)  # Using full frame as face for now
        start = time.time()
        age, confidence = age_detector.detect_age(frame, face_box)
        age_time = (time.time() - start) * 1000  # ms

        # System usage
        cpu_usages.append(psutil.cpu_percent(interval=None))
        memory_usages.append(psutil.virtual_memory().percent)

        frame_count += 1

    cap.release()

    results = {
        "avg_emotion_time_ms": np.mean(timings),
        "min_emotion_time_ms": np.min(timings),
        "max_emotion_time_ms": np.max(timings),
        "avg_emotion_score": np.mean(emotion_scores) if emotion_scores else 0,
        "fps": 1000.0 / np.mean(timings),
        "cpu_usage_avg": np.mean(cpu_usages),
        "memory_usage_avg": np.mean(memory_usages),
    }

    return results


if __name__ == "__main__":
    results = benchmark(num_frames=50)
    print("\n--- Benchmark Results ---")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")