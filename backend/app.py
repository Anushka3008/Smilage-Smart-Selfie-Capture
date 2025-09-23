# backend/app.py

import asyncio
import base64
import os
import time
from pathlib import Path

import cv2
import psutil
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from wrapper import (AgeCaffeNet, EmotionFERPlus, GenderCaffeNet,
                     ModelManager)

# ===================================================================
#  1. SETUP & CONFIGURATION
# ===================================================================

# --- File Paths & Directories ---
# Create a reliable, absolute path to the 'captures' directory
CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

app = FastAPI()

# --- CORS Middleware (for development) ---
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static File Serving ---
# This serves the actual image files from the /captures URL
app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

# --- Model Loading ---
model_mgr = ModelManager()
try:
    em_model = EmotionFERPlus("models/emotion-ferplus.onnx")
    model_mgr.register_emotion_model("ferplus", em_model)
    print("[INFO] EmotionFERPlus loaded successfully.")
except Exception as e:
    print(f"[WARN] EmotionFERPlus not loaded: {e}")
try:
    age_model = AgeCaffeNet(proto="models/age_deploy.prototxt", model="models/age_net.caffemodel")
    model_mgr.register_age_model("caffe_age", age_model)
    print("[INFO] AgeCaffeNet loaded successfully.")
except Exception as e:
    print(f"[WARN] AgeCaffeNet not loaded: {e}")
try:
    gender_model = GenderCaffeNet(proto="models/gender_deploy.prototxt", model="models/gender_net.caffemodel")
    model_mgr.register_gender_model("caffe_gender", gender_model)
    print("[INFO] GenderCaffeNet loaded successfully.")
except Exception as e:
    print(f"[WARN] GenderCaffeNet not loaded: {e}")

# --- Global Settings & Constants ---
BLUR_THRESHOLD = 100.0
SMILE_THRESHOLD = 0.7
DEFAULT_PREDICTIONS = {
    "emotion": "-", "age": "-", "gender": "-",
    "smile_score": 0.0, "is_blurry": False,
}

# ===================================================================
#  2. API ENDPOINTS (for Gallery Management)
# ===================================================================

@app.get("/api/captures")
async def get_captures():
    """Returns a list of all captured image filenames."""
    files = sorted(os.listdir(CAPTURES_DIR), reverse=True)
    return JSONResponse(content={"images": files})

@app.delete("/api/captures/{filename}")
async def delete_capture(filename: str):
    """Deletes a specific captured image."""
    try:
        os.remove(CAPTURES_DIR / filename)
        return JSONResponse(content={"status": "success", "filename": filename})
    except FileNotFoundError:
        return JSONResponse(content={"status": "error", "message": "File not found"}, status_code=404)

@app.delete("/api/captures")
async def delete_all_captures():
    """Deletes all images in the captures folder."""
    count = 0
    for filename in os.listdir(CAPTURES_DIR):
        os.remove(CAPTURES_DIR / filename)
        count += 1
    return JSONResponse(content={"status": "success", "deleted_count": count})

# ===================================================================
#  3. WEBSOCKET (for Live Video & Commands)
# ===================================================================

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # --- Shared state between concurrent tasks ---
    manual_capture_trigger = asyncio.Event()
    run_benchmark_flag = asyncio.Event()

    # --- Task 1: Receive messages from the frontend ---
    async def receive_messages():
        global SMILE_THRESHOLD
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")
                if action == "manual_capture":
                    manual_capture_trigger.set()
                elif action == "update_threshold":
                    SMILE_THRESHOLD = float(data.get("value", SMILE_THRESHOLD))
                elif action == "run_benchmark":
                    run_benchmark_flag.set()
        except WebSocketDisconnect:
            print("[INFO] Frontend disconnected.")

    # --- Task 2: Stream video and predictions to the frontend ---
    async def send_video():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("(!) Cannot open webcam")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        last_capture_time = 0
        CAPTURE_COOLDOWN = 3.0
        benchmark_data = {"frame_count": 0}
        BENCHMARK_DURATION_FRAMES = 100

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                start_time = time.time()
                is_benchmarking_active = benchmark_data["frame_count"] > 0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                predictions = DEFAULT_PREDICTIONS.copy()
                is_captured = False
                is_smiling_flag = False

                if len(faces) > 0:
                    # Process the largest face
                    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                    face_img = frame[y:y+h, x:x+w]

                    # Get predictions
                    emotion, em_conf = model_mgr.predict_emotion(face_img)
                    age, _ = model_mgr.predict_age(face_img)
                    gender, _ = model_mgr.predict_gender(face_img)
                    is_blurry = bool(cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < BLUR_THRESHOLD)
                    predictions.update({"emotion": emotion, "age": age, "gender": gender, "smile_score": em_conf if emotion == "happiness" else 0, "is_blurry": is_blurry})
                    
                    # Check for smile and capture conditions
                    is_smiling = emotion == "happiness" and em_conf >= SMILE_THRESHOLD
                    if is_smiling:
                        is_smiling_flag = True

                    can_capture_again = (time.time() - last_capture_time) > CAPTURE_COOLDOWN
                    if (is_smiling and can_capture_again) or manual_capture_trigger.is_set():
                        filename = f"selfie_{int(time.time())}.jpg"
                        cv2.imwrite(str(CAPTURES_DIR / filename), frame)
                        print(f"Selfie captured: {filename}")
                        last_capture_time, is_captured = time.time(), True
                        manual_capture_trigger.clear()

                    # Draw overlays
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 204, 153), 2)
                    cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame, f"Emotion: {emotion}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Gender: {gender}", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                # --- Benchmark Logic ---
                if run_benchmark_flag.is_set():
                    is_benchmarking_active = True
                    benchmark_data = {"times": [], "cpu": [], "mem": [], "frame_count": 1}
                    run_benchmark_flag.clear()

                if is_benchmarking_active:
                    benchmark_data["times"].append(time.time() - start_time)
                    benchmark_data["cpu"].append(psutil.cpu_percent())
                    benchmark_data["mem"].append(psutil.virtual_memory().percent)
                    benchmark_data["frame_count"] += 1
                    await websocket.send_json({"benchmark_progress": benchmark_data["frame_count"] / BENCHMARK_DURATION_FRAMES})

                    if benchmark_data["frame_count"] >= BENCHMARK_DURATION_FRAMES:
                        avg_time = sum(benchmark_data["times"]) / len(benchmark_data["times"])
                        results = {
                            "avg_cpu": sum(benchmark_data["cpu"]) / len(benchmark_data["cpu"]),
                            "avg_mem": sum(benchmark_data["mem"]) / len(benchmark_data["mem"]),
                            "avg_frame_time_ms": avg_time * 1000,
                            "fps": 1.0 / avg_time if avg_time > 0 else 0
                        }
                        await websocket.send_json({"benchmark_results": results})
                        benchmark_data["frame_count"] = 0  # End benchmark

                # --- Send Final Payload ---
                _, buffer = cv2.imencode(".jpg", frame)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
                payload = {"frame": frame_b64, "predictions": predictions, "is_smiling": is_smiling_flag}
                if is_captured:
                    payload["capture"] = True
                await websocket.send_json(payload)
                await asyncio.sleep(1/30)  # ~30 FPS

        except WebSocketDisconnect:
            print("[INFO] Frontend disconnected from video stream.")
        finally:
            cap.release()
            print("Camera released.")

    # --- Run both tasks concurrently ---
    await asyncio.gather(receive_messages(), send_video())

# ===================================================================
#  4. SERVE REACT APP (Must be last)
# ===================================================================
app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static-react")