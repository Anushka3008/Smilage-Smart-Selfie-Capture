import cv2
import time
import base64
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from wrapper import EmotionDetector, AgeDetector, GenderDetector

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize detectors
emotion_detector = EmotionDetector()
age_detector = AgeDetector()
gender_detector = GenderDetector()

cap = None
camera_on = False
smile_captured = False
MESSAGE_DURATION = 2.0  # seconds
message_text = ""
message_timer = 0
message_color = (0, 255, 255)

# Default colors
age_color = (255, 255, 0)
smile_color = (0, 255, 0)
not_smile_color = (0, 0, 255)
gender_color = (255, 0, 255)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    global cap, camera_on, smile_captured, message_text, message_timer

    await websocket.accept()
    cap = cv2.VideoCapture(0)
    camera_on = True

    while camera_on:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        overlay_message = None

        for (x, y, w, h) in faces:
            # Age detection
            age, _ = age_detector.detect_age(frame, (x, y, w, h))
            if age:
                cv2.putText(frame, f"Age: {age}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, age_color, 2)

            # Gender detection
            gender, gender_conf = gender_detector.detect_gender(frame, (x, y, w, h))
            if gender:
                cv2.putText(frame, f"Gender: {gender} ({gender_conf:.2f})", (x, y + h + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, gender_color, 2)

            # Emotion detection
            emotion, score = emotion_detector.detect_emotion(frame)
            if emotion:
                cv2.putText(frame, f"Emotion: {emotion}", (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Smile detection
            if emotion == "happy":
                cv2.putText(frame, f"Smile Detected", (x, y + h + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, smile_color, 2)
                if not smile_captured:
                    timestamp = int(time.time())
                    filename = f"captures/selfie_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    message_text = "Selfie Captured"
                    message_timer = time.time()
                    overlay_message = message_text
                    smile_captured = True
            else:
                smile_captured = False

            # Draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 204, 153), 2)

        # Overlay message
        if overlay_message and (time.time() - message_timer < MESSAGE_DURATION):
            text_size = cv2.getTextSize(overlay_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = int(frame.shape[0] * 0.1)
            cv2.rectangle(frame, (text_x - 10, text_y - 40),
                          (text_x + text_size[0] + 10, text_y),
                          message_color, -1)
            cv2.putText(frame, overlay_message, (text_x, text_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = base64.b64encode(buffer).decode("utf-8")

        await websocket.send_json({
            "frame": frame_bytes,
        })

    cap.release()