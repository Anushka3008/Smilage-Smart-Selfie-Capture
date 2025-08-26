# Face Extraction & Preprocessing Documentation

## 1. Face Detection

- **Method Used:** OpenCV Haar Cascades (optional comparison with dlib)
- **Code Location:** `preprocessing.py` → `FaceDetector` class
- **Details:**  
  - Converts frame to grayscale:  
    ```python
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ```
  - Detects faces using `detectMultiScale`:  
    ```python
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    ```
  - Returns a list of bounding boxes `(x, y, w, h)`

---

## 2. Face Extraction

- **Method Used:** Cropping detected bounding boxes from the original frame
- **Code Location:** `preprocessing.py` → `detect_and_preprocess` function
- **Details:**  
  - Crops valid faces from the frame:  
    ```python
    face_img = frame[y:y+h, x:x+w].copy()
    ```
  - Ensures only detected faces are passed to models

---

## 3. Preprocessing Pipeline

- **Purpose:** Prepare face images to match pretrained model input requirements
- **Steps Included:**  
  1. **Resizing:**  
     ```python
     face_resized = cv2.resize(face_img, input_size)
     ```
     - `input_size` depends on the model (e.g., 64x64 for FER, 227x227 for AgeNet)
  2. **Color Conversion:**  
     - Convert BGR → RGB if required by the model
  3. **Normalization:**  
     - Scale pixel values to `[0,1]` or mean-subtract according to model preprocessing

- **Code Location:** `preprocessing.py` → `detect_and_preprocess`

---

## 4. Usage Example

```python
from preprocessing import FaceDetector

detector = FaceDetector(method="haar")
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
faces = detector.detect_and_preprocess(frame)
for face_img, (x, y, w, h) in faces:
    # Pass face_img to age or emotion model
    age, conf = age_detector.detect_age(face_img)
    emotion, score = emotion_detector.detect_emotion(face_img)
```

---

## 5. Notes

- Haar cascades are faster but less accurate than dlib
- The preprocessing pipeline is modular:
    - Easy to add normalization, resizing, or color conversion for different models
- Supports multiple faces per frame
- Optimized for both live video frames and static images