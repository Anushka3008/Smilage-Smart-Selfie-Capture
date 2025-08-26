# AI-Based Image Analysis Tool for Smile & Age Prediction

This project is an **AI-powered application** that detects smiles and predicts age in real-time using a webcam. The application captures selfies automatically when a smile is detected and displays age and emotion with confidence scores. It also features a web-based interface using **FastAPI** for live video streaming.

---

## Features

- Real-time **face detection** using OpenCV Haar cascades.
- **Age prediction** using pretrained models.
- **Emotion detection** using FER.
- Automatic **selfie capture** when a smile is detected.
- Web interface for live camera feed.
- Customizable **smile threshold** for selfie capture.
- Visual feedback:
  - Age displayed above face bounding box.
  - Emotion displayed below face bounding box.
  - Pop-up message when a selfie is captured.

---

## Folder Structure

```bash
Smilage/
│
├── app.py # FastAPI application
├── wrapper.py # EmotionDetector & AgeDetector classes
├── scripts/
│ ├── test_age.py
│ ├── test_emotion.py
│ ├── selfie_capture.py
│ ├── realtime_inference.py
│ ├── preprocessing.py
│ ├── camera_test.py
│ ├── benchmark_models.py
│ └── benchmark_results.py
├── static/
│ ├── style.css # CSS for web UI
├── templates/
│ └── index.html # Web UI template
├── models/ # Pretrained model files
│ ├── age/
│ ├── emotion/
├── captures/ # Saved selfies
├── requirements.txt
└── docs/ # Documentation
```

---

## Installation

### 1. Clone the repository:
```bash
git clone --branch Anushka_Tripathi --single-branch https://github.com/Springboard-Internship-2025/AI-Based-Image-Analysis-Tool-for-Smile-Age-Prediction_August_2025.git
cd AI-Based-Image-Analysis-Tool-for-Smile-Age-Prediction_August_2025
```

### 2. Create and activate a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```
Make sure you have the models/ folder with pretrained models for age and emotion detection.

---

## Usage
### 1. Run the FastAPI app

```bash
uvicorn app:app --reload
```

### 2. Open the web interface

Open your browser and go to:

```bash
[http://127.0.0.1:8000](http://127.0.0.1:8000)
```

### 3. Features in the web UI
- Live camera feed in the center.
- Pop-up messages when a selfie is captured.
- Age displayed above face; emotion below face.
- Adjust smile detection threshold using the slider.


---

## Acknowledgements

OpenCV

FER

FastAPI framework for web UI

TensorFlow / Keras pretrained models