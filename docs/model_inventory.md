# Model Inventory for Smilage Project

| Model Type       | Model Name / Source          | Input Size | Accuracy / Notes | Dependencies | Usage |
|-----------------|-----------------------------|------------|----------------|-------------|-------|
| Emotion/Smile   | FER (Python package, wrapper for FERPlus) | 48x48      | Good real-time detection (~65-70% top-1) | fer, moviepy, mtcnn | Real-time emotion/smile detection from webcam |
| Age Prediction  | AgeNet (OpenCV Caffe model) | 227x227    | Moderate accuracy | opencv-python | Age estimation from detected face |