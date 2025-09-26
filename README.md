# 😄 Smilage Smart Selfie Capture : AI-Based Image Analysis Tool for Smile & Age Prediction 🧑‍💻

Smilage is a modern, full-stack AI application that analyzes a live webcam feed to perform real-time face detection, smile recognition, and age/gender prediction. It features an automatic selfie capture on smile detection, a fully interactive gallery, and a performance benchmark tool, all wrapped in a sleek web interface built with **React** and powered by a **FastAPI** backend.

---

## ✨ Features

- **🧠 Smart AI Predictions:**
  - 🟢 Real-time **face detection** using OpenCV.
  - 🎂 **Age & Gender prediction** using pre-trained Caffe models.
  - 😀 **Emotion detection** (including happiness/smile) using an ONNX model.
- **📸 Intelligent Capture:**
  - Automatic selfie capture when a smile is detected above a certain confidence.
  - Manual "Capture" button for full control.
- **🖼️ Interactive Gallery:**
  - A modal gallery to view all captured images.
  - **Download** and **Delete** options for each individual photo.
  - "Clear All" functionality to delete all images at once.
- **⚡ Performance & Settings:**
  - **Benchmark tool** to measure Frames Per Second (FPS), average CPU, and Memory usage.
  - An interactive **settings panel** to adjust the smile detection threshold and age confidence score in real-time.
- **🌐 Modern Architecture:**
  - A responsive Single-Page Application (SPA) frontend built with **React** and **Vite**.
  - A powerful Python backend using **FastAPI** to handle AI processing and serve the application.
  - Real-time video streaming from the backend to the frontend using **WebSockets**.

---

## 🛠️ Tech Stack

| Category      | Technology                                                                                                    |
|---------------|---------------------------------------------------------------------------------------------------------------|
| **Frontend** | `React`, `Vite`                                                                                               |
| **Backend** | `Python`, `FastAPI`, `Uvicorn`                                                                                |
| **AI / CV** | `OpenCV`, `ONNX Runtime`, `NumPy`                                                                             |
| **System** | `Psutil` (for CPU/Memory benchmark)                                                                           |

---

## 🗂️ Folder Structure

```
SMILAGE/
├── backend/
│   ├── captures/
│   ├── docs/
│   ├── models/
│   ├── scripts/
│   ├── app.py            
│   ├── requirements.txt   
│   └── wrapper.py 
│
├── frontend/
    ├── dist/
    ├── node_modules/
    ├── public/
    ├── src/
    │   ├── components/
    │   ├── App.jsx
    │   ├── index.jsx
    │   └── style.css
    ├── index.html
    ├── package-lock.json
    ├── package.json
    └── vite.config.js
├── venv/
├── .gitignore         
└── README.md
```

---

## 🛠 Installation

### 1. Clone the repository:
```bash
git clone https://github.com/Anushka3008/Smilage-Smart-Selfie-Capture
cd Smilage-Smart-Selfie-Capture
```

### 2. Set Up the Backend
```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install psutil
```

### 3. Set Up the Frontend
```bash
# Navigate to the frontend directory
cd ../frontend

# Install Node.js dependencies
npm install
```

---

## 🚀 Usage

This project can be run in two modes:

### Development Mode
(Use this when you are actively changing the code. The frontend and backend run on separate servers.)

1.  **Start the Backend Server:**
    ```bash
    # In the /backend directory
    uvicorn app:app --reload
    ```
2.  **Start the Frontend Server** (in a new terminal):
    ```bash
    # In the /frontend directory
    npm run dev
    ```
3.  Open your browser and go to `http://localhost:5173`.

### Production Mode (Unified App)
(Use this to run the final, compiled application from a single server.)

1.  **Build the Frontend:**
    ```bash
    # In the /frontend directory
    npm run build
    ```
2.  **Run the Backend Server:**
    ```bash
    # In the /backend directory
    uvicorn app:app
    ```
3.  Open your browser and go to `http://localhost:8000`.

---

## <caption> API Endpoints

The backend provides the following REST API endpoints for gallery management:

| Method   | Path                       | Description                      |
|----------|----------------------------|----------------------------------|
| `GET`    | `/api/captures`            | Get a list of all image filenames. |
| `DELETE` | `/api/captures/{filename}` | Delete a specific image.         |
| `DELETE` | `/api/captures`            | Delete all images.               |

---

## 🙏 Acknowledgements

- **[OpenCV](https://opencv.org/)** for its powerful computer vision capabilities.
- **[FastAPI](https://fastapi.tiangolo.com/)** for the high-performance backend framework.
- **[React](https://react.dev/)** and **[Vite](https://vitejs.dev/)** for the modern and reactive frontend.
- **[ONNX Runtime](https://onnxruntime.ai/)** for efficient AI model inference.
- **[Psutil](https://github.com/giampaolo/psutil)** for providing system usage metrics.
- The researchers and developers who created and shared the pre-trained Caffe and ONNX models used in this project.