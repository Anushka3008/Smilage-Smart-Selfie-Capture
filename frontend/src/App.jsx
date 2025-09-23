// frontend/src/App.jsx

import React, { useState, useRef, useEffect } from "react";
import CameraFeed from "./components/CameraFeed.jsx";
import Controls from "./components/Controls.jsx";
import OverlayMessage from "./components/OverlayMessage.jsx";
import PredictionPanel from "./components/PredictionPanel.jsx";
import Benchmark from "./components/Benchmark.jsx";
import Gallery from "./components/Gallery.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";
import GalleryModal from "./components/GalleryModal.jsx";

function App() {
  // --- State Management ---
  const [isConnected, setIsConnected] = useState(false);
  const [predictions, setPredictions] = useState({});
  const [overlay, setOverlay] = useState("");
  const [galleryImages, setGalleryImages] = useState([]);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const [benchmarkResults, setBenchmarkResults] = useState(null);
  const [benchmarkProgress, setBenchmarkProgress] = useState(0);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    smileThreshold: 0.7,
    ageConfidenceThreshold: 0.5,
  });

  // --- Refs ---
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const overlayTimeoutRef = useRef(null);

  // --- Helper Functions ---
  const setOverlayMessage = (message, duration = 2000) => {
    if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current);
    setOverlay(message);
    overlayTimeoutRef.current = setTimeout(() => setOverlay(""), duration);
  };

  // --- API Calls for Gallery ---
  const fetchGalleryImages = async () => {
    try {
      const response = await fetch("/api/captures");
      if (!response.ok) throw new Error("Network response was not ok");
      const data = await response.json();
      const formattedImages = data.images.map(filename => ({
        filename,
        url: `/captures/${filename}`
      }));
      setGalleryImages(formattedImages);
    } catch (error) { console.error("Failed to fetch gallery images:", error); }
  };

  const handleDeleteImage = async (filename) => {
    try {
      await fetch(`/api/captures/${filename}`, { method: 'DELETE' });
      fetchGalleryImages(); // Refresh gallery
    } catch (error) { console.error("Failed to delete image:", error); }
  };

  const handleDeleteAllImages = async () => {
    try {
      if (window.confirm("Are you sure you want to delete all captured selfies?")) {
        await fetch(`/api/captures`, { method: 'DELETE' });
        fetchGalleryImages(); // Refresh gallery
      }
    } catch (error) { console.error("Failed to delete all images:", error); }
  };

  // --- Effects ---
  useEffect(() => {
    fetchGalleryImages(); // Fetch initial gallery images on component mount
  }, []);

  useEffect(() => {
    // Cleanup WebSocket and timers when the component unmounts
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current);
    };
  }, []);

  // --- WebSocket Connection & Event Handlers ---
  const startCamera = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/video`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => setIsConnected(true);

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.frame && canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        const img = new Image();
        img.src = "data:image/jpeg;base64," + data.frame;
        img.onload = () => {
          if (canvasRef.current) {
            canvasRef.current.width = img.width;
            canvasRef.current.height = img.height;
            ctx.drawImage(img, 0, 0, img.width, img.height);
          }
        };
      }
      
      if (data.predictions) setPredictions(data.predictions);
      
      if (data.capture) {
        setOverlayMessage("Selfie Captured! 📸");
        setTimeout(fetchGalleryImages, 500); // Refresh gallery after capture
      } else if (data.is_smiling) {
        if (overlay !== "Selfie Captured! 📸") {
          setOverlayMessage("Smile Detected! 🙂", 1500);
        }
      }
      
      if (data.benchmark_progress) setBenchmarkProgress(data.benchmark_progress);
      if (data.benchmark_results) {
        setBenchmarkResults(data.benchmark_results);
        setBenchmarkProgress(0);
      }
    };
    
    wsRef.current.onclose = () => {
      setIsConnected(false);
      const ctx = canvasRef.current?.getContext("2d");
      if (canvasRef.current) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    };
    
    wsRef.current.onerror = (error) => console.error("WebSocket Error: ", error);
  };
  
  const stopCamera = () => {
    if (wsRef.current) wsRef.current.close();
  };
  
  const handleManualCapture = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "manual_capture" }));
    }
  };

  const handleSettingChange = (event) => {
    const { name, value } = event.target;
    setSettings(prev => ({ ...prev, [name]: value }));
    
    if (wsRef.current?.readyState === WebSocket.OPEN && name === 'smileThreshold') {
        wsRef.current.send(JSON.stringify({ action: "update_threshold", value: parseFloat(value) }));
    }
  };

  const handleRunBenchmark = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setBenchmarkResults(null); // Clear previous results
      setBenchmarkProgress(0.01);
      wsRef.current.send(JSON.stringify({ action: "run_benchmark" }));
    }
  };

  // --- Render Method ---
  return (
    <div className="app-container">
      <div className="left-panel">
        <h1>Smilage Selfie Capture</h1>
        <div className="camera-container">
          <CameraFeed ref={canvasRef} />
          <OverlayMessage message={overlay} />
        </div>
        <Controls
          isConnected={isConnected}
          onStart={startCamera}
          onStop={stopCamera}
          onCapture={handleManualCapture}
          onSettingsClick={() => setIsSettingsOpen(true)}
        />
        <PredictionPanel predictions={predictions} />
      </div>

      <div className="right-panel">
        <Gallery images={galleryImages} onOpenGallery={() => setIsGalleryOpen(true)} />
        <Benchmark
          results={benchmarkResults}
          onRunBenchmark={handleRunBenchmark}
          progress={benchmarkProgress}
        />
      </div>

      {isGalleryOpen && (
        <GalleryModal
          images={galleryImages}
          onClose={() => setIsGalleryOpen(false)}
          onDelete={handleDeleteImage}
          onDeleteAll={handleDeleteAllImages}
        />
      )}

      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSettingChange={handleSettingChange}
      />
    </div>
  );
}

export default App;