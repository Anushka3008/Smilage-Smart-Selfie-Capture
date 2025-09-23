// frontend/src/components/PredictionPanel.jsx

import React from "react";

export default function PredictionPanel({ predictions }) {
  // Use placeholder data if predictions are not available yet
  const displayPredictions = {
    emotion: "-",
    age: "-",
    gender: "-",
    smile_score: 0,
    is_blurry: false,
    ...predictions,
  };

  return (
    <div className="card prediction-panel">
      <h2>Live Predictions</h2>
      <div className="prediction-content">
        <p>
          <strong>Emotion:</strong> {displayPredictions.emotion}
        </p>
        <p>
          <strong>Age:</strong> {displayPredictions.age}
        </p>
        <p>
          <strong>Gender:</strong> {displayPredictions.gender}
        </p>
        <p>
          <strong>Smile Score:</strong> {displayPredictions.smile_score.toFixed(2)}
        </p>
        <p>
          <strong>Image Quality:</strong>
          <span className={displayPredictions.is_blurry ? "blurry" : "clear"}>
            {displayPredictions.is_blurry ? "Blurry" : "Clear"}
          </span>
        </p>
      </div>
    </div>
  );
}