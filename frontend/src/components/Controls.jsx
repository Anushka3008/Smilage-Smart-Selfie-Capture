// frontend/src/components/Controls.jsx

import React from "react";
import SettingsIcon from "./SettingsIcon";

export default function Controls({
  isConnected,
  onStart,
  onStop,
  onCapture,
  onSettingsClick,
}) {
  return (
    <div className="controls">
      <button onClick={onStart} disabled={isConnected}>
        ▶ Start Camera
      </button>
      <button onClick={onStop} disabled={!isConnected} className="stop">
        ⏹ Stop Camera
      </button>
      <button onClick={onCapture} disabled={!isConnected}>
        📸 Capture
      </button>
      <button onClick={onSettingsClick} className="settings-button" title="Settings">
        <SettingsIcon />
      </button>
    </div>
  );
}