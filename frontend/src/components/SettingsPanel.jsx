// frontend/src/components/SettingsPanel.jsx

import React from "react";

const SettingsPanel = ({ isOpen, onClose, settings, onSettingChange }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>⚙️ Settings</h2>

        {/* NEW: Smile Threshold Slider moved here */}
        <div className="setting-item">
          <label htmlFor="smileThreshold">Smile Threshold</label>
          <div className="range-container">
            <input
              type="range"
              id="smileThreshold"
              name="smileThreshold"
              min="0"
              max="1"
              step="0.01"
              value={settings.smileThreshold}
              onChange={onSettingChange}
            />
            <span>{Number(settings.smileThreshold).toFixed(2)}</span>
          </div>
        </div>
        
        <div className="setting-item">
          <label htmlFor="ageThreshold">Age Confidence Threshold</label>
          <div className="range-container">
            <input
              type="range"
              id="ageThreshold"
              name="ageConfidenceThreshold"
              min="0"
              max="1"
              step="0.05"
              value={settings.ageConfidenceThreshold}
              onChange={onSettingChange}
            />
            <span>{Number(settings.ageConfidenceThreshold).toFixed(2)}</span>
          </div>
        </div>

        <div className="setting-item">
          <label htmlFor="cameraSelect">Camera Device</label>
          <select id="cameraSelect" name="selectedCamera" onChange={onSettingChange}>
            <option value="">Default Camera</option>
          </select>
        </div>
        
        <div className="setting-item">
          <label htmlFor="resolutionSelect">Image Resolution</label>
          <select id="resolutionSelect" name="resolution" onChange={onSettingChange}>
            <option value="640x480">640x480</option>
            <option value="1280x720">1280x720 (HD)</option>
            <option value="1920x1080">1920x1080 (Full HD)</option>
          </select>
        </div>

        <button className="close-button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default SettingsPanel;