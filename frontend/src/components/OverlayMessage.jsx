import React from "react";

export default function OverlayMessage({ message }) {
  if (!message) return null;
  return (
    <div
      id="overlayMessage"
      style={{
        position: "absolute",
        top: "20px",
        left: "50%",
        transform: "translateX(-50%)",
        backgroundColor: "#03dac6",
        color: "#000",
        padding: "10px 20px",
        borderRadius: "5px",
        fontWeight: "bold",
      }}
    >
      {message}
    </div>
  );
}
