import React, { forwardRef } from "react";

const CameraFeed = forwardRef((props, ref) => {
  return <canvas ref={ref} className="camera-feed" />;
});

export default CameraFeed;