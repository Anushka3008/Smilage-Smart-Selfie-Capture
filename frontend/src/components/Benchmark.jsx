// frontend/src/components/Benchmark.jsx

import React from "react";

const Benchmark = ({ results, onRunBenchmark, progress }) => {
  return (
    <div className="card">
      <h2>📊 Benchmark</h2>
      <div className="benchmark-run-panel">
        <p>Test system performance over 100 frames.</p>
        <button onClick={onRunBenchmark} disabled={progress > 0}>
          {progress > 0 ? `Running... ${(progress * 100).toFixed(0)}%` : "Run Benchmark"}
        </button>
        {progress > 0 && <progress value={progress} max="1" />}
      </div>

      {results && (
        <div className="benchmark-results">
          <div className="benchmark-item">
            <p><strong>Avg CPU Usage:</strong> <span>{results.avg_cpu.toFixed(1)}%</span></p>
            <p><strong>Avg Memory Usage:</strong> <span>{results.avg_mem.toFixed(1)}%</span></p>
            <p><strong>Avg Frame Time:</strong> <span>{results.avg_frame_time_ms.toFixed(2)} ms</span></p>
            <p><strong>Frames Per Second (FPS):</strong> <span>{results.fps.toFixed(2)}</span></p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Benchmark;