// frontend/src/components/Gallery.jsx

import React from "react";

const Gallery = ({ images, onOpenGallery }) => {
  return (
    <div className="card">
      <div className="gallery-preview-header">
        <h2>Gallery</h2>
        <button onClick={onOpenGallery} disabled={images.length === 0}>
          Open Gallery ({images.length})
        </button>
      </div>
      <div className="gallery-preview">
        {images.slice(0, 4).map((img) => (
          <img key={img.filename} src={img.url} alt={img.filename} />
        ))}
        {images.length > 4 && <div className="more-images-indicator">+{images.length - 4} more</div>}
        {images.length === 0 && <p>No selfies yet.</p>}
      </div>
    </div>
  );
};

export default Gallery;