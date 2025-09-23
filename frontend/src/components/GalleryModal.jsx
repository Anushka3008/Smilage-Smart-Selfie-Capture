import React from "react";

const GalleryModal = ({ images, onClose, onDelete, onDeleteAll }) => {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content gallery-modal" onClick={(e) => e.stopPropagation()}>
        <div className="gallery-header">
          <h2>📸 Gallery</h2>
          <button className="clear-all-btn" onClick={onDeleteAll}>Clear All</button>
          <button className="close-button" onClick={onClose}>&times;</button>
        </div>
        <div className="gallery-grid">
          {images.length === 0 ? (
            <p className="gallery-empty-message">No selfies captured yet.</p>
          ) : (
            images.map((img) => (
              <div key={img.filename} className="gallery-item">
                <img src={img.url} alt={img.filename} />
                <div className="gallery-item-overlay">
                  <a href={img.url} download={img.filename} className="gallery-btn download">Download</a>
                  <button onClick={() => onDelete(img.filename)} className="gallery-btn delete">Delete</button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default GalleryModal;