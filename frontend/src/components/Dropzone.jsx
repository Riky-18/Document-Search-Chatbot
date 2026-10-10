import { useRef, useState } from 'react';

/**
 * Dropzone component handles click and drag-and-drop PDF uploads,
 * displaying file constraints and an inline status banner.
 */
export default function Dropzone({ onFileSelect, isProcessing, status }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleClick = () => {
    if (!isProcessing && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleKeyDown = (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !isProcessing) {
      e.preventDefault();
      handleClick();
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!isProcessing) {
      setIsDragging(true);
    }
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    if (!isProcessing) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    // Only reset dragging if cursor left the container
    if (e.currentTarget.contains(e.relatedTarget)) return;
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (isProcessing) return;

    const file = e.dataTransfer.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const handleInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
    // Reset so same file can be selected again
    e.target.value = '';
  };

  return (
    <div className="dropzone-container">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        style={{ display: 'none' }}
        onChange={handleInputChange}
        disabled={isProcessing}
        aria-hidden="true"
      />

      <div
        className={`dropzone ${isDragging ? 'dropzone-active' : ''} ${
          isProcessing ? 'dropzone-disabled' : ''
        }`}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        role="button"
        tabIndex={isProcessing ? -1 : 0}
        aria-label="Upload a PDF file. Maximum 20 Megabytes."
      >
        <div className="dropzone-icon-wrapper">
          <svg
            className="dropzone-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
            <path d="M12 12v9" />
            <path d="m8 16 4-4 4 4" />
          </svg>
        </div>

        <div className="dropzone-content">
          <span className="dropzone-title">Upload a PDF</span>
          <span className="dropzone-subtitle">Click or drag & drop</span>
          <span className="dropzone-limit">Max 20 MB</span>
        </div>
      </div>

      {/* Inline Upload Status Banner */}
      {status && status.type !== 'idle' && (
        <div
          className={`status-banner status-${status.type}`}
          role="status"
          aria-live="polite"
        >
          {status.type === 'indexing' && (
            <span className="spinner" aria-hidden="true" />
          )}

          {status.type === 'done' && (
            <svg
              className="status-icon"
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z"
                clipRule="evenodd"
              />
            </svg>
          )}

          {status.type === 'error' && (
            <svg
              className="status-icon"
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z"
                clipRule="evenodd"
              />
            </svg>
          )}

          <div className="status-text-block">
            <span className="status-title">
              {status.type === 'indexing'
                ? 'Indexing'
                : status.type === 'done'
                ? 'Done'
                : 'Upload failed'}
            </span>
            <span className="status-message">{status.message}</span>
          </div>
        </div>
      )}
    </div>
  );
}
