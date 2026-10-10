import { useState } from 'react';
import Dropzone from './Dropzone';

/**
 * Sidebar component containing the navy title bar, dropzone,
 * indexed document cards, inset stats line, and clear confirmation flow.
 */
export default function Sidebar({
  stats,
  status,
  isProcessing,
  onFileSelect,
  onClear,
  filePageCounts = {},
  isOpenMobile,
  onCloseMobile,
}) {
  const [isConfirmingClear, setIsConfirmingClear] = useState(false);
  const hasFiles = stats.files && stats.files.length > 0;

  const docCount = stats.documents ?? stats.files?.length ?? 0;
  const pageCount = stats.pages ?? 0;
  const chunkCount = stats.chunks ?? 0;

  const statsText = `${docCount} document${docCount === 1 ? '' : 's'} · ${pageCount} page${
    pageCount === 1 ? '' : 's'
  } · ${chunkCount} chunk${chunkCount === 1 ? '' : 's'}`;

  const handleStartClear = () => {
    if (!hasFiles || isProcessing) return;
    setIsConfirmingClear(true);
  };

  const handleConfirmClear = () => {
    setIsConfirmingClear(false);
    onClear();
  };

  const handleCancelClear = () => {
    setIsConfirmingClear(false);
  };

  return (
    <aside
      className={`sidebar ${isOpenMobile ? 'sidebar-mobile-open' : ''}`}
      aria-label="Document management sidebar"
    >
      {/* Sidebar Navy Title Bar */}
      <div className="sidebar-title-bar">
        <div className="sidebar-brand">
          <svg
            className="brand-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
            <path d="M6 6h10" />
            <path d="M6 10h10" />
          </svg>
          <span className="sidebar-title">Documents</span>
        </div>

        {/* Mobile close button */}
        <button
          type="button"
          className="sidebar-close-btn"
          onClick={onCloseMobile}
          aria-label="Close documents panel"
        >
          <svg
            viewBox="0 0 24 24"
            width="20"
            height="20"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden="true"
          >
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="sidebar-content">
        {/* Upload Dropzone (inset neumorphic) */}
        <Dropzone
          onFileSelect={onFileSelect}
          isProcessing={isProcessing}
          status={status}
        />

        {/* Indexed files list */}
        <div className="documents-section">
          <div className="section-header">
            <span className="section-label">Indexed Files</span>
            <span className="badge-count">{docCount}</span>
          </div>

          {!hasFiles ? (
            <div className="sidebar-empty">
              <p>No documents uploaded yet.</p>
            </div>
          ) : (
            <ul className="doc-cards-list" aria-label="List of indexed files">
              {stats.files.map((fileName) => {
                const pages = filePageCounts[fileName];
                return (
                  <li key={fileName} className="doc-card" title={fileName}>
                    <div className="doc-card-icon-wrapper" aria-hidden="true">
                      <svg
                        className="doc-pdf-icon"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-9.5 8.5h-2v1.5h2c.55 0 1-.45 1-1s-.45-.5-1-.5zm0-2.5h-2v1.5h2c.55 0 1-.45 1-1s-.45-.5-1-.5zm5.5 2.5h-1.5V13h1.5c.55 0 1-.45 1-1s-.45-.5-1-.5zm0-2.5h-1.5V10h1.5c.55 0 1-.45 1-1s-.45-.5-1-.5zM14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />
                      </svg>
                    </div>

                    <div className="doc-card-info">
                      <span className="doc-card-name">{fileName}</span>
                      {pages != null && (
                        <span className="doc-card-pages">
                          {pages} {pages === 1 ? 'page' : 'pages'}
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>

      {/* Sidebar Footer: Inset Stats and Clear Confirmation Button */}
      <div className="sidebar-footer">
        <div className="sidebar-stats" aria-label="Document statistics">
          {statsText}
        </div>

        {hasFiles && (
          <>
            {isConfirmingClear ? (
              <div className="clear-confirm-panel" role="alert">
                <span className="clear-confirm-prompt">Remove all documents?</span>
                <div className="clear-confirm-buttons">
                  <button
                    type="button"
                    className="btn-navy-confirm"
                    onClick={handleConfirmClear}
                    disabled={isProcessing}
                    aria-label="Confirm clearing all documents"
                  >
                    Clear all documents
                  </button>
                  <button
                    type="button"
                    className="btn-cancel-clear"
                    onClick={handleCancelClear}
                    disabled={isProcessing}
                    aria-label="Cancel clearing documents"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                type="button"
                className="btn-clear-trigger"
                onClick={handleStartClear}
                disabled={isProcessing}
                aria-label="Clear all indexed documents"
              >
                Clear all documents
              </button>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
