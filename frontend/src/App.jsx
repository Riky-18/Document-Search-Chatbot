import { useEffect, useRef, useState } from 'react';
import { clearIndex, getStats, uploadPdf } from './api';

export default function App() {
  // Stats state loaded from backend
  const [stats, setStats] = useState({
    documents: 0,
    pages: 0,
    chunks: 0,
    files: [],
  });

  // Upload and operation status state: 'idle' | 'indexing' | 'done' | 'error'
  const [status, setStatus] = useState({
    type: 'idle',
    message: 'Ready. Select a PDF file to index.',
    details: null,
  });

  // Track background processing to disable buttons
  const [isProcessing, setIsProcessing] = useState(false);

  // Hidden file input reference
  const fileInputRef = useRef(null);

  /**
   * Load vector store statistics and file list from backend.
   */
  const fetchStats = async () => {
    try {
      const data = await getStats();
      setStats({
        documents: data.documents || 0,
        pages: data.pages || 0,
        chunks: data.chunks || 0,
        files: Array.isArray(data.files) ? data.files : [],
      });
    } catch (err) {
      setStatus({
        type: 'error',
        message: err.message,
        details: null,
      });
    }
  };

  // Fetch stats on initial page mount
  useEffect(() => {
    fetchStats();
  }, []);

  /**
   * Trigger the hidden file input when the styled button is clicked.
   */
  const handleUploadClick = () => {
    if (fileInputRef.current && !isProcessing) {
      fileInputRef.current.click();
    }
  };

  /**
   * Handle PDF file selection and upload flow.
   */
  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Reset input so re-uploading the same file triggers onChange
    event.target.value = '';

    // Enforce PDF extension check on client before making network request
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setStatus({
        type: 'error',
        message: 'Invalid file type. Please select a .pdf file.',
        details: null,
      });
      return;
    }

    setIsProcessing(true);
    setStatus({
      type: 'indexing',
      message: `Indexing "${file.name}"...`,
      details: null,
    });

    try {
      const result = await uploadPdf(file);
      setStatus({
        type: 'done',
        message: `Indexed "${result.file_name}" successfully (${result.pages} page${result.pages === 1 ? '' : 's'}, ${result.chunks} chunk${result.chunks === 1 ? '' : 's'}).`,
        details: result,
      });
      // Refresh indexed document list and totals
      await fetchStats();
    } catch (err) {
      setStatus({
        type: 'error',
        message: err.message,
        details: null,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  /**
   * Handle clearing all indexed documents.
   */
  const handleClearClick = async () => {
    if (stats.files.length === 0 || isProcessing) return;

    const confirmed = window.confirm(
      'Are you sure you want to remove all indexed documents? This cannot be undone.'
    );
    if (!confirmed) return;

    setIsProcessing(true);
    setStatus({
      type: 'indexing',
      message: 'Clearing all indexed documents...',
      details: null,
    });

    try {
      await clearIndex();
      setStatus({
        type: 'idle',
        message: 'All documents cleared. Index is empty.',
        details: null,
      });
      await fetchStats();
    } catch (err) {
      setStatus({
        type: 'error',
        message: err.message,
        details: null,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="app-title">AI Document Chatbot</h1>
        <p className="app-subtitle">
          Upload and index your PDF documents to ask questions with citations.
        </p>
      </header>

      {/* Upload Section */}
      <section className="card">
        <h2 className="section-title">Document Upload</h2>

        <div className="action-row">
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />

          {/* Styled trigger button */}
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleUploadClick}
            disabled={isProcessing}
          >
            {isProcessing && status.type === 'indexing' && status.message.startsWith('Indexing') ? (
              <>
                <span className="spinner" aria-hidden="true"></span>
                <span>Indexing...</span>
              </>
            ) : (
              <span>Upload PDF Document</span>
            )}
          </button>

          {/* Clear documents button */}
          <button
            type="button"
            className="btn btn-danger"
            onClick={handleClearClick}
            disabled={isProcessing || stats.files.length === 0}
          >
            Clear All Documents
          </button>
        </div>

        {/* Dynamic colored status banner */}
        <div className={`status-banner status-${status.type}`} role="status">
          <strong>Status:</strong> {status.message}
        </div>
      </section>

      {/* Indexed Files Section */}
      <section className="card">
        <div className="section-title">
          <span>Indexed Documents ({stats.files.length})</span>
        </div>

        {stats.files.length === 0 ? (
          <p className="empty-state">No documents indexed yet.</p>
        ) : (
          <ul className="doc-list">
            {stats.files.map((fileName) => (
              <li key={fileName} className="doc-item">
                <span className="doc-icon" aria-hidden="true">📄</span>
                <span>{fileName}</span>
              </li>
            ))}
          </ul>
        )}

        {/* Index Totals */}
        <div className="totals-bar">
          <div className="totals-item">
            <span>Indexed Pages:</span>
            <span className="totals-value">{stats.pages}</span>
          </div>
          <div className="totals-item">
            <span>Total Chunks:</span>
            <span className="totals-value">{stats.chunks}</span>
          </div>
        </div>
      </section>
    </div>
  );
}
