import { useEffect, useState } from 'react';
import { getStats } from './api';
import Chat from './components/Chat';
import Upload from './components/Upload';

export default function App() {
  // Shared stats loaded from backend
  const [stats, setStats] = useState({
    documents: 0,
    pages: 0,
    chunks: 0,
    files: [],
  });

  // Upload and operation status: 'idle' | 'indexing' | 'done' | 'error'
  const [status, setStatus] = useState({
    type: 'idle',
    message: 'Ready. Select a PDF file to index.',
    details: null,
  });

  // Track background indexing / clearing
  const [isProcessing, setIsProcessing] = useState(false);

  // In-memory chat conversation history
  const [messages, setMessages] = useState([]);

  /**
   * Fetch vector store stats and list of indexed files from backend.
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

  // Load stats once on initial mount
  useEffect(() => {
    fetchStats();
  }, []);

  /**
   * Called when vector store is cleared.
   * Clears the chat message history as well.
   */
  const handleClearSuccess = () => {
    setMessages([]);
  };

  const hasDocuments = Boolean(stats.files && stats.files.length > 0);

  return (
    <div className="app-container">
      {/* App Header */}
      <header className="app-header">
        <h1 className="app-title">AI Document Chatbot</h1>
        <p className="app-subtitle">
          Upload and index your PDF documents to ask questions with citations.
        </p>
      </header>

      {/* Upload Section */}
      <Upload
        stats={stats}
        status={status}
        setStatus={setStatus}
        isProcessing={isProcessing}
        setIsProcessing={setIsProcessing}
        fetchStats={fetchStats}
        onClearSuccess={handleClearSuccess}
      />

      {/* Chat Section */}
      <Chat
        hasDocuments={hasDocuments}
        messages={messages}
        setMessages={setMessages}
      />
    </div>
  );
}
