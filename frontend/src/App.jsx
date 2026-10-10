import { useEffect, useState } from 'react';
import { askQuestion, clearIndex, deleteDocument, getDocuments, getStats, uploadPdf } from './api';
import ChatWindow from './components/ChatWindow';
import Sidebar from './components/Sidebar';

function isDontKnowAnswer(answer) {
  if (!answer) return false;
  const lower = answer.toLowerCase();
  return (
    lower.includes("don't know") ||
    lower.includes("dont know") ||
    lower.includes("do not know") ||
    lower.includes("not mentioned in the context") ||
    lower.includes("not provided in the context") ||
    lower.includes("not found in the context") ||
    lower.includes("no information") ||
    lower.includes("no relevant information") ||
    lower.includes("context does not contain") ||
    lower.includes("context does not mention") ||
    lower.includes("no documents are currently indexed")
  );
}

function extractRecentHistory(messages, limit = 3) {
  const turns = [];
  for (let i = 0; i < messages.length - 1; i++) {
    const current = messages[i];
    const next = messages[i + 1];
    if (current.role === 'user' && next.role === 'assistant') {
      if (next.isError) continue;
      if (isDontKnowAnswer(next.content)) continue;
      if (!current.content || !next.content) continue;
      turns.push({
        question: current.content,
        answer: next.content,
      });
    }
  }
  return turns.slice(-limit);
}

export default function App() {
  const [stats, setStats] = useState({
    documents: 0,
    pages: 0,
    chunks: 0,
    files: [],
  });

  const [filePageCounts, setFilePageCounts] = useState({});

  const [status, setStatus] = useState({
    type: 'idle',
    message: '',
    details: null,
  });

  const [isProcessing, setIsProcessing] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isOpenMobileSidebar, setIsOpenMobileSidebar] = useState(false);

  const fetchStats = async () => {
    try {
      const [data, docs] = await Promise.all([getStats(), getDocuments()]);
      setStats({
        documents: data.documents || 0,
        pages: data.pages || 0,
        chunks: data.chunks || 0,
        files: Array.isArray(data.files) ? data.files : [],
      });
      if (Array.isArray(docs)) {
        const pageMap = {};
        docs.forEach((doc) => {
          pageMap[doc.file_name] = doc.pages;
        });
        setFilePageCounts((prev) => ({ ...prev, ...pageMap }));
      }
    } catch (err) {
      setStatus({
        type: 'error',
        message: err.message,
        details: null,
      });
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const handleFileSelect = async (file) => {
    if (!file) return;

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
      setFilePageCounts((prev) => ({
        ...prev,
        [result.file_name]: result.pages,
      }));
      setStatus({
        type: 'done',
        message: `Indexed "${result.file_name}" (${result.pages} page${
          result.pages === 1 ? '' : 's'
        }, ${result.chunks} chunk${result.chunks === 1 ? '' : 's'}).`,
        details: result,
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

  const handleClear = async () => {
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
        message: 'All documents cleared.',
        details: null,
      });
      setFilePageCounts({});
      setMessages([]);
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

  const handleDeleteDocument = async (fileName) => {
    try {
      const result = await deleteDocument(fileName);
      setStatus({
        type: 'done',
        message: `Removed "${result.file_name}" (${result.pages_removed} page${
          result.pages_removed === 1 ? '' : 's'
        }, ${result.chunks_removed} chunk${result.chunks_removed === 1 ? '' : 's'}).`,
        details: result,
      });
      setFilePageCounts((prev) => {
        const next = { ...prev };
        delete next[fileName];
        return next;
      });
      await fetchStats();
      return result;
    } catch (err) {
      setStatus({
        type: 'error',
        message: `Failed to remove "${fileName}": ${err.message}`,
        details: null,
      });
      throw err;
    }
  };

  const handleNewChat = () => {
    setMessages([]);
  };

  const handleSendMessage = async (question) => {
    const trimmed = question.trim();
    if (!trimmed || isThinking) return;

    const recentHistory = extractRecentHistory(messages, 3);

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsThinking(true);

    try {
      const response = await askQuestion(trimmed, recentHistory);
      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        sources: response.sources || [],
        timing: {
          response_time_seconds: response.response_time_seconds,
          retrieval_seconds: response.retrieval_seconds,
          llm_seconds: response.llm_seconds,
          rewrite_seconds: response.rewrite_seconds,
        },
        standalone_question: response.standalone_question,
        isError: false,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: err.message,
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsThinking(false);
    }
  };

  const hasDocuments = Boolean(stats.files && stats.files.length > 0);
  const docCount = stats.documents || stats.files?.length || 0;

  return (
    <div className="app-shell">
      {/* Top Header Bar Spanning Entire Width in Navy */}
      <header className="top-header-bar">
        <div className="header-brand-block">
          <div className="header-title-row">
            <svg
              className="header-brand-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            <h1 className="header-title">Document Chat</h1>
          </div>
          <p className="header-subtitle">
            Ask questions about your PDFs. Answers come only from your documents.
          </p>
        </div>

        {/* Header Action Buttons */}
        <div className="header-actions">
          <button
            type="button"
            className="btn-new-chat"
            onClick={handleNewChat}
            aria-label="New chat"
          >
            <svg
              viewBox="0 0 24 24"
              width="15"
              height="15"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12 5v14M5 12h14" />
            </svg>
            <span className="btn-new-chat-text">New chat</span>
          </button>

          {/* Mobile Raised "Documents (N)" Button in Navy Header */}
          <button
            type="button"
            className="mobile-raised-docs-btn"
            onClick={() => setIsOpenMobileSidebar(true)}
            aria-label={`Open documents panel (${docCount} indexed)`}
          >
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
            </svg>
            <span>Documents</span>
            <span className="mobile-docs-badge">{docCount}</span>
          </button>
        </div>
      </header>

      {/* Main Two-Column Layout */}
      <div className="app-layout">
        {isOpenMobileSidebar && (
          <div
            className="sidebar-backdrop"
            onClick={() => setIsOpenMobileSidebar(false)}
            aria-hidden="true"
          />
        )}

        <Sidebar
          stats={stats}
          status={status}
          isProcessing={isProcessing}
          onFileSelect={handleFileSelect}
          onClear={handleClear}
          onDeleteDocument={handleDeleteDocument}
          filePageCounts={filePageCounts}
          isOpenMobile={isOpenMobileSidebar}
          onCloseMobile={() => setIsOpenMobileSidebar(false)}
        />

        <ChatWindow
          hasDocuments={hasDocuments}
          messages={messages}
          isThinking={isThinking}
          onSendMessage={handleSendMessage}
        />
      </div>
    </div>
  );
}
