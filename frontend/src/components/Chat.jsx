import { useEffect, useRef, useState } from 'react';
import { askQuestion } from '../api';

/**
 * Chat component providing question-and-answer interactions
 * over indexed PDF documents.
 */
export default function Chat({ hasDocuments, messages, setMessages }) {
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);

  // Reference to message list container to auto-scroll
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  /**
   * Automatically scroll to the latest message whenever messages or thinking state change.
   */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  /**
   * Submit the current question to the backend.
   */
  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isThinking || !hasDocuments) return;

    // Add user question to message list
    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsThinking(true);

    try {
      const response = await askQuestion(trimmed);

      // Add assistant response with answer, sources, and timing metrics
      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        sources: response.sources || [],
        timing: {
          response_time_seconds: response.response_time_seconds,
          retrieval_seconds: response.retrieval_seconds,
          llm_seconds: response.llm_seconds,
        },
        isError: false,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      // Display backend error as an assistant message in red without rewriting text
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

  /**
   * Keydown handler: Enter sends, Shift+Enter adds a newline.
   */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isSendDisabled = !input.trim() || isThinking || !hasDocuments;

  return (
    <section className="card chat-card">
      <h2 className="section-title">Chat with Documents</h2>

      {/* Scrollable messages container */}
      <div className="chat-messages-container" tabIndex={0} aria-label="Chat message history">
        {messages.length === 0 && !isThinking ? (
          <div className="chat-empty-state">
            Upload a PDF, then ask a question about it.
          </div>
        ) : (
          <div className="chat-messages-list">
            {messages.map((msg) => {
              const isUser = msg.role === 'user';
              return (
                <div
                  key={msg.id}
                  className={`chat-message-row ${isUser ? 'row-user' : 'row-assistant'}`}
                >
                  <div
                    className={`message-bubble ${
                      isUser
                        ? 'bubble-user'
                        : msg.isError
                        ? 'bubble-assistant-error'
                        : 'bubble-assistant'
                    }`}
                  >
                    <div className="message-content">{msg.content}</div>

                    {/* Sources chips under assistant answers */}
                    {!isUser && !msg.isError && msg.sources && msg.sources.length > 0 && (
                      <div className="sources-container">
                        <span className="sources-title">Sources:</span>
                        <div className="sources-chips">
                          {msg.sources.map((src, index) => (
                            <span
                              key={`${src.file_name}-${src.page_number}-${index}`}
                              className="source-chip"
                            >
                              {src.file_name}, p. {src.page_number}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Muted timing info line */}
                    {!isUser && !msg.isError && msg.timing && (
                      <div className="timing-info">
                        Answered in {msg.timing.response_time_seconds}s (retrieval{' '}
                        {msg.timing.retrieval_seconds}s, LLM {msg.timing.llm_seconds}s)
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* "Thinking..." assistant bubble while waiting for API */}
            {isThinking && (
              <div className="chat-message-row row-assistant">
                <div className="message-bubble bubble-assistant bubble-thinking">
                  <span className="spinner" aria-hidden="true"></span>
                  <span>Thinking...</span>
                </div>
              </div>
            )}

            {/* Anchor ref for auto-scrolling */}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <form
        className="chat-input-form"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
      >
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            !hasDocuments
              ? 'Upload a PDF first.'
              : 'Ask a question about the document(s)... (Enter to send, Shift+Enter for newline)'
          }
          disabled={!hasDocuments || isThinking}
          rows={2}
          aria-label="Question input"
        />
        <button
          type="submit"
          className="btn btn-primary btn-send"
          disabled={isSendDisabled}
        >
          Send
        </button>
      </form>
    </section>
  );
}
