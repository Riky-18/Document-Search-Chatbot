import { useEffect, useRef, useState } from 'react';
import MessageBubble, { TypingIndicator } from './MessageBubble';

const EXAMPLE_QUESTIONS = [
  'What is the main summary of this document?',
  'What are the key findings and conclusions?',
  'Explain the methodology and key concepts',
];

/**
 * ChatWindow manages the chat feed, empty states with raised example chips,
 * autoscroll, and the inset neumorphic textarea with navy send button.
 */
export default function ChatWindow({
  hasDocuments,
  messages,
  isThinking,
  onSendMessage,
}) {
  const [input, setInput] = useState('');
  const [lastQuestion, setLastQuestion] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isThinking || !hasDocuments) return;

    setLastQuestion(trimmed);
    onSendMessage(trimmed);
    setInput('');

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleChipClick = (question) => {
    setInput(question);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const handleRetry = (questionToRetry) => {
    if (isThinking || !hasDocuments) return;
    const q = questionToRetry || lastQuestion;
    if (q) {
      onSendMessage(q);
    }
  };

  const handleInputInput = (e) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`;
  };

  const isSendDisabled = !input.trim() || isThinking || !hasDocuments;

  return (
    <main className="chat-window" aria-label="Chat conversation">
      {/* Scrollable messages container */}
      <div
        className="chat-scroll-area"
        tabIndex={0}
        aria-label="Conversation message history"
      >
        {messages.length === 0 && !isThinking ? (
          <div className="chat-empty-state">
            <div className="empty-state-icon-wrapper" aria-hidden="true">
              <svg
                className="empty-state-icon"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v7.018Z"
                />
              </svg>
            </div>

            {!hasDocuments ? (
              <div className="empty-content">
                <h2 className="empty-heading">No documents indexed</h2>
                <p className="empty-description">
                  Upload a PDF to get started
                </p>
              </div>
            ) : (
              <div className="empty-content">
                <h2 className="empty-heading">Ask anything about your documents</h2>
                <p className="empty-description">
                  Select an example below or type your own question:
                </p>
                <div className="example-chips-grid">
                  {EXAMPLE_QUESTIONS.map((chipText) => (
                    <button
                      key={chipText}
                      type="button"
                      className="example-chip"
                      onClick={() => handleChipClick(chipText)}
                      aria-label={`Ask: ${chipText}`}
                    >
                      <span>{chipText}</span>
                      <svg
                        className="chip-arrow"
                        viewBox="0 0 16 16"
                        width="14"
                        height="14"
                        fill="currentColor"
                        aria-hidden="true"
                      >
                        <path
                          fillRule="evenodd"
                          d="M1 8a.5.5 0 0 1 .5-.5h11.793l-3.147-3.146a.5.5 0 0 1 .708-.708l4 4a.5.5 0 0 1 0 .708l-4 4a.5.5 0 0 1-.708-.708L13.293 8.5H1.5A.5.5 0 0 1 1 8z"
                        />
                      </svg>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="messages-feed">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onRetry={handleRetry}
                lastQuestion={lastQuestion}
              />
            ))}

            {isThinking && <TypingIndicator />}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input container pinned at bottom */}
      <div className="chat-input-container">
        <form className="chat-input-box" onSubmit={handleSubmit}>
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            value={input}
            onChange={handleInputInput}
            onKeyDown={handleKeyDown}
            placeholder={
              !hasDocuments
                ? 'Upload a PDF to start asking questions...'
                : 'Ask a question about your documents...'
            }
            disabled={!hasDocuments || isThinking}
            rows={1}
            aria-label="Ask a question about your documents"
          />

          <button
            type="submit"
            className="btn-send-navy"
            disabled={isSendDisabled}
            aria-label="Send question"
          >
            <svg
              className="send-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </form>
        <div className="chat-input-hint">
          Enter to send, Shift+Enter for new line
        </div>
      </div>
    </main>
  );
}
