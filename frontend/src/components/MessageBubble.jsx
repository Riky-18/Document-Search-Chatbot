import SourceBadges from './SourceBadges';

/**
 * MessageBubble renders an individual message within the chat feed.
 * Handles user bubbles (indigo), bot cards (white with border),
 * error cards with a retry button, and animated typing indicator.
 */
export default function MessageBubble({
  message,
  onRetry,
  lastQuestion,
}) {
  const isUser = message.role === 'user';
  const isError = Boolean(message.isError);

  if (isUser) {
    return (
      <div className="message-row message-row-user">
        <div className="bubble-user">
          <div className="message-text">{message.content}</div>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="message-row message-row-bot">
        <div className="bubble-error" role="alert">
          <div className="error-header">
            <svg
              className="error-icon"
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
            <span className="error-title">Unable to answer</span>
          </div>
          <p className="error-message">{message.content}</p>
          {onRetry && lastQuestion && (
            <button
              type="button"
              className="btn-retry"
              onClick={() => onRetry(lastQuestion)}
              aria-label="Retry asking this question"
            >
              <svg
                viewBox="0 0 16 16"
                width="14"
                height="14"
                fill="currentColor"
                aria-hidden="true"
              >
                <path
                  fillRule="evenodd"
                  d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"
                />
                <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z" />
              </svg>
              <span>Retry question</span>
            </button>
          )}
        </div>
      </div>
    );
  }

  // Standard Bot response card
  return (
    <div className="message-row message-row-bot">
      <div className="bubble-bot">
        <div className="message-text">{message.content}</div>

        {/* Sources Badges */}
        {message.sources && message.sources.length > 0 && (
          <SourceBadges sources={message.sources} />
        )}

        {/* Response Timing */}
        {message.timing && message.timing.response_time_seconds != null && (
          <div className="message-timing">
            Answered in {message.timing.response_time_seconds}s
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * TypingIndicator renders an animated three-dot indicator inside a bot bubble.
 */
export function TypingIndicator() {
  return (
    <div className="message-row message-row-bot">
      <div
        className="bubble-bot bubble-typing"
        role="status"
        aria-live="polite"
        aria-label="Assistant is generating an answer"
      >
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  );
}
