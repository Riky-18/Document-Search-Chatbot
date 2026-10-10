/**
 * SourceBadges component renders citation pills under an assistant message.
 * Format: "filename.pdf · p.4"
 */
export default function SourceBadges({ sources }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-container">
      <span className="sources-label">Sources</span>
      <div className="sources-list">
        {sources.map((src, index) => (
          <span
            key={`${src.file_name}-${src.page_number}-${index}`}
            className="source-badge"
            title={`${src.file_name} · Page ${src.page_number}`}
          >
            <svg
              className="source-icon"
              viewBox="0 0 16 16"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M4 1.5A1.5 1.5 0 0 0 2.5 3v10A1.5 1.5 0 0 0 4 14.5h8a1.5 1.5 0 0 0 1.5-1.5V6.207a1.5 1.5 0 0 0-.44-1.06L9.854 1.94A1.5 1.5 0 0 0 8.793 1.5H4zm5 1.207L12.293 6H9V2.707zM3.5 3a.5.5 0 0 1 .5-.5h4V6.5A1.5 1.5 0 0 0 9.5 8h3.5v5a.5.5 0 0 1-.5.5H4a.5.5 0 0 1-.5-.5V3z" />
            </svg>
            <span className="source-text">
              {src.file_name} · p.{src.page_number}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
