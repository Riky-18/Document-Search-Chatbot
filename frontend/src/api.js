/**
 * API client for the AI Document Chatbot backend.
 */

const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

/**
 * Extract human-readable error detail from backend response.
 * Handles string details as well as FastAPI 422 validation error arrays.
 */
async function parseErrorDetail(response) {
  try {
    const data = await response.json();
    if (data && data.detail) {
      if (Array.isArray(data.detail)) {
        const msgs = data.detail
          .map((item) => (item && item.msg ? item.msg : String(item)))
          .filter(Boolean);
        if (msgs.length > 0) {
          return msgs.join(', ');
        }
      }
      if (typeof data.detail === 'string') {
        return data.detail;
      }
      return JSON.stringify(data.detail);
    }
  } catch {
    // Response was not JSON
  }
  return `Request failed (${response.status})`;
}

/**
 * Execute fetch request with network error and HTTP error handling.
 */
async function request(endpoint, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${endpoint}`, options);
  } catch {
    throw new Error('Cannot reach the server. Is the backend running?');
  }

  if (!response.ok) {
    const errorMsg = await parseErrorDetail(response);
    throw new Error(errorMsg);
  }

  return response.json();
}

/**
 * Upload and index a PDF file.
 * @param {File} file
 * @returns {Promise<{file_name: string, pages: number, chunks: number}>}
 */
export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append('file', file);

  return request('/upload', {
    method: 'POST',
    body: formData,
  });
}

/**
 * Ask a question against the indexed documents with optional conversation history.
 * @param {string} question
 * @param {Array<{question: string, answer: string}>} [history]
 * @returns {Promise<{answer: string, sources: Array<{file_name: string, page_number: number}>, response_time_seconds: number, retrieval_seconds: number, llm_seconds: number, rewrite_seconds?: number, standalone_question?: string}>}
 */
export async function askQuestion(question, history = []) {
  const payload = { question };
  if (Array.isArray(history) && history.length > 0) {
    payload.history = history;
  }

  return request('/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
}

/**
 * Get current vector store stats and list of indexed documents.
 * @returns {Promise<{documents: number, pages: number, chunks: number, files: string[]}>}
 */
export async function getStats() {
  return request('/stats', {
    method: 'GET',
  });
}

/**
 * Reset and clear all indexed documents from the vector store.
 * @returns {Promise<{status: string, message: string}>}
 */
export async function clearIndex() {
  return request('/clear', {
    method: 'POST',
  });
}

/**
 * Get detailed list of indexed documents with page and chunk counts.
 * @returns {Promise<Array<{file_name: string, pages: number, chunks: number}>>}
 */
export async function getDocuments() {
  return request('/documents', {
    method: 'GET',
  });
}

/**
 * Remove a single indexed document from the vector store.
 * @param {string} fileName
 * @returns {Promise<{file_name: string, chunks_removed: number, pages_removed: number}>}
 */
export async function deleteDocument(fileName) {
  return request(`/documents/${encodeURIComponent(fileName)}`, {
    method: 'DELETE',
  });
}
