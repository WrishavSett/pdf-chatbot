/**
 * api.js
 * -------
 * All communication with the Flask API (api.py).
 * This module is the ONLY place that knows the API's URL and
 * request/response shapes. Nothing else in the frontend should
 * call fetch() directly.
 *
 * Mirrors app.py's four API interactions:
 *   fetch_languages()  → GET  /languages
 *   upload_pdf()       → POST /upload   (multipart/form-data)
 *   get_answer()       → POST /chat     (JSON)
 *   reset_chat()       → POST /reset    (JSON)
 */

'use strict';

const Api = (() => {

  // ------------------------------------------------------------
  // CONFIG
  // Matches API_BASE_URL in .config (http://localhost:8000).
  // The browser fetches the HTML from nginx (port 3000) and calls
  // the Flask API on port 8000 — a cross-origin request.
  // flask-cors (CORS(app) in api.py) already allows all origins,
  // so NO changes to api.py are needed.
  //
  // To change the API host (e.g. in production), update this value.
  // ------------------------------------------------------------
  const BASE_URL = 'http://localhost:8000';

  // ------------------------------------------------------------
  // INTERNAL HELPERS
  // ------------------------------------------------------------

  /**
   * Shared error handler. Throws a descriptive Error so callers
   * can display it via the UI layer.
   * @param {Response} response
   */
  async function _handleError(response) {
    let message = `Server error (${response.status})`;
    try {
      const body = await response.json();
      if (body.error) message = body.error;
    } catch (_) { /* ignore parse errors, keep default message */ }
    throw new Error(message);
  }

  // ------------------------------------------------------------
  // PUBLIC API METHODS
  // ------------------------------------------------------------

  /**
   * Fetch the list of supported translation languages.
   * Mirrors: fetch_languages() in app.py → GET /languages
   *
   * @returns {Promise<string[]>} e.g. ["Spanish", "French", ...]
   */
  async function fetchLanguages() {
    const response = await fetch(`${BASE_URL}/languages`, {
      method: 'GET',
    });
    if (!response.ok) await _handleError(response);
    const data = await response.json();
    return data.languages; // string[]
  }

  /**
   * Upload a PDF and receive the session_id + summary.
   * Mirrors: upload_pdf() in app.py → POST /upload (multipart)
   *
   * @param {File}   file     - The PDF File object from <input type="file">
   * @param {string} language - Selected language, or '' for no translation
   * @returns {Promise<{
   *   session_id: string,
   *   summary: string,
   *   translated_summary?: string
   * }>}
   */
  async function uploadPdf(file, language) {
    const formData = new FormData();
    formData.append('file', file);

    // Only attach language if user actually selected one
    // (mirrors app.py: if language and language != "None")
    if (language && language !== '') {
      formData.append('language', language);
    }

    const response = await fetch(`${BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
      // NOTE: Do NOT set Content-Type header — browser sets it
      // automatically with the correct multipart boundary.
    });

    if (!response.ok) await _handleError(response);
    return response.json(); // { session_id, summary, translated_summary? }
  }

  /**
   * Send a chat question and receive the answer.
   * Mirrors: get_answer() in app.py → POST /chat (JSON)
   *
   * @param {string} sessionId - Active session UUID
   * @param {string} question  - User's question text
   * @param {string} language  - Selected language, or '' for no translation
   * @returns {Promise<{
   *   answer: string,
   *   translated_answer?: string
   * }>}
   */
  async function getAnswer(sessionId, question, language) {
    const payload = {
      question:   question,
      session_id: sessionId,
    };

    // Only attach language if user actually selected one
    // (mirrors app.py: if language and language != "None")
    if (language && language !== '') {
      payload.language = language;
    }

    const response = await fetch(`${BASE_URL}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!response.ok) await _handleError(response);
    return response.json(); // { answer, translated_answer? }
  }

  /**
   * Reset (clear) the current session on the server.
   * Mirrors: reset_chat() in app.py → POST /reset (JSON)
   *
   * @param {string} sessionId - Session UUID to clear
   * @returns {Promise<void>}
   */
  async function resetSession(sessionId) {
    if (!sessionId) return; // nothing to clear

    try {
      const response = await fetch(`${BASE_URL}/reset`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: sessionId }),
      });
      // Best-effort — mirrors app.py's try/except with only a warning on failure
      if (!response.ok) {
        console.warn('[Api] Reset call failed:', response.status);
      }
    } catch (err) {
      console.warn('[Api] Reset call threw:', err);
    }
  }

  // Expose only the public interface
  return { fetchLanguages, uploadPdf, getAnswer, resetSession };

})();
