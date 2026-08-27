/**
 * state.js
 * ---------
 * Client-side session state — the JS equivalent of Streamlit's
 * st.session_state. Every field mirrors a key from app.py.
 *
 * Streamlit key → JS field:
 *   uploaded          → State.uploaded         (bool)
 *   session_id        → State.sessionId         (string | null)
 *   summary           → State.summary           (string | null)
 *   language          → State.language          (string)
 *   messages          → State.messages          (Message[])
 *
 * Additionally, we track:
 *   fileName          — captured from File object at selection time
 *   fileSize          — captured from File object at selection time
 *   (these are NOT available from the API; captured client-side only)
 */

'use strict';

const State = (() => {

  // -------------------------------------------------------
  // PRIVATE STATE — single source of truth
  // -------------------------------------------------------
  let _state = _buildInitialState();

  function _buildInitialState() {
    return {
      // mirrors st.session_state.uploaded
      uploaded: false,

      // mirrors st.session_state.session_id
      sessionId: null,

      // mirrors st.session_state.summary
      summary: null,

      // mirrors st.session_state.language (default: no translation)
      language: '',

      // mirrors st.session_state.messages — array of { role, content, translatedContent? }
      // role: 'assistant' | 'user'
      messages: [],

      // client-side only: captured from the File object before upload
      fileName: '',
      fileSize: 0,
    };
  }

  // -------------------------------------------------------
  // PUBLIC GETTERS
  // -------------------------------------------------------
  function get(key) {
    return _state[key];
  }

  function getAll() {
    // Shallow copy to prevent external mutation
    return { ..._state };
  }

  // -------------------------------------------------------
  // PUBLIC SETTERS
  // -------------------------------------------------------
  function set(key, value) {
    if (!(key in _state)) {
      console.warn(`[State] Unknown key: "${key}"`);
    }
    _state[key] = value;
  }

  /**
   * Append a message to the messages list.
   * Mirrors app.py: st.session_state.messages.append(...)
   *
   * @param {'user'|'assistant'} role
   * @param {string} content              — primary (English) content
   * @param {string|null} [translatedContent] — optional translated content
   */
  function addMessage(role, content, translatedContent = null) {
    _state.messages.push({ role, content, translatedContent });
  }

  /**
   * Reset all state back to initial values.
   * Mirrors app.py: for key in list(st.session_state.keys()): del st.session_state[key]
   */
  function reset() {
    _state = _buildInitialState();
  }

  // -------------------------------------------------------
  // PUBLIC INTERFACE
  // -------------------------------------------------------
  return { get, getAll, set, addMessage, reset };

})();
