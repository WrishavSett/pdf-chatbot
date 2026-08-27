/**
 * chat.js
 * --------
 * Handles everything on the chat screen:
 *   - Sending a user question → POST /chat
 *   - Rendering the answer (+ translated answer if applicable)
 *   - "New Chat" → POST /reset → back to upload screen
 *
 * Mirrors the chat half of app.py exactly:
 *   - get_answer() call → answer, translated_answer?
 *   - Two separate messages when translation exists
 *   - reset_chat() → /reset → clear state → st.rerun()
 */

'use strict';

const Chat = (() => {

  // -------------------------------------------------------
  // SEND MESSAGE FLOW
  // Mirrors app.py: user_input block + get_answer() call
  // -------------------------------------------------------

  async function _handleSend() {
    const question = Ui.els.chatInput.value.trim();
    if (!question) return;

    const sessionId = State.get('sessionId');
    const language  = State.get('language');

    // Clear previous chat errors
    Ui.clearChatError();

    // 1. Append user message immediately
    //    Mirrors: st.session_state.messages.append({"role": "user", ...})
    State.addMessage('user', question);
    Ui.appendMessage('user', question);
    Ui.clearChatInput();

    // 2. Show typing indicator + disable input
    //    Mirrors: with st.spinner("Thinking...")
    Ui.setSendBtnLoading(true);
    const typingEl = Ui.showTypingIndicator();

    try {
      // 3. POST /chat — mirrors get_answer() in app.py
      const data = await Api.getAnswer(sessionId, question, language);

      // 4. Remove typing indicator, re-enable input
      Ui.removeTypingIndicator();
      Ui.setSendBtnLoading(false);

      // 5. Append bot answer bubble
      //    Mirrors: st.markdown(answer)
      //    And if translated: second assistant message
      const translatedAnswer = data.translated_answer || null;

      State.addMessage('assistant', data.answer, translatedAnswer);
      Ui.appendMessage('assistant', data.answer, translatedAnswer, language);

    } catch (err) {
      // Mirrors: st.error("Failed to get a response from the server.")
      Ui.removeTypingIndicator();
      Ui.setSendBtnLoading(false);
      Ui.showChatError(err.message || 'Failed to get a response from the server.');
    }
  }

  // -------------------------------------------------------
  // NEW CHAT (RESET) FLOW
  // Mirrors reset_chat() in app.py exactly:
  //   POST /reset → clear all session state → st.rerun()
  // -------------------------------------------------------

  async function _handleStartOver() {
    const sessionId = State.get('sessionId');

    // POST /reset — fire-and-forget, matching app.py's try/except with warning
    await Api.resetSession(sessionId);

    // Clear all client state — mirrors: for key in st.session_state.keys(): del
    State.reset();

    // Clear UI
    Ui.clearChatMessages();
    Ui.clearChatError();
    Ui.clearSummary();
    Ui.clearFileInfoBar();
    Upload.reset();

    // Re-enable input for next session
    Ui.setChatInputEnabled(false);

    // Transition back to upload screen — mirrors st.rerun()
    Ui.showUploadScreen();
  }

  // -------------------------------------------------------
  // EVENT BINDINGS
  // -------------------------------------------------------

  function init() {
    // Send button click
    Ui.els.sendBtn.addEventListener('click', _handleSend);

    // Enter key in chat input — mirrors st.chat_input behaviour
    Ui.els.chatInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!Ui.els.sendBtn.disabled) {
          _handleSend();
        }
      }
    });

    // New Chat button
    Ui.els.startOverBtn.addEventListener('click', _handleStartOver);
  }

  return { init };

})();
