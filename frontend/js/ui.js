/**
 * ui.js
 * ------
 * All DOM manipulation lives here. No API calls, no state writes.
 * Other modules call UI functions; UI functions do not call back.
 *
 * Responsibilities:
 *   - Screen switching (upload ↔ chat)
 *   - Rendering the chat message list
 *   - Showing / hiding loading states (spinners, typing indicator)
 *   - Populating the language dropdown
 *   - Populating the file info bar on the chat screen
 *   - Displaying inline error banners
 *   - Enabling / disabling interactive controls
 */

'use strict';

const Ui = (() => {

  // -------------------------------------------------------
  // DOM ELEMENT REFERENCES
  // Cached once — avoids repeated querySelector calls.
  // -------------------------------------------------------
  const els = {
    // Screens
    screenUpload: document.getElementById('screen-upload'),
    screenChat:   document.getElementById('screen-chat'),

    // Upload screen
    dropZone:        document.getElementById('drop-zone'),
    fileInput:       document.getElementById('file-input'),
    chooseFileBtn:   document.getElementById('choose-file-btn'),
    filePreview:     document.getElementById('file-preview'),
    filePreviewName: document.getElementById('file-preview-name'),
    filePreviewSize: document.getElementById('file-preview-size'),
    removeFileBtn:   document.getElementById('remove-file-btn'),
    languageSelect:  document.getElementById('language-select'),
    uploadBtn:       document.getElementById('upload-btn'),
    uploadBtnLabel:  document.getElementById('upload-btn-label'),
    uploadBtnArrow:  document.getElementById('upload-btn-arrow'),
    uploadBtnSpinner:document.getElementById('upload-btn-spinner'),
    uploadError:     document.getElementById('upload-error'),

    // Chat screen
    infoFilename:          document.getElementById('info-filename'),
    infoMeta:              document.getElementById('info-meta'),
    infoLanguage:          document.getElementById('info-language'),
    startOverBtn:          document.getElementById('start-over-btn'),
    summaryEnglish:        document.getElementById('summary-english'),
    summaryTranslatedBlock:document.getElementById('summary-translated-block'),
    summaryTranslatedLabel:document.getElementById('summary-translated-label'),
    summaryTranslated:     document.getElementById('summary-translated'),
    chatMessages:          document.getElementById('chat-messages'),
    chatError:             document.getElementById('chat-error'),
    chatInput:             document.getElementById('chat-input'),
    sendBtn:               document.getElementById('send-btn'),
    sendIcon:              document.getElementById('send-icon'),
    sendSpinner:           document.getElementById('send-spinner'),
  };

  // -------------------------------------------------------
  // HELPERS
  // -------------------------------------------------------

  /** Format bytes to "X.XX MB" or "X KB" */
  function _formatFileSize(bytes) {
    if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    if (bytes >= 1024)        return (bytes / 1024).toFixed(0) + ' KB';
    return bytes + ' B';
  }

  /** Format a Date to "HH:MM AM/PM" */
  function _formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  /** Build the bot avatar SVG element */
  function _makeBotAvatar() {
    const div = document.createElement('div');
    div.className = 'message__avatar';
    div.setAttribute('aria-hidden', 'true');
    div.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 28 28" fill="none">
        <rect x="4" y="8" width="20" height="14" rx="4" fill="#1a56db"/>
        <circle cx="10" cy="14" r="2" fill="white"/>
        <circle cx="18" cy="14" r="2" fill="white"/>
        <rect x="11" y="17" width="6" height="1.5" rx="0.75" fill="white"/>
        <rect x="12" y="3" width="4" height="5" rx="2" fill="#1a56db"/>
        <rect x="2" y="12" width="3" height="5" rx="1.5" fill="#1a56db"/>
        <rect x="23" y="12" width="3" height="5" rx="1.5" fill="#1a56db"/>
      </svg>`;
    return div;
  }

  /** Build the user avatar SVG element */
  function _makeUserAvatar() {
    const div = document.createElement('div');
    div.className = 'message__avatar';
    div.setAttribute('aria-hidden', 'true');
    div.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>`;
    return div;
  }

  // -------------------------------------------------------
  // SCREEN TRANSITIONS
  // -------------------------------------------------------

  /** Switch to the upload screen */
  function showUploadScreen() {
    els.screenChat.classList.add('hidden');
    els.screenUpload.classList.remove('hidden');
  }

  /** Switch to the chat screen */
  function showChatScreen() {
    els.screenUpload.classList.add('hidden');
    els.screenChat.classList.remove('hidden');
  }

  // -------------------------------------------------------
  // UPLOAD SCREEN — LANGUAGE DROPDOWN
  // -------------------------------------------------------

  /**
   * Populate the language <select> from the /languages response.
   * Mirrors: st.session_state.languages = fetch_languages()
   * @param {string[]} languages
   */
  function populateLanguages(languages) {
    // Remove any previously injected options (keep the default)
    const existing = els.languageSelect.querySelectorAll('option.dynamic');
    existing.forEach(o => o.remove());

    languages.forEach(lang => {
      const option = document.createElement('option');
      option.value = lang;
      option.textContent = lang;
      option.className = 'dynamic';
      els.languageSelect.appendChild(option);
    });
  }

  // -------------------------------------------------------
  // UPLOAD SCREEN — FILE PREVIEW STRIP
  // -------------------------------------------------------

  /**
   * Show the selected file name and size under the drop zone.
   * @param {string} name
   * @param {number} sizeBytes
   */
  function showFilePreview(name, sizeBytes) {
    els.filePreviewName.textContent = name;
    els.filePreviewSize.textContent = _formatFileSize(sizeBytes);
    els.filePreview.classList.remove('hidden');
  }

  /** Hide and clear the file preview strip */
  function clearFilePreview() {
    els.filePreviewName.textContent = '';
    els.filePreviewSize.textContent = '';
    els.filePreview.classList.add('hidden');
  }

  // -------------------------------------------------------
  // UPLOAD SCREEN — UPLOAD BUTTON STATE
  // -------------------------------------------------------

  /** Enable or disable the Upload & Process button */
  function setUploadBtnEnabled(enabled) {
    els.uploadBtn.disabled = !enabled;
    els.uploadBtn.setAttribute('aria-disabled', String(!enabled));
  }

  /**
   * Show the uploading spinner in the button.
   * Mirrors: st.spinner("Processing PDF. Please wait...")
   */
  function setUploadBtnLoading(loading) {
    if (loading) {
      els.uploadBtnLabel.textContent = 'Processing…';
      els.uploadBtnArrow.classList.add('hidden');
      els.uploadBtnSpinner.classList.remove('hidden');
      els.uploadBtn.disabled = true;
    } else {
      els.uploadBtnLabel.innerHTML = 'Upload &amp; Process';
      els.uploadBtnArrow.classList.remove('hidden');
      els.uploadBtnSpinner.classList.add('hidden');
      // Re-enable only if a file is selected — caller handles this
    }
  }

  // -------------------------------------------------------
  // UPLOAD SCREEN — DROP ZONE DRAG STATES
  // -------------------------------------------------------

  function setDropZoneDragOver(active) {
    els.dropZone.classList.toggle('drag-over', active);
  }

  // -------------------------------------------------------
  // ERROR BANNERS
  // -------------------------------------------------------

  /**
   * Show an error message on the upload screen.
   * Mirrors: st.error(...)
   * @param {string} message
   */
  function showUploadError(message) {
    els.uploadError.textContent = message;
    els.uploadError.classList.remove('hidden');
  }

  function clearUploadError() {
    els.uploadError.textContent = '';
    els.uploadError.classList.add('hidden');
  }

  /**
   * Show an error message in the chat section.
   * Mirrors: st.error("Failed to get a response from the server.")
   * @param {string} message
   */
  function showChatError(message) {
    els.chatError.textContent = message;
    els.chatError.classList.remove('hidden');
  }

  function clearChatError() {
    els.chatError.textContent = '';
    els.chatError.classList.add('hidden');
  }

  // -------------------------------------------------------
  // CHAT SCREEN — FILE INFO BAR
  // -------------------------------------------------------

  /**
   * Populate the file info bar shown at the top of the chat screen.
   * File name and size are captured from the File object client-side
   * (the API does not return them).
   *
   * @param {string} fileName
   * @param {number} fileSizeBytes
   * @param {string} language - '' means no translation selected
   */
  function populateFileInfoBar(fileName, fileSizeBytes, language) {
    els.infoFilename.textContent = fileName;
    els.infoMeta.textContent = `${_formatFileSize(fileSizeBytes)} · Uploaded`;
    els.infoLanguage.textContent = language || 'English (no translation)';
  }

  /** Clear the current document information bar */
  function clearFileInfoBar() {
    els.infoFilename.textContent = '—';
    els.infoMeta.textContent = '—';
    els.infoLanguage.textContent = '—';
  }

  // -------------------------------------------------------
  // CHAT SCREEN — SUMMARY SECTION
  // -------------------------------------------------------

  /**
   * Render both summary blocks.
   * Mirrors app.py: two messages appended when language selected,
   * one message when no language selected.
   *
   * @param {string}      summary            - English summary from API
   * @param {string|null} translatedSummary  - Translated summary, or null
   * @param {string}      language           - Selected language name
   */
  function renderSummary(summary, translatedSummary, language) {
    els.summaryEnglish.textContent = summary;

    if (translatedSummary && language) {
      // Show translated block with dynamic language label
      // Mirrors: f"Summary ({selected_language})"
      els.summaryTranslatedLabel.textContent = `${language} Translation`;
      els.summaryTranslated.textContent = translatedSummary;
      els.summaryTranslatedBlock.classList.remove('hidden');
    } else {
      els.summaryTranslatedBlock.classList.add('hidden');
    }
  }

  /** Clear all rendered summary content */
  function clearSummary() {
    els.summaryEnglish.textContent = '';
    els.summaryTranslatedLabel.textContent = '';
    els.summaryTranslated.textContent = '';
    els.summaryTranslatedBlock.classList.add('hidden');
  }

  // -------------------------------------------------------
  // CHAT SCREEN — MESSAGE LIST
  // -------------------------------------------------------

  /**
   * Append a single message bubble to the chat list.
   * Mirrors: st.chat_message(role) + st.markdown(content)
   *
   * @param {'user'|'assistant'} role
   * @param {string}      content            - Primary (English) text
   * @param {string|null} translatedContent  - Optional translation
   * @param {string}      language           - Language name for label
   */
  function appendMessage(role, content, translatedContent = null, language = '') {
    const isUser = role === 'user';
    const time   = _formatTime(new Date());

    // Outer message row
    const row = document.createElement('div');
    row.className = `message message--${isUser ? 'user' : 'bot'}`;

    // Avatar
    const avatar = isUser ? _makeUserAvatar() : _makeBotAvatar();

    // Bubble wrapper — holds primary bubble + optional translated bubble
    const wrap = document.createElement('div');
    wrap.className = 'message__bubble-wrap';

    // Primary bubble
    const bubble = document.createElement('div');
    bubble.className = 'message__bubble';
    bubble.textContent = content;

    // Timestamp row
    const meta = document.createElement('div');
    meta.className = 'message__meta';
    meta.textContent = time;

    wrap.appendChild(bubble);
    wrap.appendChild(meta);

    // Optional translated bubble for bot messages
    // Mirrors: if translated_answer: append second assistant message
    if (!isUser && translatedContent && language) {
      const translatedBubble = document.createElement('div');
      translatedBubble.className = 'message__bubble message__bubble--translated';

      // Label pill: "Answer (Hindi)" — mirrors f"Answer ({language})"
      const label = document.createElement('span');
      label.className = 'summary-block__label';
      label.style.cssText = 'display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:99px;background:#dcfce7;color:#15803d;margin-bottom:6px;';
      label.textContent = `Answer (${language})`;

      const translatedText = document.createElement('p');
      translatedText.style.cssText = 'font-size:14px;line-height:1.65;white-space:pre-wrap;word-wrap:break-word;';
      translatedText.textContent = translatedContent;

      translatedBubble.appendChild(label);
      translatedBubble.appendChild(translatedText);
      wrap.appendChild(translatedBubble);
    }

    // Assemble row
    row.appendChild(avatar);
    row.appendChild(wrap);

    els.chatMessages.appendChild(row);
    _scrollToBottom();

    return row;
  }

  /**
   * Re-render the entire message history from State.
   * Mirrors: st.rerun() re-renders all messages from session_state.messages
   *
   * @param {Array}  messages  — from State.get('messages')
   * @param {string} language
   */
  function renderAllMessages(messages, language) {
    els.chatMessages.innerHTML = '';
    messages.forEach(msg => {
      appendMessage(msg.role, msg.content, msg.translatedContent, language);
    });
  }

  // -------------------------------------------------------
  // CHAT SCREEN — TYPING INDICATOR
  // -------------------------------------------------------

  /**
   * Insert an animated typing indicator bubble.
   * Mirrors: st.spinner("Thinking...")
   * @returns {HTMLElement} — reference so it can be removed
   */
  function showTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'message message--bot typing-indicator';
    row.id = 'typing-indicator';

    const avatar = _makeBotAvatar();
    const wrap   = document.createElement('div');
    wrap.className = 'message__bubble-wrap';

    const bubble = document.createElement('div');
    bubble.className = 'message__bubble';

    [0, 1, 2].forEach(() => {
      const dot = document.createElement('span');
      dot.className = 'typing-dot';
      dot.setAttribute('aria-hidden', 'true');
      bubble.appendChild(dot);
    });

    wrap.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(wrap);
    row.setAttribute('aria-label', 'Assistant is thinking');

    els.chatMessages.appendChild(row);
    _scrollToBottom();
    return row;
  }

  /** Remove the typing indicator from the DOM */
  function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  // -------------------------------------------------------
  // CHAT SCREEN — INPUT BAR STATE
  // -------------------------------------------------------

  /**
   * Enable or disable the chat input + send button.
   * Mirrors: st.chat_input being available only after upload.
   */
  function setChatInputEnabled(enabled) {
    els.chatInput.disabled = !enabled;
    els.sendBtn.disabled   = !enabled;
  }

  /**
   * Show the spinner in the send button while waiting for answer.
   * Mirrors: with st.spinner("Thinking...")
   */
  function setSendBtnLoading(loading) {
    if (loading) {
      els.sendIcon.classList.add('hidden');
      els.sendSpinner.classList.remove('hidden');
      els.sendBtn.disabled  = true;
      els.chatInput.disabled = true;
    } else {
      els.sendIcon.classList.remove('hidden');
      els.sendSpinner.classList.add('hidden');
      els.sendBtn.disabled  = false;
      els.chatInput.disabled = false;
    }
  }

  /** Clear the chat input field */
  function clearChatInput() {
    els.chatInput.value = '';
  }

  /** Clear all rendered chat messages */
  function clearChatMessages() {
    els.chatMessages.innerHTML = '';
  }

  // -------------------------------------------------------
  // SCROLL HELPER
  // -------------------------------------------------------

  function _scrollToBottom() {
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  }

  // -------------------------------------------------------
  // PUBLIC INTERFACE
  // -------------------------------------------------------
  return {
    // Element references (for event binding in other modules)
    els,

    // Screen transitions
    showUploadScreen,
    showChatScreen,

    // Upload screen
    populateLanguages,
    showFilePreview,
    clearFilePreview,
    setUploadBtnEnabled,
    setUploadBtnLoading,
    setDropZoneDragOver,
    showUploadError,
    clearUploadError,

    // Chat screen
    populateFileInfoBar,
    clearFileInfoBar,
    renderSummary,
    clearSummary,
    appendMessage,
    renderAllMessages,
    showTypingIndicator,
    removeTypingIndicator,
    setChatInputEnabled,
    setSendBtnLoading,
    clearChatInput,
    clearChatMessages,
    showChatError,
    clearChatError,
  };

})();
