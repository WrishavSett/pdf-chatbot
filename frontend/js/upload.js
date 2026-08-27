/**
 * upload.js
 * ----------
 * Handles everything on the upload screen:
 *   - File selection via button click
 *   - File selection via drag-and-drop
 *   - Client-side file validation (type + size)
 *   - Triggering POST /upload and transitioning to the chat screen
 *
 * Mirrors the upload half of app.py exactly:
 *   - file type restriction (pdf only)
 *   - MAX_FILE_SIZE check (20 MB)
 *   - upload_pdf() call → session_id, summary, translated_summary
 *   - state population after success
 *   - st.rerun() → showChatScreen()
 */

'use strict';

const Upload = (() => {

  // 20 MB in bytes — matches MAX_FILE_SIZE in config.py
  const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;

  // Currently selected File object (or null)
  let _selectedFile = null;

  // -------------------------------------------------------
  // FILE VALIDATION
  // Mirrors app.py: type check + size check before upload call
  // -------------------------------------------------------

  /**
   * @param {File} file
   * @returns {string|null} Error message, or null if valid
   */
  function _validateFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf') || file.type !== 'application/pdf') {
      return 'Only PDF files are supported. Please choose a .pdf file.';
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      // Mirrors: st.error("File too large. Maximum allowed file size is 20MB.")
      return 'File too large. Maximum allowed file size is 20 MB.';
    }
    return null;
  }

  // -------------------------------------------------------
  // FILE SELECTION — shared handler for input + drag-drop
  // -------------------------------------------------------

  function _onFileSelected(file) {
    if (!file) return;

    Ui.clearUploadError();

    const error = _validateFile(file);
    if (error) {
      Ui.showUploadError(error);
      _selectedFile = null;
      Ui.clearFilePreview();
      Ui.setUploadBtnEnabled(false);
      return;
    }

    _selectedFile = file;
    Ui.showFilePreview(file.name, file.size);
    Ui.setUploadBtnEnabled(true);
  }

  // -------------------------------------------------------
  // UPLOAD FLOW
  // Mirrors upload_pdf() + post-upload state assignments in app.py
  // -------------------------------------------------------

  async function _handleUpload() {
    if (!_selectedFile) return;

    const language = Ui.els.languageSelect.value; // '' means no translation

    Ui.clearUploadError();
    Ui.setUploadBtnLoading(true);

    try {
      // POST /upload — mirrors upload_pdf() in app.py
      const data = await Api.uploadPdf(_selectedFile, language);

      // Store state — mirrors st.session_state assignments in app.py
      State.set('uploaded',   true);
      State.set('sessionId',  data.session_id);
      State.set('summary',    data.summary);
      State.set('language',   language);

      // Capture file info client-side (not available from API)
      State.set('fileName',   _selectedFile.name);
      State.set('fileSize',   _selectedFile.size);

      // Transition to chat screen — mirrors st.rerun() after upload
      _enterChatScreen(data.summary, data.translated_summary, language);

    } catch (err) {
      // Mirrors: st.error("Failed to upload and process PDF.")
      Ui.showUploadError(err.message || 'Failed to upload and process PDF.');
      Ui.setUploadBtnLoading(false);
      Ui.setUploadBtnEnabled(true);
    }
  }

  // -------------------------------------------------------
  // TRANSITION TO CHAT SCREEN
  // Mirrors app.py's post-upload st.rerun() behaviour
  // -------------------------------------------------------

  function _enterChatScreen(summary, translatedSummary, language) {
    // Populate the file info bar (client-side data)
    Ui.populateFileInfoBar(
      State.get('fileName'),
      State.get('fileSize'),
      language
    );

    // Render summary blocks
    Ui.renderSummary(summary, translatedSummary, language);

    // Enable chat input
    Ui.setChatInputEnabled(true);

    // Switch screen
    Ui.showChatScreen();

    // Append the opening bot greeting AFTER screen switch
    // Mirrors: first message in chat — "Hello! I've analyzed your document..."
    const greeting = "Hello! I've analyzed your document. You can now ask me any questions about it.";
    State.addMessage('assistant', greeting);
    Ui.appendMessage('assistant', greeting);
  }

  // -------------------------------------------------------
  // DRAG AND DROP
  // -------------------------------------------------------

  function _initDragDrop() {
    const dropZone = Ui.els.dropZone;

    dropZone.addEventListener('dragenter', e => {
      e.preventDefault();
      Ui.setDropZoneDragOver(true);
    });

    dropZone.addEventListener('dragover', e => {
      e.preventDefault(); // required to allow drop
      Ui.setDropZoneDragOver(true);
    });

    dropZone.addEventListener('dragleave', e => {
      // Only remove if leaving the drop zone itself (not a child)
      if (!dropZone.contains(e.relatedTarget)) {
        Ui.setDropZoneDragOver(false);
      }
    });

    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      Ui.setDropZoneDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) {
        // Sync the native input so the same validation path is used
        _onFileSelected(file);
      }
    });

    // Keyboard accessibility for the drop zone (role="button")
    dropZone.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        Ui.els.fileInput.click();
      }
    });
  }

  // -------------------------------------------------------
  // EVENT BINDINGS
  // -------------------------------------------------------

  function init() {
    // "Choose PDF File" button → open native file picker
    Ui.els.chooseFileBtn.addEventListener('click', () => {
      Ui.els.fileInput.click();
    });

    // Native file input change
    Ui.els.fileInput.addEventListener('change', () => {
      const file = Ui.els.fileInput.files[0];
      _onFileSelected(file || null);
      // Reset input value so re-selecting the same file triggers change
      Ui.els.fileInput.value = '';
    });

    // Remove selected file
    Ui.els.removeFileBtn.addEventListener('click', e => {
      e.stopPropagation(); // don't trigger drop zone click
      _selectedFile = null;
      Ui.clearFilePreview();
      Ui.setUploadBtnEnabled(false);
      Ui.clearUploadError();
    });

    // Upload & Process button
    Ui.els.uploadBtn.addEventListener('click', _handleUpload);

    // Drag and drop
    _initDragDrop();
  }

  function reset() {
    _selectedFile = null;
    Ui.els.fileInput.value = '';
    Ui.els.languageSelect.value = '';
    Ui.clearFilePreview();
    Ui.clearUploadError();
    Ui.setUploadBtnLoading(false);
    Ui.setUploadBtnEnabled(false);
  }

  return { init, reset };

})();
