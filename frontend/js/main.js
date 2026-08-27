/**
 * main.js
 * --------
 * Application entry point.
 * Runs after all other scripts have loaded.
 *
 * Responsibilities:
 *   1. Fetch languages from GET /languages and populate the dropdown
 *      (mirrors: if "languages" not in st.session_state: fetch_languages())
 *   2. Initialise the Upload module event listeners
 *   3. Initialise the Chat module event listeners
 *   4. Show the upload screen (initial state)
 */

'use strict';

document.addEventListener('DOMContentLoaded', async () => {

  // ----------------------------------------------------------
  // 1. INITIALISE EVENT LISTENERS
  // Must happen before any async work so the UI is responsive.
  // ----------------------------------------------------------
  Upload.init();
  Chat.init();

  // ----------------------------------------------------------
  // 2. FETCH SUPPORTED LANGUAGES
  // Mirrors: st.session_state.languages = fetch_languages()
  // Populates the <select> on the upload screen.
  // Shows an error banner if the API is unreachable.
  // ----------------------------------------------------------
  try {
    const languages = await Api.fetchLanguages();
    Ui.populateLanguages(languages);
  } catch (err) {
    // Mirrors: st.error("Failed to fetch supported languages.")
    Ui.showUploadError(
      'Could not connect to the server. Please ensure the API is running and refresh the page.'
    );
  }

  // ----------------------------------------------------------
  // 3. ENSURE UPLOAD SCREEN IS SHOWN ON LOAD
  // ----------------------------------------------------------
  Ui.showUploadScreen();

});
