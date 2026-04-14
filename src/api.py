# api.py

# Imports
import os
import uuid
import tempfile
from typing import Generator, Dict

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

import gc
import time
import threading

from core_ai import (
    AISession,
    initialize_session,
    translate_text,
    # stream_rag_answer,
    get_rag_answer
)

# Setup logging
from logger import get_logger
logger = get_logger("api")

from config import (
    SESSION_TTL,
    MAX_FILE_SIZE,
    FLASK_HOST,
    FLASK_PORT,
    CLEANUP_INTERVAL,
)

# Supported translation languages
SUPPORTED_LANGUAGES = [
    "Spanish", "French", "German", "Hindi", "Bengali",
    "Arabic", "Chinese (Simplified)", "Japanese", "Portuguese", "Russian"
]

# Flask app setup
app = Flask(__name__)
CORS(app)

# Session Store with TTL
SESSION_TTL_SECONDS = SESSION_TTL
# Structure:
# _sessions = {
#     session_id (str): {
#     "session": AISession (object),
#     "last_accessed": timestamp (datetime)
#     }
# }

_sessions: Dict[str, dict] = {}
_sessions_lock = threading.Lock()

def get_session(session_id: str) -> AISession:
    with _sessions_lock:
        if session_id not in _sessions:
            logger.warning("Session not found (session_id=%s)", session_id)
            raise RuntimeError("No active session found.")

        _sessions[session_id]["last_accessed"] = time.time()
        return _sessions[session_id]["session"]

def set_session(session_id: str, session: AISession):
    with _sessions_lock:
        _sessions[session_id] = {
            "session": session,
            "last_accessed": time.time()
        }
    
    logger.info("Session %s registered", session_id)
    logger.info("Current active sessions=%d", len(_sessions))

def _destroy_chroma_collection(session_id: str, session: AISession):
    if session.vectorstore:
        try:
            session.vectorstore.delete_collection()
            logger.debug("Chroma collection deleted for session_id=%s", session_id)
        except Exception:
            logger.exception("Failed to delete Chroma collection (session_id=%s)", session_id)
    gc.collect()

def clear_session(session_id: str):
    session_to_delete = None

    with _sessions_lock:
        data = _sessions.get(session_id)
        if not data:
            logger.warning("clear_session() called for unknown session_id=%s", session_id)
            return

        session_to_delete = data["session"]
        del _sessions[session_id]
    
    logger.info("Session %s cleared", session_id)
    logger.info("Current active sessions=%d", len(_sessions))

    _destroy_chroma_collection(session_id, session_to_delete)

# Background Cleanup Thread
def cleanup_expired_sessions():
    while True:
        time.sleep(CLEANUP_INTERVAL)
        now = time.time()
        sessions_to_delete = []

        with _sessions_lock:
            for sid, data in list(_sessions.items()):
                if now - data["last_accessed"] > SESSION_TTL_SECONDS:
                    sessions_to_delete.append((sid, data["session"]))
                    del _sessions[sid]

        for sid, session in sessions_to_delete:
            logger.warning("Expiring session (TTL exceeded) (session_id=%s)", sid)
            _destroy_chroma_collection(sid, session)

# Start cleanup thread at module level
cleanup_thread = threading.Thread(
    target=cleanup_expired_sessions,
    daemon=True
)
cleanup_thread.start()
logger.info("Session cleanup thread started with TTL set to %ds", SESSION_TTL_SECONDS)

# Upload and Process PDF
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File type not supported. Only PDF files are allowed."}), 415
    
    # Exact file size check
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return jsonify({
            "error": "File too large. Maximum allowed file size is 20MB."
        }), 413

    language = request.form.get("language")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        pdf_path = tmp.name

    try:
        session = initialize_session(pdf_path)
        session_id = str(uuid.uuid4())
        set_session(session_id, session)

        response_payload = {
            "session_id": session_id,
            "summary": session.summary
        }

        if language and language in SUPPORTED_LANGUAGES:
            logger.info("Translating summary to %s during upload (session_id=%s)", language, session_id)
            translated_summary = translate_text(session._llm, session.summary, language)
            response_payload["translated_summary"] = translated_summary

        return jsonify(response_payload), 201

    except Exception as e:
        logger.exception("Failed to process uploaded PDF (filename=%s)", file.filename)
        return jsonify({"error": "Failed to process PDF"}), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# # RAG QA (Streaming)
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()

#     if not data or "question" not in data:
#         return jsonify({"error": "Missing question"}), 400

#     if "session_id" not in data:
#         return jsonify({"error": "Missing session_id"}), 400

#     question = data["question"]
#     session_id = data["session_id"]

#     try:
#         session = get_session(session_id)

#         def generate() -> Generator[str, None, None]:
#             for token in stream_rag_answer(session, question):
#                 yield token

#         return Response(
#             stream_with_context(generate()),
#             mimetype="text/plain"
#         )
#     except RuntimeError as e:
#         return jsonify({"error": str(e)}), 404
#     except Exception as e:
#         logger.exception("Failed to process chat request (session_id=%s)", session_id)
#         return jsonify({"error": "Failed to process chat request"}), 500

# RAG QA (Non-Streaming)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    if "session_id" not in data:
        return jsonify({"error": "Missing session_id"}), 400

    question = data["question"]
    session_id = data["session_id"]
    language = data.get("language")  # optional

    try:
        session = get_session(session_id)
        answer = get_rag_answer(session, question)

        response_payload = {"answer": answer}

        if language and language in SUPPORTED_LANGUAGES:
            translated_answer = translate_text(session._llm, answer, language)
            response_payload["translated_answer"] = translated_answer

        return jsonify(response_payload), 200
        
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Failed to process chat request (session_id=%s)", session_id)
        return jsonify({"error": "Failed to process chat request"}), 500

# Reset session
@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json()

    if data and "session_id" in data:
        session_id = data["session_id"]
        clear_session(session_id)

    return jsonify({"status": "Reset successful"}), 200

# Get supported languages
@app.route("/languages", methods=["GET"])
def get_languages():
    return jsonify({"languages": SUPPORTED_LANGUAGES}), 200

# Local Dev entry point
# if __name__ == "__main__":
#     app.run(
#         host="0.0.0.0",
#         port=8000,
#         debug=True
#     )

if __name__ == "__main__":
    from waitress import serve
    serve(app, host=FLASK_HOST, port=FLASK_PORT)