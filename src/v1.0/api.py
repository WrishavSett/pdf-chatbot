# api.py

# -------------------------
# Imports
# -------------------------

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
    stream_rag_answer
)

# -------------------------
# Flask App Setup
# -------------------------

app = Flask(__name__)
CORS(app)

# -------------------------
# Session Store with TTL
# -------------------------

SESSION_TTL_SECONDS = 900
'''
Structure:
_sessions = {
    session_id (str): {
    "session": AISession (object),
    "last_accessed": timestamp (datetime)
    }
}
'''

_sessions: Dict[str, dict] = {}
_sessions_lock = threading.Lock()

def get_session(session_id: str) -> AISession:
    with _sessions_lock:
        if session_id not in _sessions:
            raise RuntimeError("No active session found.")

        # Refresh TTL on access
        _sessions[session_id]["last_accessed"] = time.time()
        return _sessions[session_id]["session"]

def set_session(session_id: str, session: AISession):
    with _sessions_lock:
        _sessions[session_id] = {
            "session": session,
            "last_accessed": time.time()
        }

def clear_session(session_id: str):
    with _sessions_lock:
        if session_id in _sessions:
            session = _sessions[session_id]["session"]

            # Explicitly delete vectorstore collection
            if session.vectorstore:
                try:
                    session.vectorstore.delete_collection()
                except Exception:
                    pass

            del _sessions[session_id]

    gc.collect()

# -------------------------
# Background Cleanup Thread
# -------------------------

def cleanup_expired_sessions():
    while True:
        time.sleep(60)  # Check every 60 seconds
        now = time.time()

        with _sessions_lock:
            expired_sessions = [
                sid for sid, data in _sessions.items()
                if now - data["last_accessed"] > SESSION_TTL_SECONDS
            ]

        for sid in expired_sessions:
            clear_session(sid)

# -------------------------
# Upload & Process PDF
# -------------------------

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        pdf_path = tmp.name

    try:
        session = initialize_session(pdf_path)
        session_id = str(uuid.uuid4())
        set_session(session_id, session)

        return jsonify(
            {
                "session_id": session_id,
                "summary": session.summary
            }
        )
    except Exception as e:
        return jsonify({"error": "Failed to process PDF"}), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# -------------------------
# Stream RAG QA
# -------------------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    if "session_id" not in data:
        return jsonify({"error": "Missing session_id"}), 400

    question = data["question"]
    session_id = data["session_id"]

    try:
        session = get_session(session_id)

        def generate() -> Generator[str, None, None]:
            for token in stream_rag_answer(session, question):
                yield token

        return Response(
            stream_with_context(generate()),
            mimetype="text/plain"
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Failed to process chat request"}), 500

# -------------------------
# Reset Session
# -------------------------

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json()

    if data and "session_id" in data:
        session_id = data["session_id"]
        clear_session(session_id)

    return jsonify({"status": "reset successful"})

# -------------------------
# Local Dev Entry Point
# -------------------------

# if __name__ == "__main__":
#     app.run(
#         host="0.0.0.0",
#         port=8000,
#         debug=True
#     )

if __name__ == "__main__":
    # Start background cleanup thread
    cleanup_thread = threading.Thread(
        target=cleanup_expired_sessions,
        daemon=True
    )
    cleanup_thread.start()

    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)