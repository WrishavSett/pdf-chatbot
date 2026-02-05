# api.py

# -------------------------
# Imports
# -------------------------

import os
import uuid
import tempfile
from typing import Generator, Dict
import logging

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from core_ai import (
    AISession,
    initialize_session,
    stream_rag_answer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------
# Flask App Setup
# -------------------------

app = Flask(__name__)
CORS(app)

# -------------------------
# Session Store (Multiple Sessions)
# -------------------------

_sessions: Dict[str, AISession] = {}

def get_session(session_id: str) -> AISession:
    if session_id not in _sessions:
        raise RuntimeError("No active session found.")
    return _sessions[session_id]

def set_session(session_id: str, session: AISession):
    _sessions[session_id] = session

def clear_session(session_id: str):
    if session_id in _sessions:
        session = _sessions[session_id]
        # Explicitly delete vectorstore
        if session.vectorstore:
            try:
                session.vectorstore.delete_collection()
            except Exception:
                pass
        del _sessions[session_id]

# -------------------------
# Upload & Process PDF
# -------------------------

@app.route("/upload", methods=["POST"])
def upload_pdf():
    logging.info("Received request to upload PDF.")
    if "file" not in request.files:
        logging.warning("No file uploaded in the request.")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".pdf"):
        logging.warning("Uploaded file is not a PDF.")
        return jsonify({"error": "Only PDF files are supported"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        pdf_path = tmp.name

    try:
        logging.info("Initializing session for uploaded PDF.")
        session = initialize_session(pdf_path)
        session_id = str(uuid.uuid4())
        set_session(session_id, session)

        logging.info(f"Session initialized successfully with ID: {session_id}")
        return jsonify(
            {
                "session_id": session_id,
                "summary": session.summary
            }
        )
    except Exception as e:
        logging.error(f"Error processing uploaded PDF: {e}")
        return jsonify({"error": "Failed to process PDF"}), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# -------------------------
# Stream RAG QA
# -------------------------

@app.route("/chat", methods=["POST"])
def chat():
    logging.info("Received chat request.")
    data = request.get_json()

    if not data or "question" not in data:
        logging.warning("Chat request missing 'question' field.")
        return jsonify({"error": "Missing question"}), 400

    if "session_id" not in data:
        logging.warning("Chat request missing 'session_id' field.")
        return jsonify({"error": "Missing session_id"}), 400

    question = data["question"]
    session_id = data["session_id"]

    try:
        session = get_session(session_id)
        logging.info(f"Streaming response for session ID: {session_id}")

        def generate() -> Generator[str, None, None]:
            for token in stream_rag_answer(session, question):
                yield token

        return Response(
            stream_with_context(generate()),
            mimetype="text/plain"
        )
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error during chat processing: {e}")
        return jsonify({"error": "Failed to process chat request"}), 500

# -------------------------
# Reset Session
# -------------------------

@app.route("/reset", methods=["POST"])
def reset():
    logging.info("Received request to reset session.")
    data = request.get_json()

    if data and "session_id" in data:
        session_id = data["session_id"]
        clear_session(session_id)
        logging.info(f"Session {session_id} cleared successfully.")

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
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)