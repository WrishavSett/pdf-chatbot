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
    """
    1. Accepts PDF upload
    2. Parses PDF
    3. Generates summary
    4. Builds vector store
    5. Returns summary and session_id
    """

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save PDF temporarily (session-scoped)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        pdf_path = tmp.name

    try:
        session = initialize_session(pdf_path)
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        set_session(session_id, session)

        return jsonify(
            {
                "session_id": session_id,
                "summary": session.summary
            }
        )

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# -------------------------
# Stream RAG QA
# -------------------------

@app.route("/chat", methods=["POST"])
def chat():
    """
    Streams QA responses token-by-token.
    """

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    if "session_id" not in data:
        return jsonify({"error": "Missing session_id"}), 400

    question = data["question"]
    session_id = data["session_id"]

    try:
        session = get_session(session_id)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    def generate() -> Generator[str, None, None]:
        for token in stream_rag_answer(session, question):
            yield token

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain"
    )

# -------------------------
# Reset Session
# -------------------------

@app.route("/reset", methods=["POST"])
def reset():
    """
    Clears all session data:
    - PDF
    - Vector store
    - Memory
    """

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
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)