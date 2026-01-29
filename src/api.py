# api.py

# -------------------------
# Imports
# -------------------------

import os
import tempfile
from typing import Generator

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
# In-Memory Session Store
# -------------------------

_active_session: AISession | None = None

def get_session() -> AISession:
    if _active_session is None:
        raise RuntimeError("No active session found.")
    return _active_session

def set_session(session: AISession):
    global _active_session
    _active_session = session

def clear_session():
    global _active_session
    _active_session = None

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
    5. Returns summary
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
        set_session(session)

        return jsonify(
            {
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

    question = data["question"]

    try:
        session = get_session()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    def generate() -> Generator[str, None, None]:
        for token in stream_rag_answer(session, question):
            yield token

    # return Response(
    #     generate(),
    #     mimetype="text/plain"
    # )
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

    clear_session()
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
