# api.py

# -------------------------
# Imports
# -------------------------

import os
import uuid
import tempfile
from typing import Dict

from flask import Flask, request, jsonify
from flask_cors import CORS

from core_ai import (
    AISession,
    initialize_session,
    get_rag_answer
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
# LangGraph RAG QA
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
        
        # Use LangGraph to get the answer
        answer = get_rag_answer(session, question)
        
        return jsonify({"answer": answer})
        
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
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)