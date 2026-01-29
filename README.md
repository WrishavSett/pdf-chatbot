# PDF RAG Chatbot

A Python-based chatbot that allows users to upload PDF documents, automatically generate a summary, and interact with the document using Retrieval-Augmented Generation (RAG). The chatbot streams responses in real-time and supports session-scoped memory.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Installation](#installation)
6. [Environment Setup](#environment-setup)
7. [Running the Application](#running-the-application)
8. [Usage](#usage)
9. [File Structure](#file-structure)
10. [Notes & Recommendations](#notes--recommendations)
11. [License](#license)

---

## Overview

This project enables users to:

* Upload a PDF document.
* Automatically generate a concise, high-level summary.
* Ask questions about the document content and receive answers grounded strictly in the PDF.
* Stream answers token-by-token for real-time interaction.
* Start a new chat session that clears memory, uploaded documents, and embeddings.

The system is designed with session-scoped memory only — no persistent storage across sessions.

---

## Features

* **PDF Upload & Parsing:** Supports PDF files, splits content into manageable chunks.
* **Summary Generation:** Generates a concise summary immediately after upload.
* **RAG QA:** Uses vector embeddings and LangChain to answer questions based strictly on document content.
* **Streaming Responses:** Answers are streamed token-by-token to the frontend.
* **Session Management:**

  * Memory, embeddings, and uploaded PDFs are cleared on “New Chat” or tab/browser close.
* **Frontend:** Streamlit UI with chat interface.

---

## Architecture

```
+----------------+       +-----------------------+       +---------------------+
|  Streamlit UI  | <---> |  Flask API (Waitress) | <---> | Core AI / LangChain |
+----------------+       +-----------------------+       +---------------------+
        |                        |                           |
        | Upload PDF / Ask Qs    | Handles endpoints:        |
        |                        | /upload, /chat, /reset    |
        |                        |                           |
        v                        v                           v
  Upload & Chat UI       Session-scoped AISession       LLMs, Embeddings,
                         Vectorstore (Chroma)         LangGraph Nodes
```

* **Streamlit UI:** Upload PDFs, display summary, chat interface with streaming answers.
* **Flask API:** Handles PDF uploads, session state, RAG queries, and resets.
* **Core AI (`core_ai.py`):** PDF parsing, summary generation, RAG graph construction, streaming QA, and vector embeddings.

---

## Tech Stack

* **Python:** 3.11.2
* **Frontend:** Streamlit 1.53.1
* **Backend:** Flask 3.1.2, Flask-CORS 6.0.2, Waitress (for production-ready serving)
* **AI:** OpenAI API (ChatOpenAI, embeddings)
* **Libraries:**

  * LangChain 1.2.7
  * LangGraph 1.0.7
  * LangSmith 0.6.6
  * Chroma (vector database)
  * langchain_community (PyPDFLoader)
  * langchain_text_splitters (RecursiveCharacterTextSplitter)

---

## Installation

1. **Clone the repository:**

```bash
git clone <repo-url>
cd medical-report-chatbot
```

2. **Create virtual environment:**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source .venv/bin/activate  # Linux / macOS
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Environment Setup

1. **Create a `.env` file** in the project root with your OpenAI API key:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

2. **Verify key loading:**

```python
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))  # Should print your key
```

---

## Running the Application

### Start Backend API

```bash
python api.py
```

* Backend runs via Waitress on `http://localhost:8000`.
* Endpoints: `/upload`, `/chat`, `/reset`.

### Start Frontend

In a separate terminal:

```bash
streamlit run app.py
```

* Opens the Streamlit chat interface in your browser.

---

## Usage

1. **Upload PDF:** Click “Browse files” and select a PDF.
2. **View Summary:** Summary appears automatically after processing.
3. **Ask Questions:** Use the chat input to query the PDF.
4. **Streamed Answers:** Responses appear token-by-token in real-time.
5. **New Chat:** Click “New Chat” to reset session and start fresh.

---

## File Structure

```
medical-report-chatbot/
├── .env                       # OpenAI API key
├── core_ai.py                  # Core AI logic: PDF parsing, summary, RAG, streaming
├── api.py                      # Flask API endpoints: /upload, /chat, /reset
├── app.py                      # Streamlit frontend
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Notes & Recommendations

* Only **PDFs are supported**. Other file types will throw an error.
* Session memory is **cleared on reset or tab/browser close** — no persistent storage.
* Large PDFs may take longer to process; `/upload` request has `timeout=None` to handle them.
* Always ensure `.env` is correctly formatted (`UTF-8`, no quotes or trailing spaces).
* Use **Waitress** for Windows or production environments; Flask built-in server is only for development.

---

## License

MIT License. See `LICENSE` file for details.
