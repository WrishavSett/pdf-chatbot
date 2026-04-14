# app.py

# Imports
import requests
import streamlit as st

# Setup logging
from logger import get_logger
logger = get_logger("app")

from config import (
    API_BASE_URL,
    MAX_FILE_SIZE,
)

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")

# Helper functions
def fetch_languages() -> list:
    try:
        response = requests.get(f"{API_BASE_URL}/languages")
        response.raise_for_status()
        return response.json()["languages"]
    except Exception as e:
        logger.exception("Failed to fetch supported languages.")
        st.error("Failed to fetch supported languages.")
        raise

# Session State initialization
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "languages" not in st.session_state:
    st.session_state.languages = fetch_languages()

# Helper functions
def reset_chat():
    session_id = st.session_state.session_id
    logger.info("Reset initiated for session_id=%s", session_id)

    if session_id:
        try:
            requests.post(
                f"{API_BASE_URL}/reset",
                json={"session_id": session_id}
            )
            logger.info("Reset API call successful for session_id=%s", session_id)
        except Exception as e:
            logger.warning("Reset API call failed for session_id=%s", session_id, exc_info=True)

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    logger.debug("Streamlit session state cleared")
    st.rerun()

def upload_pdf(file, language: str | None):
    try:
        form_data = {"file": file}
        extra_fields = {}

        if language and language != "None":
            extra_fields["language"] = language

        with st.spinner("Processing PDF. Please wait..."):
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=form_data,
                data=extra_fields,
                timeout=None
            )

        response.raise_for_status()
        data = response.json()
        return data["session_id"], data["summary"], data.get("translated_summary")
    except Exception as e:
        st.error("Failed to upload and process PDF.")
        raise

# # RAG QA (Streaming)
# def stream_answer(question):
#     try:
#         response = requests.post(
#             f"{API_BASE_URL}/chat",
#             json={
#                 "question": question,
#                 "session_id": st.session_state.session_id
#             },
#             stream=True
#         )

#         response.raise_for_status()

#         for chunk in response.iter_content(chunk_size=None):
#             if chunk:
#                 yield chunk.decode("utf-8")
#     except Exception as e:
#         st.error("Failed to get a response from the server.")
#         raise

# RAG QA (Non-Streaming)
def get_answer(question: str) -> tuple[str, str | None]:
    try:
        language = st.session_state.get("language", "None")

        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }

        if language and language != "None":
            payload["language"] = language

        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=None
        )

        response.raise_for_status()
        data = response.json()
        return data["answer"], data.get("translated_answer")

    except Exception as e:
        st.error("Failed to get a response from the server.")
        raise

# UI Rendering
st.title("PDF Chatbot")

# New Chat Button (only after upload)
if st.session_state.uploaded:
    if st.button("New Chat"):
        reset_chat()

# Upload Screen
if not st.session_state.uploaded:
    st.subheader("Upload a PDF to begin")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False
    )

    languages = st.session_state.languages
    options = ["None"] + languages
    selected_language = st.selectbox("Translate summary to (optional)", options)

    submit = st.button("Upload", disabled=uploaded_file is None)

    if submit and uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE:
            logger.warning(
                "File too large (filename=%s, size=%d). Maximum allowed file size is 20MB.",
                uploaded_file.name, uploaded_file.size
            )
            st.error("File too large. Maximum allowed file size is 20MB.")
        else:
            session_id, summary, translated_summary = upload_pdf(uploaded_file, selected_language)

            st.session_state.uploaded = True
            st.session_state.session_id = session_id
            st.session_state.summary = summary
            st.session_state.language = selected_language

            if selected_language and selected_language != "None":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Summary (English)**\n\n{summary}"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Summary ({selected_language})**\n\n{translated_summary}"
                })
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": summary}
                )

            st.rerun()

# Chat Screen
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the document")

    if user_input:
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # # Streaming response
        # with st.chat_message("assistant"):
        #     placeholder = st.empty()
        #     streamed_text = ""

        #     for token in stream_answer(user_input):
        #         streamed_text += token
        #         placeholder.markdown(streamed_text)

        # Non-Streaming response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, translated_answer = get_answer(user_input)
            st.markdown(answer)

        # st.session_state.messages.append(
        #     {"role": "assistant", "content": streamed_text}
        # )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        if translated_answer:
            language = st.session_state.get("language", "None")
            translated_content = f"**Answer ({language})**\n\n{translated_answer}"

            with st.chat_message("assistant"):
                st.markdown(translated_content)

            st.session_state.messages.append(
                {"role": "assistant", "content": translated_content}
            )