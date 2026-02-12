# app.py

# Imports
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")

# Session State initialization
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper functions
def reset_chat():
    if st.session_state.session_id:
        try:
            requests.post(
                f"{API_BASE_URL}/reset",
                json={"session_id": st.session_state.session_id}
            )
        except Exception as e:
            pass

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()

def upload_pdf(file):
    try:
        with st.spinner("Processing PDF. Please wait..."):
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files={"file": file},
                timeout=None
            )

        response.raise_for_status()
        data = response.json()
        return data["session_id"], data["summary"]
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
def get_answer(question: str) -> str:
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "question": question,
                "session_id": st.session_state.session_id
            },
            timeout=None
        )

        response.raise_for_status()
        data = response.json()
        return data["answer"]
        
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

    if uploaded_file is not None:
        session_id, summary = upload_pdf(uploaded_file)

        st.session_state.uploaded = True
        st.session_state.session_id = session_id
        st.session_state.summary = summary

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
                answer = get_answer(user_input)
            st.markdown(answer)

        # st.session_state.messages.append(
        #     {"role": "assistant", "content": streamed_text}
        # )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )