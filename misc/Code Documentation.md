# Code Documentation

This document provides a detailed, line-by-line explanation of the Python source code for the Medical Report Chatbot application.

---

## `src/api.py`

This file creates the backend web server using Flask. It exposes API endpoints that the frontend can call.

```python
# api.py

# -------------------------
# Imports
# -------------------------

import os
# Line 7: Imports the 'os' module, which provides a way to interact with the operating system, like removing files.

import tempfile
# Line 10: Imports the 'tempfile' module, used to create temporary files. This is useful for storing the uploaded PDF before processing.

from typing import Generator
# Line 13: Imports the 'Generator' type for type hinting, indicating that a function is a generator (i.e., it uses 'yield').

from flask import Flask, request, jsonify, Response, stream_with_context
# Line 15: Imports necessary components from the Flask framework.
# - Flask: The main class to create a web application instance.
# - request: An object that holds all data from an incoming HTTP request (like files and JSON).
# - jsonify: A function to convert Python dictionaries into a JSON formatted response.
# - Response: A class to create a custom HTTP response, used here for streaming.
# - stream_with_context: A function that streams a response while keeping the context of the original request alive.

from flask_cors import CORS
# Line 21: Imports the CORS class from the Flask-CORS extension to handle Cross-Origin Resource Sharing. This allows the frontend (on a different origin) to make requests to this backend.

from core_ai import (
    AISession,
    initialize_session,
    stream_rag_answer
)
# Lines 23-28: Imports the core AI functionalities from the 'core_ai.py' file.
# - AISession: The class that encapsulates a user's session data (documents, vector store, etc.).
# - initialize_session: The function that processes a PDF and sets up a new 'AISession'.
# - stream_rag_answer: The generator function that produces the AI's answer token-by-token.

# -------------------------
# Flask App Setup
# -------------------------

app = Flask(__name__)
# Line 34: Creates an instance of the Flask web application. '__name__' is a special Python variable that gets the name of the current file.

CORS(app)
# Line 35: Applies CORS to the Flask app, allowing web pages from any domain to make requests to this API.

# -------------------------
# In-Memory Session Store
# -------------------------

_active_session: AISession | None = None
# Line 41: Declares a global variable to hold the current AI session. It's initialized to 'None' and serves as a simple, single-user in-memory database.

def get_session() -> AISession:
# Line 43: Defines a function to safely retrieve the active session.
    if _active_session is None:
# Line 44: Checks if a session has been created.
        raise RuntimeError("No active session found.")
# Line 45: If no session exists, it raises an error.
    return _active_session
# Line 46: If a session exists, it returns the session object.

def set_session(session: AISession):
# Line 48: Defines a function to set the global session variable.
    global _active_session
# Line 49: Declares that this function intends to modify the global '_active_session' variable.
    _active_session = session
# Line 50: Assigns the provided session object to the global variable.

def clear_session():
# Line 52: Defines a function to clear the session.
    global _active_session
# Line 53: Declares intent to modify the global variable.
    _active_session = None
# Line 54: Resets the global session variable to 'None', effectively deleting the session.

# -------------------------
# Upload & Process PDF
# -------------------------

@app.route("/upload", methods=["POST"])
# Line 60: A Flask decorator that registers the 'upload_pdf' function to handle HTTP POST requests made to the '/upload' URL.
def upload_pdf():
# Line 61: Defines the function that will execute when a request hits the '/upload' endpoint.
    """ Docstring explaining the function's purpose. """

    if "file" not in request.files:
# Line 69: Checks if the incoming request has a file attached with the key "file".
        return jsonify({"error": "No file uploaded"}), 400
# Line 70: If no file is found, it returns a JSON error message with a 400 (Bad Request) status code.

    file = request.files["file"]
# Line 72: Retrieves the file object from the request.

    if not file.filename.lower().endswith(".pdf"):
# Line 74: Checks if the uploaded file's name ends with ".pdf" (case-insensitive).
        return jsonify({"error": "Only PDF files are supported"}), 400
# Line 75: If the file is not a PDF, returns an error.

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# Line 78: Creates a temporary file on the server's disk to store the PDF content. 'delete=False' prevents it from being deleted immediately when the 'with' block is exited.
        file.save(tmp.name)
# Line 79: Saves the content of the uploaded file to this temporary file.
        pdf_path = tmp.name
# Line 80: Gets the file path of the temporary file.

    try:
# Line 82: Starts a 'try' block to handle potential errors during file processing.
        session = initialize_session(pdf_path)
# Line 83: Calls the function from 'core_ai.py' to process the PDF and create a new AI session.
        set_session(session)
# Line 84: Sets the newly created session as the active one.

        return jsonify(
            {
                "summary": session.summary
            }
        )
# Lines 86-90: Returns the generated document summary as a JSON response with a 200 (OK) status code.

    finally:
# Line 92: The 'finally' block ensures this code runs whether the 'try' block succeeded or failed.
        if os.path.exists(pdf_path):
# Line 93: Checks if the temporary file still exists.
            os.remove(pdf_path)
# Line 94: Deletes the temporary file to clean up the server's disk space.

# -------------------------
# Stream RAG QA
# -------------------------

@app.route("/chat", methods=["POST"])
# Line 100: Registers the 'chat' function to handle POST requests to the '/chat' URL.
def chat():
# Line 101: Defines the function to handle chat requests.
    """ Docstring explaining the function's purpose. """

    data = request.get_json()
# Line 106: Parses the JSON data sent in the request body into a Python dictionary.

    if not data or "question" not in data:
# Line 108: Validates that the request body contains JSON and has a "question" key.
        return jsonify({"error": "Missing question"}), 400
# Line 109: Returns an error if the validation fails.

    question = data["question"]
# Line 111: Extracts the user's question from the dictionary.

    try:
# Line 113: Starts a 'try' block to handle the case where a session might not exist.
        session = get_session()
# Line 114: Retrieves the current AI session.
    except RuntimeError as e:
# Line 115: Catches the error thrown by 'get_session' if no session is active.
        return jsonify({"error": str(e)}), 400
# Line 116: Returns an error to the user.

    def generate() -> Generator[str, None, None]:
# Line 118: Defines a nested generator function. This function will 'yield' data piece by piece rather than returning it all at once.
        for token in stream_rag_answer(session, question):
# Line 119: Iterates through the stream of tokens (words or sub-words) produced by the core AI function.
            yield token
# Line 120: Yields each token, sending it immediately to the client.

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain"
    )
# Lines 126-129: Creates a streaming response. 'generate()' is the generator providing the content. 'stream_with_context' ensures the generator runs within the request context. 'mimetype="text/plain"' tells the browser to treat the response as plain text.

# -------------------------
# Reset Session
# -------------------------

@app.route("/reset", methods=["POST"])
# Line 135: Registers the 'reset' function to handle POST requests to the '/reset' URL.
def reset():
# Line 136: Defines the function to reset the session.
    """ Docstring explaining the function's purpose. """

    clear_session()
# Line 143: Calls the helper function to clear the global session variable.
    return jsonify({"status": "reset successful"})
# Line 144: Returns a JSON message confirming the reset was successful.

# -------------------------
# Local Dev Entry Point
# -------------------------

if __name__ == "__main__":
# Line 153: A standard Python entry point. This code block runs only when the script is executed directly (e.g., 'python src/api.py').
    from waitress import serve
# Line 154: Imports the 'serve' function from the 'waitress' library, a production-ready web server.
    serve(app, host="0.0.0.0", port=8000)
# Line 155: Starts the web server. It listens on port 8000 on all available network interfaces ('0.0.0.0'), making it accessible from other devices on the same network.
```

---

## `src/app.py`

This file creates the web-based user interface using the Streamlit library.

```python
# app.py

# -------------------------
# Imports
# -------------------------

import requests
# Line 7: Imports the 'requests' library, which is used to send HTTP requests to the backend API.

import streamlit as st
# Line 10: Imports the 'streamlit' library and assigns it the conventional alias 'st'.

API_BASE_URL = "http://localhost:8000"
# Line 12: Defines a constant for the base URL of the backend API, making it easy to change if needed.

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
# Line 14: Configures the basic properties of the web page, such as the title that appears in the browser tab and a centered page layout.

# -------------------------
# Session State Initialization
# -------------------------

if "uploaded" not in st.session_state:
# Line 20: Checks if the key "uploaded" has been initialized in Streamlit's session state. Session state is a dictionary-like object that persists across user interactions.
    st.session_state.uploaded = False
# Line 21: If not initialized, it sets the initial value of 'uploaded' to 'False'.

if "summary" not in st.session_state:
# Line 23: Checks if "summary" is in the session state.
    st.session_state.summary = None
# Line 24: If not, initializes it to 'None'.

if "messages" not in st.session_state:
# Line 26: Checks if "messages" is in the session state.
    st.session_state.messages = []
# Line 27: If not, initializes it as an empty list to store the chat message history.

# -------------------------
# Helper Functions
# -------------------------

def reset_chat():
# Line 33: Defines a function to reset the entire chat application state.
    try:
# Line 34: Starts a 'try' block to handle potential network errors.
        requests.post(f"{API_BASE_URL}/reset")
# Line 35: Sends an HTTP POST request to the backend's '/reset' endpoint to clear the AI session.
    except Exception:
# Line 36: If the request fails (e.g., the backend server is down), this block is executed.
        pass
# Line 37: 'pass' does nothing, effectively ignoring the error and proceeding.

    for key in list(st.session_state.keys()):
# Line 39: Iterates over a copy of all keys in the Streamlit session state.
        del st.session_state[key]
# Line 40: Deletes each key-value pair, clearing the session state.

    st.rerun()
# Line 42: Tells Streamlit to stop executing the current script and run it again from the top, which refreshes the UI to its initial state.

def upload_pdf(file):
# Line 44: Defines a function that sends the uploaded PDF to the backend.
    with st.spinner("Processing PDF. Please wait..."):
# Line 45: Displays a loading spinner with a message while the code inside this 'with' block runs.
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files={"file": file},
            timeout=None
        )
# Lines 46-50: Sends a POST request to the '/upload' endpoint.
# - 'files={"file": file}': Attaches the PDF file to the request.
# - 'timeout=None': Disables the request timeout, as processing a large PDF can take a long time.

    response.raise_for_status()
# Line 52: Checks if the request was successful. If the backend returned an error (e.g., 4xx or 5xx status code), this will raise an exception.
    return response.json()["summary"]
# Line 53: Parses the JSON response from the server and returns the value of the "summary" key.

def stream_answer(question):
# Line 55: Defines a generator function to stream the AI's answer from the backend.
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"question": question},
        stream=True
    )
# Lines 56-60: Sends a POST request to the '/chat' endpoint.
# - 'json={"question": question}': Sends the user's question as a JSON payload.
# - 'stream=True': This is critical. It tells 'requests' to start receiving the response immediately instead of waiting for the entire response to be downloaded.

    response.raise_for_status()
# Line 62: Checks for HTTP errors.

    for chunk in response.iter_content(chunk_size=None):
# Line 64: Iterates over the response content in chunks. 'chunk_size=None' lets the library choose an optimal chunk size.
        if chunk:
# Line 65: Checks if the chunk is not empty.
            yield chunk.decode("utf-8")
# Line 66: Decodes the binary chunk into a UTF-8 string and 'yields' it, passing it to the caller.

# -------------------------
# UI Rendering
# -------------------------

st.title("PDF Chatbot")
# Line 72: Renders the main title of the web page.

# --- New Chat Button (only after upload) ---
if st.session_state.uploaded:
# Line 75: This block of code only runs if a PDF has been successfully uploaded ('uploaded' is True).
    if st.button("New Chat"):
# Line 76: Renders a button labeled "New Chat". If the user clicks it, the 'if' condition becomes true.
        reset_chat()
# Line 77: Calls the function to reset the entire application.

# --- Upload Screen ---
if not st.session_state.uploaded:
# Line 80: This block runs only if a PDF has NOT been uploaded yet.
    st.subheader("Upload a PDF to begin")
# Line 81: Renders a subheader with instructions.

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False
    )
# Lines 83-87: Renders the file uploader widget. It's configured with a label, an allowed file type ('pdf'), and to only accept one file.

    if uploaded_file is not None:
# Line 89: This block runs when the user has selected a file in the uploader widget.
        summary = upload_pdf(uploaded_file)
# Line 90: Calls the helper function to send the file to the backend and get the summary.

        st.session_state.uploaded = True
# Line 92: Sets the 'uploaded' flag in the session state to 'True'.
        st.session_state.summary = summary
# Line 93: Stores the received summary in the session state.

        st.session_state.messages.append(
            {"role": "assistant", "content": summary}
        )
# Lines 95-97: Adds the summary to the chat history as the first message from the 'assistant'.

        st.rerun()
# Line 99: Reruns the script to switch from the upload view to the chat view.

# --- Chat Screen ---
else:
# Line 102: This block runs if 'st.session_state.uploaded' is 'True'.
    # Display chat history
    for message in st.session_state.messages:
# Line 104: Iterates through all the messages stored in the session state.
        with st.chat_message(message["role"]):
# Line 105: Renders a chat message container, using the 'role' ("user" or "assistant") to determine the avatar and style.
            st.markdown(message["content"])
# Line 106: Renders the content of the message. Using 'st.markdown' allows for rich text formatting.

    # Chat input
    user_input = st.chat_input("Ask a question about the document")
# Line 109: Renders the chat input box at the bottom of the screen. When the user types a message and presses Enter, the message is assigned to 'user_input'.

    if user_input:
# Line 111: This block runs only if the user has submitted a message.
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
# Lines 113-115: Appends the user's message to the chat history.

        with st.chat_message("user"):
# Line 117: Renders a new chat bubble for the user's message.
            st.markdown(user_input)
# Line 118: Displays the user's message inside the bubble.

        # Stream assistant response
        with st.chat_message("assistant"):
# Line 121: Renders a new chat bubble for the AI's response.
            placeholder = st.empty()
# Line 122: Creates an empty element on the page. This will be the target for our streaming text.
            streamed_text = ""
# Line 123: Initializes an empty string to accumulate the full response from the AI.

            for token in stream_answer(user_input):
# Line 125: Iterates through the tokens yielded by the 'stream_answer' function.
                streamed_text += token
# Line 126: Appends each new token to the 'streamed_text' variable.
                placeholder.markdown(streamed_text)
# Line 127: Updates the content of the 'placeholder' element with the latest version of 'streamed_text', creating the "typing" effect.

        st.session_state.messages.append(
            {"role": "assistant", "content": streamed_text}
        )
# Lines 129-131: After the loop finishes, appends the complete AI response to the chat history so it's displayed correctly on subsequent reruns.
```

---

## `src/core_ai.py`

This file contains the "brain" of the application, handling all interactions with the language model and document processing using the LangChain library.

```python
# core_ai.py

# -------------------------
# Imports
# -------------------------

import os
# Line 7: Imports the 'os' module to access environment variables.

from typing import List, Dict, Generator, Optional
# Line 9: Imports various type hints for better code readability and static analysis.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Line 11: Imports classes from LangChain's OpenAI integration.
# - ChatOpenAI: Interface to OpenAI's chat models (like GPT-4).
# - OpenAIEmbeddings: Interface to OpenAI's text embedding models.

from langchain_core.documents import Document
# Line 14: Imports the 'Document' class, a standard LangChain object for holding a piece of text and its associated metadata.

from langchain_core.messages import HumanMessage, AIMessage
# Line 15: Imports classes to represent messages in a conversation, used for chat history.

from langchain_core.prompts import ChatPromptTemplate
# Line 16: Imports a class for creating reusable and flexible prompt templates for chat models.

from langchain_core.output_parsers import StrOutputParser
# Line 17: Imports a simple parser that extracts the string content from a language model's output.

from langchain_community.document_loaders import PyPDFLoader
# Line 19: Imports a document loader from the community package that specifically handles loading and parsing text from PDF files.

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Line 20: Imports a text splitter that recursively tries to split text by a list of separators (like newline, space, etc.) to keep related text together.

from langchain_chroma import Chroma
# Line 21: Imports the 'Chroma' vector store integration, used to store document embeddings and perform similarity searches.

from langgraph.graph import StateGraph, END
# Line 23: Imports components for building stateful, cyclic graphs.
# - StateGraph: The main class for defining a graph of operations.
# - END: A special node indicating that the graph's execution is complete.

# -------------------------
# Load API Key
# -------------------------

from dotenv import load_dotenv
# Line 30: Imports the function to load environment variables from a file.
load_dotenv()
# Line 31: Executes the function, which looks for a '.env' file in the project directory and loads its key-value pairs into the environment.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Line 33: Retrieves the value of the "OPENAI_API_KEY" from the environment variables.

# -------------------------
# Session State Definition
# -------------------------

class ChatState(dict):
# Line 39: Defines a class 'ChatState' that inherits from Python's built-in 'dict'. This will be the state object that is passed between nodes in our LangGraph.
    """ Docstring explaining the expected keys in the state dictionary. """
    pass

# -------------------------
# Core AI Session Container
# -------------------------

class AISession:
# Line 49: Defines a class to hold all the AI-related objects for a single user session.
    """ Docstring explaining the class's purpose. """

    def __init__(self):
# Line 55: The constructor method, called when a new 'AISession' object is created.
        self.documents: List[Document] = []
# Line 56: Initializes an instance variable to hold the list of document chunks from the PDF.
        self.vectorstore: Optional[Chroma] = None
# Line 57: Initializes the vector store variable. It will be assigned a 'Chroma' instance later.
        self.summary: Optional[str] = None
# Line 58: Initializes the summary variable.
        self.chat_history: List = []
# Line 59: Initializes a list to store the history of the conversation.

        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
# Lines 61-66: Creates a non-streaming instance of the language model for tasks like summarization. 'temperature=0' makes the output highly deterministic.

        self._streaming_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=True,
            api_key=OPENAI_API_KEY
        )
# Lines 68-74: Creates a separate, streaming-enabled instance of the language model for the interactive chat.

        self._embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY
        )
# Lines 76-78: Creates an instance of the embeddings model, which will be used to convert text chunks into numerical vectors.

        self._rag_graph = None
# Line 80: Initializes the LangGraph instance variable.

# -------------------------
# PDF Processing
# -------------------------

def load_and_split_pdf(pdf_path: str) -> List[Document]:
# Line 86: Defines a function to load a PDF and split it into chunks.
    loader = PyPDFLoader(pdf_path)
# Line 87: Creates an instance of the PDF loader with the provided file path.
    docs = loader.load()
# Line 88: Loads the PDF, with each page typically becoming a separate 'Document' object.

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
# Lines 90-93: Creates an instance of the text splitter. It will create chunks of up to 1000 characters, with an overlap of 150 characters between chunks to preserve context.

    return splitter.split_documents(docs)
# Line 95: Splits the loaded documents into smaller chunks and returns them as a list.

# -------------------------
# Summary Generation
# -------------------------

def generate_summary(llm: ChatOpenAI, docs: List[Document]) -> str:
# Line 101: Defines a function to create a summary of the document.
    full_text = "\n\n".join(doc.page_content for doc in docs)
# Line 102: Combines the text content of all document chunks into a single large string.

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert analyst. Generate a concise, high-level summary "
            "of the following document. Do not add external information."
        ),
        ("human", "{text}")
    ])
# Lines 104-112: Creates a prompt template. The "system" message sets the AI's persona, and the "human" message is a placeholder for the document text.

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
# Lines 114-118: Creates a processing pipeline using LangChain Expression Language (LCEL). The input flows through the prompt, then to the language model, and finally through the output parser.

    return chain.invoke({"text": full_text})
# Line 120: Executes the chain by passing the document's full text into the 'text' placeholder and returns the final string output.

# -------------------------
# Vector Store Creation
# -------------------------

def create_vectorstore(
    docs: List[Document],
    embeddings: OpenAIEmbeddings
) -> Chroma:
# Lines 126-129: Defines a function to create the vector store.
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )
# Lines 130-133: A convenience method that handles the entire process:
# 1. It takes each document chunk.
# 2. Uses the provided 'embeddings' model to create a vector for each chunk.
# 3. Stores the document and its corresponding vector in an in-memory Chroma database.
# 4. Returns the Chroma database object.

# -------------------------
# RAG Prompt
# -------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a question-answering assistant. "
        "Answer strictly using the provided context. "
        "If the answer is not contained in the context, say: "
        "'The document does not contain this information.'"
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])
# Lines 139-148: Defines the main prompt for the Retrieval-Augmented Generation (RAG) task. It instructs the AI on how to behave and provides placeholders for the 'context' (retrieved documents) and the 'question'.

# -------------------------
# LangGraph Nodes
# -------------------------

def retrieve_node(state: ChatState, session: AISession) -> ChatState:
# Line 154: Defines the first node in the graph: retrieval. It takes the current state and the AI session as input.
    retriever = session.vectorstore.as_retriever(search_kwargs={"k": 4})
# Line 155: Creates a retriever from the vector store. A retriever is an object specialized in fetching documents. 'search_kwargs={"k": 4}' configures it to find the 4 most relevant documents.
    docs = retriever.invoke(state["question"])
# Line 156: Uses the retriever to perform a similarity search on the vector store using the user's question, returning the most relevant documents.
    state["context"] = docs
# Line 157: Adds the retrieved documents to the graph's state object under the key "context".
    return state
# Line 158: Returns the updated state.


def generate_node(state: ChatState, session: AISession) -> ChatState:
# Line 161: Defines the second node in the graph: generation.
    context_text = "\n\n".join(doc.page_content for doc in state["context"])
# Line 162: Concatenates the content of the retrieved documents into a single string.

    chain = (
        RAG_PROMPT
        | session._llm
        | StrOutputParser()
    )
# Lines 164-168: Creates an LCEL chain for generation, using the non-streaming LLM.

    answer = chain.invoke(
        {
            "context": context_text,
            "question": state["question"]
        }
    )
# Lines 170-175: Executes the chain, providing the retrieved context and original question to the prompt.

    state["answer"] = answer
# Line 177: Adds the generated answer to the state under the key "answer".
    return state
# Line 178: Returns the updated state.

# -------------------------
# Graph Builder
# -------------------------

def build_rag_graph(session: AISession):
# Line 184: Defines a function to construct the graph.
    graph = StateGraph(ChatState)
# Line 185: Initializes a new state graph, specifying 'ChatState' as the structure of its state.

    graph.add_node("retrieve", lambda s: retrieve_node(s, session))
# Line 187: Adds the 'retrieve_node' function to the graph with the name "retrieve". The lambda is used to pass the 'session' object to the node function.
    graph.add_node("generate", lambda s: generate_node(s, session))
# Line 188: Adds the 'generate_node' function with the name "generate".

    graph.set_entry_point("retrieve")
# Line 190: Specifies that the "retrieve" node is the starting point of the graph.
    graph.add_edge("retrieve", "generate")
# Line 191: Creates a directed edge, so that after "retrieve" finishes, "generate" will be called next.
    graph.add_edge("generate", END)
# Line 192: Creates an edge from "generate" to the special 'END' state, meaning the graph execution finishes after the generation step.

    return graph.compile()
# Line 194: Compiles the graph definition into a runnable object.

# -------------------------
# Streaming QA Generator
# -------------------------

def stream_rag_answer(
    session: AISession,
    question: str
) -> Generator[str, None, None]:
# Lines 200-203: Defines the generator function that will be used by the API to stream answers. This function performs the RAG process manually to allow for streaming.
    """ Docstring explaining the function's purpose. """

    retriever = session.vectorstore.as_retriever(search_kwargs={"k": 4})
# Line 208: Creates a retriever to get relevant documents.
    docs = retriever.invoke(question)
# Line 209: Retrieves documents based on the user's question.

    context_text = "\n\n".join(doc.page_content for doc in docs)
# Line 211: Combines the document content into a single context string.

    messages = RAG_PROMPT.format_messages(
        context=context_text,
        question=question
    )
# Lines 216-219: Uses the prompt template to format the input into a list of 'Message' objects suitable for the chat model.

    session.chat_history.append(HumanMessage(content=question))
# Line 221: Adds the user's current question to the session's chat history.

    buffer = ""
# Line 223: Initializes an empty string to accumulate the full response for saving to history.
    for chunk in session._streaming_llm.stream(messages):
# Line 224: Calls the '.stream()' method on the streaming LLM instance, which returns a generator of message chunks.
        if chunk.content:
# Line 225: Checks if the chunk contains text content.
            buffer += chunk.content
# Line 226: Appends the chunk's content to the buffer.
            yield chunk.content
# Line 227: Immediately yields the chunk's content to the caller (the API).

    session.chat_history.append(AIMessage(content=buffer))
# Line 229: After the stream is complete, adds the entire response (from the buffer) to the chat history as a single AI message.

# -------------------------
# High-Level Session API
# -------------------------

def initialize_session(pdf_path: str) -> AISession:
# Line 235: Defines the main function called by the API to set up a new session.
    """ Docstring explaining the function's purpose. """

    session = AISession()
# Line 242: Creates a new, empty 'AISession' object.

    session.documents = load_and_split_pdf(pdf_path)
# Line 244: Calls the helper function to load and split the PDF, storing the resulting chunks in the session.
    session.summary = generate_summary(session._llm, session.documents)
# Line 245: Calls the helper function to generate and store the summary.
    session.vectorstore = create_vectorstore(
        session.documents,
        session._embeddings
    )
# Lines 246-249: Calls the helper function to create the vector store using the document chunks and the embedding model.

    session._rag_graph = build_rag_graph(session)
# Line 251: Builds the LangGraph and stores it in the session. (Note: This graph is built but not used by the current streaming chat implementation).

    return session
# Line 253: Returns the fully initialized session object to the API.
```