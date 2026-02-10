# core_ai.py

# -------------------------
# Imports
# -------------------------

import os
import uuid
from typing import List, Dict, Generator, Optional
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------
# Load API Key
# -------------------------

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Session State Definition
# -------------------------

class ChatState(dict):
    """
    LangGraph-compatible state.
    Keys:
        - question: str
        - context: List[Document]
        - answer: str
        - chat_history: List[HumanMessage | AIMessage]
    """
    pass

# -------------------------
# Core AI Session Container
# -------------------------

class AISession:
    """
    Holds all AI-related objects for a single active chat session.
    Destroyed on reset.
    """

    def __init__(self):
        self.documents: List[Document] = []
        self.vectorstore: Optional[Chroma] = None
        self.summary: Optional[str] = None
        self.chat_history: List = []
        self.collection_name: str = f"rag_collection_{uuid.uuid4().hex[:8]}"

        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        self._embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY
        )

        self._rag_graph = None

# -------------------------
# PDF Processing
# -------------------------

def load_and_split_pdf(pdf_path: str) -> List[Document]:
    logging.info(f"Loading and splitting PDF: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250
        )

        logging.info("PDF successfully split into chunks.")
        return splitter.split_documents(docs)
    except Exception as e:
        logging.error(f"Error loading and splitting PDF: {e}")
        raise

# -------------------------
# Summary Generation
# -------------------------

def generate_summary(llm: ChatOpenAI, docs: List[Document]) -> str:
    logging.info("Generating summary for the document.")
    try:
        full_text = "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert analyst. Generate a concise, high-level summary "
                "of the following document. Do not add external information."
            ),
            ("human", "{text}")
        ])

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        summary = chain.invoke({"text": full_text})
        logging.info("Summary generation completed.")
        return summary
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        raise

# -------------------------
# Vector Store Creation
# -------------------------

def create_vectorstore(
    docs: List[Document],
    embeddings: OpenAIEmbeddings,
    collection_name: str
) -> Chroma:
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name
    )

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

# -------------------------
# LangGraph Nodes
# -------------------------

def retrieve_node(state: ChatState, session: AISession) -> ChatState:
    retriever = session.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(state["question"])
    state["context"] = docs
    return state


def generate_node(state: ChatState, session: AISession) -> ChatState:
    context_text = "\n\n".join(doc.page_content for doc in state["context"])

    chain = (
        RAG_PROMPT
        | session._llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context_text,
            "question": state["question"]
        }
    )

    state["answer"] = answer
    return state

# -------------------------
# Graph Builder
# -------------------------

def build_rag_graph(session: AISession):
    graph = StateGraph(ChatState)

    graph.add_node("retrieve", lambda s: retrieve_node(s, session))
    graph.add_node("generate", lambda s: generate_node(s, session))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

# -------------------------
# Streaming QA Generator
# -------------------------

def stream_rag_answer(
    session: AISession,
    question: str
) -> Generator[str, None, None]:
    """
    Streams ONLY the QA response tokens.
    """

    retriever = session.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in docs)

    messages = RAG_PROMPT.format_messages(
        context=context_text,
        question=question
    )

    session.chat_history.append(HumanMessage(content=question))

    buffer = ""
    for chunk in session._llm.stream(messages):
        if chunk.content:
            buffer += chunk.content
            yield chunk.content

    session.chat_history.append(AIMessage(content=buffer))

# -------------------------
# High-Level Session API
# -------------------------

def initialize_session(pdf_path: str) -> AISession:
    logging.info("Initializing AI session.")
    try:
        session = AISession()

        session.documents = load_and_split_pdf(pdf_path)
        session.summary = generate_summary(session._llm, session.documents)
        session.vectorstore = create_vectorstore(
            session.documents,
            session._embeddings,
            session.collection_name
        )

        session._rag_graph = build_rag_graph(session)
        logging.info("AI session initialized successfully.")
        return session
    except Exception as e:
        logging.error(f"Error initializing session: {e}")
        raise