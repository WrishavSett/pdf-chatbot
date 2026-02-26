# core_ai.py

# Imports
import os
import uuid
from typing import List, Dict, Generator, Optional, TypedDict

import tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END

# Load API key
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Shared LLM and Embeddings (module-level singletons)
_shared_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

_shared_embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY
)

# Session State definition
class ChatState(TypedDict):
    """
    LangGraph-compatible state.
    Keys:
        - question: str
        - context: List[Document]
        - answer: str
        - chat_history: List[HumanMessage | AIMessage]
    """
    question: str
    context: List[Document]
    answer: str
    chat_history: List[HumanMessage | AIMessage]

# AISession container
class AISession:
    def __init__(self):
        self.pages: List[Document] = []
        self.documents: List[Document] = []
        self.vectorstore: Optional[Chroma] = None
        self.summary: Optional[str] = None
        self.chat_history: List[HumanMessage | AIMessage] = []
        self.collection_name: str = f"rag_collection_{uuid.uuid4().hex[:8]}"

        self._llm = _shared_llm

        self._embeddings = _shared_embeddings

        self._rag_graph = None

# PDF loading
def load_pdf(pdf_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        raise

# PDF splitting
def split_pdf(pages: List[Document]) -> List[Document]:
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250
        )
        return splitter.split_documents(pages)
    except Exception as e:
        raise

# Map-Reduce prompts
MAP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert analyst. Write a concise summary of the following document section."
    ),
    ("human", "{text}")
])

COMBINE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert analyst. The following are summaries of sections of a document. "
        "Combine them into a single, concise, high-level summary of the full document. "
        "Do not add external information."
    ),
    ("human", "{text}")
])

# Summary generation
def generate_summary(llm: ChatOpenAI, pages: List[Document]) -> str:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        full_text = "\n\n".join(doc.page_content for doc in pages)
        token_count = len(encoding.encode(full_text))

        if token_count < 98000:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an expert analyst. Generate a concise, high-level summary "
                    "of the following document. Do not add external information."
                ),
                ("human", "{text}")
            ])
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"text": full_text})

        else:
            # Map: summarize each page in parallel
            map_chain = MAP_PROMPT | llm | StrOutputParser()
            summaries = map_chain.batch(
                [{"text": doc.page_content} for doc in pages],
                config={"max_concurrency": 5}
            )

            # Reduce: combine all summaries into a final summary
            combine_chain = COMBINE_PROMPT | llm | StrOutputParser()
            return combine_chain.invoke({"text": "\n\n".join(summaries)})

    except Exception as e:
        raise

# Vectorstore creation
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

# RAG prompt
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

# LangGraph Retrieval node
def retrieve_node(state: ChatState, session: AISession) -> ChatState:
    retriever = session.vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(state["question"])
    state["context"] = docs
    return state

# LangGraph Generate node
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

# Graph builder
def build_rag_graph(session: AISession):
    graph = StateGraph(ChatState)

    graph.add_node("retrieve", lambda s: retrieve_node(s, session))
    graph.add_node("generate", lambda s: generate_node(s, session))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

# RAG QA (Non-Streaming)
def get_rag_answer(
    session: AISession,
    question: str
) -> str:
    initial_state: ChatState = {
        "question": question,
        "context": [],
        "answer": "",
        "chat_history": session.chat_history
    }
    
    final_state = session._rag_graph.invoke(initial_state)
    
    answer = final_state["answer"]
    
    session.chat_history.append(HumanMessage(content=question))
    session.chat_history.append(AIMessage(content=answer))
    
    return answer

# # RAG QA (Streaming)
# def stream_rag_answer(
#     session: AISession,
#     question: str
# ) -> Generator[str, None, None]:
#     retriever = session.vectorstore.as_retriever(search_kwargs={"k": 4})
#     docs = retriever.invoke(question)

#     context_text = "\n\n".join(doc.page_content for doc in docs)

#     messages = RAG_PROMPT.format_messages(
#         context=context_text,
#         question=question
#     )

#     session.chat_history.append(HumanMessage(content=question))

#     buffer = ""
#     for chunk in session._llm.stream(messages):
#         if chunk.content:
#             buffer += chunk.content
#             yield chunk.content

#     session.chat_history.append(AIMessage(content=buffer))

# Session API provider
def initialize_session(pdf_path: str) -> AISession:
    try:
        session = AISession()

        session.pages = load_pdf(pdf_path)
        session.documents = split_pdf(session.pages)
        session.summary = generate_summary(session._llm, session.pages)
        session.vectorstore = create_vectorstore(
            session.documents,
            session._embeddings,
            session.collection_name
        )

        session._rag_graph = build_rag_graph(session)
        return session
    except Exception as e:
        raise