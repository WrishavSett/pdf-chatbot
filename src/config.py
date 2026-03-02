import os
from dotenv import load_dotenv

load_dotenv(".env")
load_dotenv(".config")

# LLM configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# RAG configurations
CHUNK_SIZE = os.getenv("CHUNK_SIZE")
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP")
MAX_TOKENS = os.getenv("MAX_TOKENS")
MAX_CONCURRENCY = os.getenv("MAX_CONCURRENCY")
TOP_K = os.getenv("TOP_K")

# API configurations
SESSION_TTL = os.getenv("SESSION_TTL")
MAX_FILE_SIZE = os.getenv("MAX_FILE_SIZE")
API_BASE_URL = os.getenv("API_BASE_URL")

# Logging configurations
LOG_FILE = os.getenv("LOG_FILE")
LOG_LEVEL = os.getenv("LOG_LEVEL")
MAX_LOG_SIZE = os.getenv("MAX_LOG_SIZE")
LOG_BACKUP_COUNT = os.getenv("LOG_BACKUP_COUNT")