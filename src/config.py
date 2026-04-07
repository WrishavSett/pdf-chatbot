import os
from dotenv import load_dotenv

# Absolute path to .config, always resolved relative to this file's location
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
load_dotenv(os.path.join(_root, ".env"))
load_dotenv(os.path.join(_root, ".config"))

# LLM configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# RAG configurations
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY"))
TOP_K = int(os.getenv("TOP_K"))

# API configurations
SESSION_TTL = int(os.getenv("SESSION_TTL")) * 60  # stored as minutes, used as seconds
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE")) * 1024 * 1024  # stored as MB, used as bytes
FLASK_HOST = os.getenv("FLASK_HOST")
FLASK_PORT = int(os.getenv("FLASK_PORT"))
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL")) * 60 # stored as minutes, used as seconds
API_BASE_URL = os.getenv("API_BASE_URL")

# Logging configurations
LOG_FILE = os.getenv("LOG_FILE")
LOG_LEVEL = os.getenv("LOG_LEVEL")
MAX_BYTES = int(os.getenv("MAX_BYTES")) * 1024 * 1024  # stored as MB, used as bytes
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT"))