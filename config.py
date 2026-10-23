"""Part 1 - Configuration: models, chunking and retrieval settings."""
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

CHAT_MODEL = "openai:gpt-4o-mini"            # used by the Deep Agent + sub-agents
EMBEDDING_MODEL = "text-embedding-3-small"   # used for the vector store

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5                                    # chunks returned per search

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
