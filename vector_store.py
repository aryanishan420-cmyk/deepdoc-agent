"""Part 3 - Embeddings + in-memory vector store."""
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL


def build_vector_store(chunks: List[Document]) -> InMemoryVectorStore:
    """Embed all chunks with OpenAI embeddings and index them in memory."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(chunks)
    return store
