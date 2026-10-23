"""Part 2 - Data preparation: load files -> split into chunks."""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, SUPPORTED_EXTENSIONS


def load_file(path: str) -> List[Document]:
    """Load a single PDF / TXT / MD file into LangChain Documents."""
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    else:  # .txt and .md are both plain text
        loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = Path(path).name
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping, searchable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def prepare_chunks(paths: List[str]) -> List[Document]:
    """Full pipeline for many files: load + split."""
    all_docs: List[Document] = []
    for p in paths:
        all_docs.extend(load_file(p))
    return split_documents(all_docs)
