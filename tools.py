"""Part 4 - Agent tools.

search_documents  -> semantic search; saves results under a unique batch ID
read_chunk        -> returns the full text of one chunk from a batch
                     (used by the sub-agents so each analyses ONE chunk)
"""
import uuid
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from config import TOP_K


def make_tools(vector_store: InMemoryVectorStore):
    # batch_id -> list of retrieved chunks
    batches: Dict[str, List[Document]] = {}

    @tool
    def search_documents(query: str) -> str:
        """Semantic search over the uploaded documents.

        Returns a batch ID plus a short preview of each matching chunk.
        Use read_chunk(batch_id, chunk_index) to read a chunk in full.
        """
        results = vector_store.similarity_search(query, k=TOP_K)
        if not results:
            return "No relevant chunks found."

        batch_id = uuid.uuid4().hex[:8]
        batches[batch_id] = results

        lines = [f"batch_id: {batch_id}", f"chunks: {len(results)}", ""]
        for i, doc in enumerate(results):
            preview = doc.page_content[:150].replace("\n", " ")
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            where = f"{src}" + (f", page {page + 1}" if page is not None else "")
            lines.append(f"[{i}] ({where}) {preview}...")
        return "\n".join(lines)

    @tool
    def read_chunk(batch_id: str, chunk_index: int) -> str:
        """Return the full text of one chunk from a search batch."""
        batch = batches.get(batch_id)
        if batch is None:
            return f"Unknown batch_id '{batch_id}'."
        if not 0 <= chunk_index < len(batch):
            return f"chunk_index must be between 0 and {len(batch) - 1}."
        doc = batch[chunk_index]
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        where = f"{src}" + (f", page {page + 1}" if page is not None else "")
        return f"SOURCE: {where}\n\n{doc.page_content}"

    return search_documents, read_chunk
