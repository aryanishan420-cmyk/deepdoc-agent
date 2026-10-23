"""Part 6 - Deep Agent (main agent + specialist sub-agent)."""
from deepagents import create_deep_agent
from langchain_core.vectorstores import InMemoryVectorStore

from config import CHAT_MODEL
from prompts import CHUNK_ANALYST_PROMPT, MAIN_AGENT_PROMPT
from tools import make_tools


def build_agent(vector_store: InMemoryVectorStore):
    search_documents, read_chunk = make_tools(vector_store)

    chunk_analyst = {
        "name": "chunk-analyst",
        "description": (
            "Reads ONE chunk from a search batch (needs batch_id and "
            "chunk_index) and extracts facts relevant to the question."
        ),
        "system_prompt": CHUNK_ANALYST_PROMPT,
        "tools": [read_chunk],
    }

    return create_deep_agent(
        model=CHAT_MODEL,
        tools=[search_documents],
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=[chunk_analyst],
    )


def ask(agent, history: list[dict]) -> str:
    """history = [{"role": "user"|"assistant", "content": "..."}, ...]"""
    result = agent.invoke({"messages": history})
    content = result["messages"][-1].content
    if isinstance(content, list):  # some models return content blocks
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content
