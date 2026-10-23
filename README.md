# Agentic RAG (LangChain + Deep Agents + Streamlit)

## Setup
    python -m venv .venv && source .venv/bin/activate   # or: uv venv
    pip install -r requirements.txt                      # or: uv pip install -r requirements.txt
    cp .env.example .env                                 # add your OPENAI_API_KEY
    streamlit run app.py

## Structure
| Part | File            | Purpose                                         |
|------|-----------------|-------------------------------------------------|
| 1    | config.py       | Models, chunk size, top-k                       |
| 2    | ingestion.py    | Load PDF/TXT/MD and split into chunks           |
| 3    | vector_store.py | OpenAI embeddings + in-memory vector store      |
| 4    | tools.py        | search_documents (batch IDs) + read_chunk       |
| 5    | prompts.py      | Main-agent and sub-agent instructions           |
| 6    | agent.py        | Deep Agent with 'chunk-analyst' sub-agent       |
| 7    | app.py          | Streamlit chat UI                               |
