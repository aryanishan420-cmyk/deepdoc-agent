"""Part 7 - Streamlit frontend.   Run with:  streamlit run app.py"""
import os
import tempfile

import streamlit as st

from agent import ask, build_agent
from ingestion import prepare_chunks
from vector_store import build_vector_store

st.set_page_config(page_title="Agentic RAG", page_icon="📚")
st.title("📚 Agentic RAG Assistant")

# ---- session state ----
st.session_state.setdefault("agent", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_files", [])

# ---- sidebar: upload + index ----
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Process documents", disabled=not uploads):
        with st.spinner("Loading, chunking and embedding..."):
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for f in uploads:
                    p = os.path.join(tmp, f.name)
                    with open(p, "wb") as out:
                        out.write(f.getbuffer())
                    paths.append(p)
                chunks = prepare_chunks(paths)
            store = build_vector_store(chunks)
            st.session_state.agent = build_agent(store)
            st.session_state.indexed_files = [f.name for f in uploads]
            st.session_state.messages = []
        st.success(f"Indexed {len(chunks)} chunks from {len(uploads)} file(s).")

    if st.session_state.indexed_files:
        st.caption("Active: " + ", ".join(st.session_state.indexed_files))

# ---- chat ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about your documents..."):
    if st.session_state.agent is None:
        st.warning("Upload and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Agent is searching and analysing..."):
                answer = ask(st.session_state.agent, st.session_state.messages)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
