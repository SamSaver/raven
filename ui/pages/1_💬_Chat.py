"""Chat page — streaming RAG conversation with sources."""

import json

import requests
import streamlit as st

API = st.session_state.get("api_url", "http://localhost:8000")

# ── Session state init ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None

# ── Header ──────────────────────────────────────────────
header_cols = st.columns([4, 1, 1, 1])
header_cols[0].markdown("### 💬 Chat")
if st.session_state.last_model:
    header_cols[1].caption(f"🤖 {st.session_state.last_model}")
if st.session_state.last_confidence is not None:
    pct = int(st.session_state.last_confidence * 100)
    header_cols[2].caption(f"🎯 {pct}%")
if header_cols[3].button("🗑️ Clear"):
    st.session_state.messages = []
    st.session_state.sources = []
    st.session_state.last_confidence = None
    st.session_state.last_model = None
    st.rerun()

# ── Sources in sidebar (not a column) ──────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("📚 Sources")
    if st.session_state.sources:
        for i, src in enumerate(st.session_state.sources, 1):
            filename = src.get("filename", "unknown")
            page = src.get("page_number")
            preview = src.get("content_preview", "")[:150]
            score = src.get("relevance_score", 0)
            page_str = f" — p.{page}" if page else ""

            with st.expander(f"[{i}] {filename}{page_str} ({score:.0%})", expanded=False):
                st.caption(preview)
    else:
        st.caption("Sources appear here after you ask a question.")

# ── Chat history ────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input (must be at top level for Streamlit to pin it to bottom) ──
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build history
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    config = st.session_state.get("retrieval_config", {
        "top_k": 10,
        "similarity_threshold": 0.5,
        "hybrid_weight": 0.7,
        "reranker_enabled": True,
        "max_context_tokens": 4096,
    })
    model = st.session_state.get("selected_model", None)

    # Stream response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            resp = requests.post(
                f"{API}/api/chat",
                json={
                    "query": prompt,
                    "retrieval_config": config,
                    "history": history,
                    "model": model,
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "sources":
                    st.session_state.sources = event.get("data", [])

                elif event.get("type") == "token":
                    full_response += event.get("data", "")
                    placeholder.markdown(full_response + "▌")

                elif event.get("type") == "done":
                    data = event.get("data", {})
                    st.session_state.last_confidence = data.get("confidence")
                    st.session_state.last_model = data.get("model_used")
                    if data.get("citations"):
                        st.session_state.sources = data["citations"]

            placeholder.markdown(full_response)

        except requests.exceptions.ConnectionError:
            full_response = "⚠️ Cannot connect to the backend. Make sure the API server is running."
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()  # Rerun to update sidebar sources and move input bar below new messages
