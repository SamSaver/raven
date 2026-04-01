"""Settings page — configure retrieval pipeline and model selection."""

import requests
import streamlit as st

API = st.session_state.get("api_url", "http://localhost:8000")

st.header("⚙️ Settings")

# ── Init defaults in session ────────────────────────────
if "retrieval_config" not in st.session_state:
    st.session_state.retrieval_config = {
        "top_k": 10,
        "similarity_threshold": 0.5,
        "hybrid_weight": 0.7,
        "reranker_enabled": True,
        "max_context_tokens": 4096,
    }

config = st.session_state.retrieval_config

# ── Model selection ─────────────────────────────────────
st.subheader("🤖 LLM Model")

models = []
default_model = ""
try:
    resp = requests.get(f"{API}/api/models", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("models", [])
    default_model = data.get("default", "")
except Exception:
    st.warning("Cannot fetch models from backend.")

if models:
    current = st.session_state.get("selected_model", default_model)
    idx = models.index(current) if current in models else 0
    selected = st.selectbox(
        "Active model",
        models,
        index=idx,
        help="Select which Ollama model to use for generation",
    )
    st.session_state.selected_model = selected
    st.caption(f"{len(models)} model{'s' if len(models) != 1 else ''} available in Ollama")
else:
    st.caption("No models found. Is Ollama running?")

st.divider()

# ── Retrieval configuration ─────────────────────────────
st.subheader("🔍 Retrieval Configuration")

col1, col2 = st.columns(2)

with col1:
    config["top_k"] = st.slider(
        "Top K",
        min_value=1, max_value=50, value=config["top_k"],
        help="Number of chunks to retrieve",
    )

    config["similarity_threshold"] = st.slider(
        "Similarity threshold",
        min_value=0.0, max_value=1.0, value=config["similarity_threshold"], step=0.05,
        help="Minimum cosine similarity score to include a chunk",
    )

    config["hybrid_weight"] = st.slider(
        "Hybrid weight",
        min_value=0.0, max_value=1.0, value=config["hybrid_weight"], step=0.05,
        help="0.0 = pure BM25 (keyword), 1.0 = pure semantic (embedding)",
    )

with col2:
    config["max_context_tokens"] = st.slider(
        "Max context tokens",
        min_value=512, max_value=16384, value=config["max_context_tokens"], step=512,
        help="Maximum tokens to include in the LLM context window",
    )

    config["reranker_enabled"] = st.toggle(
        "Cross-encoder reranking",
        value=config["reranker_enabled"],
        help="Rerank results with a cross-encoder for better precision (adds latency)",
    )

st.session_state.retrieval_config = config

st.divider()

# ── System actions ──────────────────────────────────────
st.subheader("🛠️ System")

sys_col1, sys_col2, sys_col3 = st.columns(3)

with sys_col1:
    if st.button("🗑️ Clear Response Cache"):
        try:
            requests.post(f"{API}/api/admin/cache/clear", timeout=10)
            st.success("Cache cleared")
        except Exception:
            st.error("Failed to clear cache")

with sys_col2:
    if st.button("🔄 Reset to Defaults"):
        st.session_state.retrieval_config = {
            "top_k": 10,
            "similarity_threshold": 0.5,
            "hybrid_weight": 0.7,
            "reranker_enabled": True,
            "max_context_tokens": 4096,
        }
        st.rerun()

with sys_col3:
    if st.button("🕸️ Build Knowledge Graph"):
        with st.spinner("Building graph (this may take a while)..."):
            try:
                resp = requests.post(
                    f"{API}/api/graph/build",
                    json={"rebuild": False},
                    timeout=600,
                )
                resp.raise_for_status()
                result = resp.json()
                st.success(
                    f"Graph built: {result.get('entities_extracted', 0)} entities, "
                    f"{result.get('relationships_extracted', 0)} relationships"
                )
            except Exception as e:
                st.error(f"Graph build failed: {e}")

# ── Graph stats ─────────────────────────────────────────
with st.expander("🕸️ Knowledge Graph Stats"):
    try:
        resp = requests.get(f"{API}/api/graph/stats", timeout=5)
        resp.raise_for_status()
        stats = resp.json()
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Nodes", stats.get("nodes", 0))
        g2.metric("Edges", stats.get("edges", 0))
        g3.metric("Communities", stats.get("communities", 0))
        g4.metric("Density", f"{stats.get('density', 0):.4f}")
    except Exception:
        st.caption("Graph not available.")
