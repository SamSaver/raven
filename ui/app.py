"""Raven RAG — Streamlit UI entry point.

Run with: streamlit run ui/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Raven RAG",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared styling ──────────────────────────────────────
st.markdown("""
<style>
    /* Tighter padding */
    .block-container { padding-top: 2rem; }
    /* Chat message styling */
    .stChatMessage { max-width: 85%; }
    /* Source cards */
    .source-card {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background: #1a1a2e;
    }
    .source-card .filename { font-weight: 600; font-size: 0.85rem; }
    .source-card .preview { font-size: 0.78rem; color: #aaa; margin-top: 4px; }
    .source-card .score { font-size: 0.72rem; color: #888; }
    /* Metric cards */
    .metric-card {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        background: #1a1a2e;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; }
    .metric-card .label { font-size: 0.78rem; color: #888; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — status + navigation ───────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("🐦 Raven RAG")
        st.caption("Zero-budget retrieval system")
        st.divider()

        # Health check
        try:
            import requests
            api = st.session_state.get("api_url", "http://localhost:8000")
            resp = requests.get(f"{api}/health", timeout=3).json()
            col1, col2 = st.columns(2)
            col1.markdown(
                f"{'🟢' if resp.get('ollama_connected') else '🔴'} Ollama"
            )
            col2.markdown(
                f"{'🟢' if resp.get('qdrant_connected') else '🔴'} ChromaDB"
            )
            models = resp.get("available_models", [])
            if models:
                st.caption(f"{len(models)} model{'s' if len(models) != 1 else ''} available")
        except Exception:
            st.markdown("🔴 Backend unreachable")
            st.caption("Start the API server first")

        st.divider()

        # API URL config
        api_url = st.text_input(
            "Backend URL",
            value=st.session_state.get("api_url", "http://localhost:8000"),
            help="FastAPI backend address",
        )
        st.session_state["api_url"] = api_url


render_sidebar()

# ── Page routing ────────────────────────────────────────
chat_page = st.Page("pages/1_💬_Chat.py", title="Chat", icon="💬", default=True)
docs_page = st.Page("pages/2_📄_Documents.py", title="Documents", icon="📄")
settings_page = st.Page("pages/3_⚙️_Settings.py", title="Settings", icon="⚙️")
eval_page = st.Page("pages/4_📊_Evaluation.py", title="Evaluation", icon="📊")

pg = st.navigation([chat_page, docs_page, settings_page, eval_page])
pg.run()
