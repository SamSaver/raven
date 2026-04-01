"""Documents page — upload, list, and manage documents."""

import time

import requests
import streamlit as st

API = st.session_state.get("api_url", "http://localhost:8000")

st.header("📄 Documents")

# ── Upload section ──────────────────────────────────────
st.subheader("Upload Documents")

upload_col1, upload_col2 = st.columns([2, 1])

with upload_col2:
    strategy = st.selectbox(
        "Chunking strategy",
        ["recursive", "fixed", "semantic", "hierarchical"],
        index=0,
        help="How to split the document into chunks",
    )
    chunk_size = st.number_input("Chunk size (chars)", 200, 5000, 1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", 0, 1000, 200, step=50)

with upload_col1:
    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Supports PDF, TXT, Markdown, CSV",
    )

    if uploaded_files:
        for f in uploaded_files:
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                # Submit as background task — returns immediately
                status_placeholder.info(f"📤 Uploading **{f.name}**...")
                resp = requests.post(
                    f"{API}/api/ingest",
                    files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")},
                    params={
                        "chunking_strategy": strategy,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "background": "true",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                task_info = resp.json()
                task_id = task_info.get("task_id")

                if not task_id:
                    st.error("Backend did not return a task ID")
                    continue

                # Poll for completion
                status_map = {
                    "queued": ("⏳", "Queued..."),
                    "parsing": ("📖", "Parsing document..."),
                    "chunking": ("✂️", "Splitting into chunks..."),
                    "embedding": ("🧠", "Generating embeddings..."),
                    "storing": ("💾", "Storing in vector database..."),
                    "completed": ("✅", "Done!"),
                    "failed": ("❌", "Failed"),
                }

                while True:
                    poll = requests.get(f"{API}/api/ingest/status/{task_id}", timeout=10).json()
                    current_status = poll.get("status", "unknown")

                    icon, label = status_map.get(current_status, ("🔄", current_status))
                    chunk_count = poll.get("chunk_count", 0)
                    extra = f" ({chunk_count} chunks)" if chunk_count else ""
                    status_placeholder.info(f"{icon} **{f.name}**: {label}{extra}")

                    if current_status == "completed":
                        result = poll.get("result", {})
                        status_placeholder.success(
                            f"✅ **{result.get('filename', f.name)}** — "
                            f"{result.get('chunk_count', '?')} chunks created"
                        )
                        break
                    elif current_status == "failed":
                        status_placeholder.error(f"❌ **{f.name}**: {poll.get('error', 'Unknown error')}")
                        break

                    time.sleep(2)

            except requests.exceptions.ConnectionError:
                status_placeholder.error("Cannot connect to backend. Is the API server running?")
            except Exception as e:
                status_placeholder.error(f"Upload failed: {e}")

st.divider()

# ── Document list ───────────────────────────────────────
st.subheader("Uploaded Documents")

try:
    resp = requests.get(f"{API}/api/documents", timeout=10)
    resp.raise_for_status()
    documents = resp.json()
except Exception:
    documents = []
    st.info("No documents found, or backend is unreachable.")

if documents:
    for doc in documents:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        col1.markdown(f"**{doc['filename']}**")
        col2.caption(doc.get("file_type", "").split("/")[-1].upper())
        col3.caption(f"{doc['chunk_count']} chunks")
        col4.caption(doc.get("upload_time", "")[:10])

        if col5.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete document"):
            try:
                requests.delete(f"{API}/api/documents/{doc['doc_id']}", timeout=10)
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
else:
    st.caption("No documents uploaded yet. Use the uploader above to add files.")

# ── Collection info ─────────────────────────────────────
st.divider()
with st.expander("🔍 Vector Store Info"):
    try:
        resp = requests.get(f"{API}/api/collections/info", timeout=10)
        resp.raise_for_status()
        info = resp.json()
        c1, c2 = st.columns(2)
        c1.metric("Collection", info.get("name", "—"))
        c2.metric("Vectors", info.get("points_count", 0))
        st.caption(f"Status: {info.get('status', 'unknown')}")
    except Exception:
        st.caption("Could not retrieve collection info.")
