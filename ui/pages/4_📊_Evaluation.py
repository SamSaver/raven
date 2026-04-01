"""Evaluation dashboard — RAGAS metrics, BEIR benchmarks, synthetic tests.

All long-running evaluations use background tasks with polling so the
Streamlit frontend never times out.
"""

import time

import requests
import streamlit as st
import plotly.graph_objects as go

API = st.session_state.get("api_url", "http://localhost:8000")


def _poll_eval_task(task_id: str, status_placeholder, timeout: int = 600) -> dict | None:
    """Poll /api/eval/status/{task_id} until completed or failed."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{API}/api/eval/status/{task_id}", timeout=10)
            resp.raise_for_status()
            task = resp.json()
        except Exception as e:
            status_placeholder.error(f"Polling failed: {e}")
            return None

        status = task.get("status", "unknown")
        label = task.get("label", "Evaluation")

        if status == "running":
            elapsed = int(time.time() - start)
            status_placeholder.info(f"⏳ **{label}** running... ({elapsed}s)")
        elif status == "completed":
            status_placeholder.success(f"✅ **{label}** completed!")
            return task.get("result")
        elif status == "failed":
            status_placeholder.error(f"❌ **{label}** failed: {task.get('error', 'Unknown error')}")
            return None

        time.sleep(3)

    status_placeholder.error("⏰ Evaluation timed out.")
    return None


st.header("📊 Evaluation")

tab_ragas, tab_bench, tab_synth, tab_history = st.tabs(["RAGAS Metrics", "BEIR Benchmarks", "Synthetic Tests", "History"])

# ── RAGAS tab ───────────────────────────────────────────
with tab_ragas:
    st.subheader("RAGAS Evaluation")
    st.caption(
        "Evaluate RAG quality using Faithfulness, Answer Relevancy, "
        "and Context Precision on recent query logs."
    )

    ragas_col1, ragas_col2 = st.columns([1, 2])

    with ragas_col1:
        limit = st.number_input("Max queries to evaluate", 10, 500, 50, step=10)

        if st.button("▶️ Run RAGAS on Query Logs", type="primary"):
            status_ph = st.empty()
            try:
                resp = requests.post(
                    f"{API}/api/eval/ragas/from-logs",
                    params={"limit": limit},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                if "task_id" in data:
                    result = _poll_eval_task(data["task_id"], status_ph)
                    if result:
                        st.session_state.ragas_result = result
                elif data.get("detail"):
                    status_ph.warning(data["detail"])
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    status_ph.warning("No query logs available. Ask some questions in Chat first.")
                else:
                    status_ph.error(f"Error: {e}")
            except Exception as e:
                status_ph.error(f"Error: {e}")

    with ragas_col2:
        if "ragas_result" in st.session_state and st.session_state.ragas_result:
            result = st.session_state.ragas_result
            metrics = result.get("aggregate_metrics", {})

            if metrics:
                labels = [k.replace("_", " ").title() for k in metrics.keys()]
                values = [v * 100 for v in metrics.values()]
                values_closed = values + [values[0]]
                labels_closed = labels + [labels[0]]

                fig = go.Figure(data=go.Scatterpolar(
                    r=values_closed, theta=labels_closed,
                    fill="toself", line_color="#6366f1",
                    fillcolor="rgba(99, 102, 241, 0.15)",
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False, height=350,
                    margin=dict(l=40, r=40, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                mcols = st.columns(len(metrics))
                for col, (name, val) in zip(mcols, metrics.items()):
                    col.metric(name.replace("_", " ").title(), f"{val:.1%}")
            else:
                st.info("No metrics computed yet.")


# ── Benchmarks tab ──────────────────────────────────────
with tab_bench:
    st.subheader("BEIR Retrieval Benchmarks")
    st.caption(
        "Evaluate embedding retrieval quality against academic datasets: "
        "MS MARCO, HotpotQA, SciFact."
    )

    bench_col1, bench_col2 = st.columns([1, 3])

    with bench_col1:
        datasets = st.multiselect(
            "Datasets", ["msmarco", "hotpotqa", "scifact"],
            default=["scifact"],
        )
        max_corpus = st.number_input("Max corpus size", 500, 10000, 5000, step=500)
        max_queries = st.number_input("Max queries", 50, 500, 200, step=50)

        if st.button("▶️ Run Benchmarks", type="primary"):
            status_ph = st.empty()
            try:
                resp = requests.post(
                    f"{API}/api/eval/benchmark",
                    json={"datasets": datasets or None, "max_corpus": max_corpus, "max_queries": max_queries},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                if "task_id" in data:
                    result = _poll_eval_task(data["task_id"], status_ph, timeout=900)
                    if result:
                        st.session_state.bench_results = result.get("benchmarks", [])
            except Exception as e:
                status_ph.error(f"Benchmark failed: {e}")

    with bench_col2:
        if "bench_results" in st.session_state:
            for result in st.session_state.bench_results:
                if "error" in result:
                    st.error(f"**{result.get('dataset', '?')}**: {result['error']}")
                    continue

                with st.container():
                    st.markdown(f"### {result['dataset'].upper()}")
                    st.caption(
                        f"{result.get('description', '')} | "
                        f"{result.get('corpus_size', 0)} docs | "
                        f"{result.get('num_queries', 0)} queries | "
                        f"{result.get('duration_seconds', 0):.1f}s"
                    )

                    bm = result.get("metrics", {})
                    if bm:
                        fig = go.Figure(data=[go.Bar(
                            x=list(bm.keys()),
                            y=[v * 100 for v in bm.values()],
                            marker_color="#6366f1",
                            text=[f"{v:.1%}" for v in bm.values()],
                            textposition="auto",
                        )])
                        fig.update_layout(
                            yaxis_title="Score (%)", yaxis_range=[0, 100],
                            height=280, margin=dict(l=40, r=20, t=10, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    st.divider()


# ── Synthetic Tests tab ─────────────────────────────────
with tab_synth:
    st.subheader("Synthetic Test Generation")
    st.caption(
        "Generate QA test sets from your ingested documents using the LLM. "
        "Use these for regression testing and RAGAS evaluation."
    )

    synth_col1, synth_col2 = st.columns([1, 2])

    with synth_col1:
        max_chunks = st.number_input("Chunks to process", 5, 50, 20, step=5)
        qa_per_chunk = st.number_input("QA pairs per chunk", 1, 5, 3)
        multi_hop = st.toggle("Include multi-hop questions", value=True)

        if st.button("▶️ Generate Test Set", type="primary"):
            status_ph = st.empty()
            try:
                resp = requests.post(
                    f"{API}/api/eval/synthetic/generate",
                    json={"max_chunks": max_chunks, "qa_per_chunk": qa_per_chunk, "include_multi_hop": multi_hop},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                if "task_id" in data:
                    result = _poll_eval_task(data["task_id"], status_ph, timeout=600)
                    if result:
                        total = result.get("total_samples", 0)
                        tsid = result.get("test_set_id", "?")
                        st.success(f"Generated **{total}** QA pairs (ID: {tsid})")
            except Exception as e:
                status_ph.error(f"Generation failed: {e}")

    with synth_col2:
        try:
            resp = requests.get(f"{API}/api/eval/synthetic/list", timeout=10)
            resp.raise_for_status()
            test_sets = resp.json().get("test_sets", [])
        except Exception:
            test_sets = []

        if test_sets:
            for ts in test_sets:
                with st.container():
                    ts_col1, ts_col2, ts_col3, ts_col4 = st.columns([2, 1, 1, 1])
                    ts_col1.markdown(f"**{ts['test_set_id']}**")
                    ts_col2.caption(f"{ts['total_samples']} samples")
                    ts_col3.caption(ts.get("created_at", "")[:10])

                    if ts_col4.button("📊 Evaluate", key=f"eval_{ts['test_set_id']}"):
                        eval_status_ph = st.empty()
                        try:
                            resp = requests.post(
                                f"{API}/api/eval/synthetic/{ts['test_set_id']}/evaluate",
                                timeout=15,
                            )
                            resp.raise_for_status()
                            data = resp.json()

                            if "task_id" in data:
                                result = _poll_eval_task(data["task_id"], eval_status_ph, timeout=900)
                                if result:
                                    eval_metrics = result.get("evaluation", {}).get("aggregate_metrics", {})
                                    if eval_metrics:
                                        mcols = st.columns(len(eval_metrics))
                                        for col, (name, val) in zip(mcols, eval_metrics.items()):
                                            col.metric(name.replace("_", " ").title(), f"{val:.1%}")
                                    else:
                                        st.warning("No metrics returned.")
                        except Exception as e:
                            eval_status_ph.error(f"Evaluation failed: {e}")
        else:
            st.caption("No test sets generated yet.")

# ── History tab ─────────────────────────────────────────
with tab_history:
    st.subheader("Evaluation History")
    st.caption("All saved evaluation results — RAGAS, benchmarks, and synthetic test evaluations.")

    filter_type = st.selectbox(
        "Filter by type",
        ["all", "ragas", "ragas_logs", "benchmark", "synthetic_eval"],
        index=0,
    )

    try:
        params = {} if filter_type == "all" else {"eval_type": filter_type}
        resp = requests.get(f"{API}/api/eval/history", params=params, timeout=10)
        resp.raise_for_status()
        history = resp.json().get("results", [])
    except Exception:
        history = []

    if history:
        for entry in history:
            eval_id = entry.get("eval_id", "?")
            eval_type = entry.get("eval_type", "?")
            test_set_id = entry.get("test_set_id")
            total = entry.get("total_samples", 0)
            model = entry.get("model_used", "")
            duration = entry.get("duration_seconds")
            created = entry.get("created_at", "")[:19]
            metrics = entry.get("aggregate_metrics", {})

            type_labels = {
                "ragas": "RAGAS",
                "ragas_logs": "RAGAS (Query Logs)",
                "benchmark": "BEIR Benchmark",
                "synthetic_eval": "Synthetic QA Eval",
            }
            label = type_labels.get(eval_type, eval_type)

            with st.expander(
                f"**{label}** — {created}"
                + (f" | {total} samples" if total else "")
                + (f" | {model}" if model else "")
                + (f" | {duration:.1f}s" if duration else ""),
                expanded=False,
            ):
                if test_set_id:
                    st.caption(f"Test set: `{test_set_id}`")

                if metrics:
                    mcols = st.columns(min(len(metrics), 5))
                    for col, (name, val) in zip(mcols, metrics.items()):
                        display_val = f"{val:.1%}" if isinstance(val, float) and val <= 1 else f"{val:.4f}"
                        col.metric(name.replace("_", " ").title(), display_val)

                    # Bar chart for metrics
                    fig = go.Figure(data=[go.Bar(
                        x=[k.replace("_", " ").title() for k in metrics.keys()],
                        y=[v * 100 if isinstance(v, float) and v <= 1 else v for v in metrics.values()],
                        marker_color="#6366f1",
                        text=[f"{v:.1%}" if isinstance(v, float) and v <= 1 else f"{v:.2f}" for v in metrics.values()],
                        textposition="auto",
                    )])
                    fig.update_layout(
                        yaxis_title="Score (%)", height=250,
                        margin=dict(l=40, r=20, t=10, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("No metrics available.")

                st.caption(f"ID: `{eval_id}`")
    else:
        st.caption("No evaluation history yet. Run a RAGAS evaluation, benchmark, or synthetic test evaluation to see results here.")
