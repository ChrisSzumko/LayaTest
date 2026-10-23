"""Run with `python main.py` or `streamlit run main.py`."""
import subprocess
import sys
from pathlib import Path


def app():
    """Render the Streamlit playground, evaluation controls, and policy reference.

    Cache a shared local backend across reruns and keep request/evaluation results
    in session state. Translate requests and run evaluations only when their
    buttons are pressed. Show model failures and provide report downloads without
    executing any real application actions.
    """
    import streamlit as st
    from domain import ACTIONS, GROUPS, USERS
    from evaluation import evaluate, report_html, report_json
    from inference import LayaBackend
    from pipeline import Pipeline
    from scenarios import scenarios

    st.set_page_config(page_title="Laya action playground", layout="wide")
    st.title("Laya action playground")
    st.caption("Local model · stub users and groups · dry run only — no real actions")

    @st.cache_resource
    def backend():
        """Create the backend resource cached by Streamlit across reruns.

        Returns:
            An unloaded LayaBackend whose checkpoint is loaded on demand.
        """
        return LayaBackend()

    model = backend()
    with st.sidebar:
        st.header("Test identity")
        actor = st.selectbox("Acting user", list(USERS), format_func=lambda key: f"{USERS[key].username} · {USERS[key].role}")
        st.write("Group memberships:", ", ".join(USERS[actor].groups))
        target_user = st.selectbox("Fallback target user", [None, *USERS], format_func=lambda key: "Self (default)" if key is None else USERS[key].username)
        target_group = st.selectbox("Fallback target group", [None, *GROUPS], format_func=lambda key: "Own group (if unique)" if key is None else GROUPS[key].name)
        st.caption("Explicit text targets override fallbacks. Personal edits use target user; group actions use target group.")
        threshold = st.slider("Experimental confidence threshold", 0.0, 1.0, 0.8, 0.01)
        st.caption("Model confidence is not a guarantee of correctness.")
        if st.button("Load / verify local model"):
            try:
                with st.spinner("Loading English checkpoint (first use downloads weights)…"):
                    model.load()
                st.success("Model loaded on CPU.")
            except Exception as exc:
                st.error(f"Model loading failed: {type(exc).__name__}: {exc}")
        if model.load_ms is not None:
            st.caption(f"Model load: {model.load_ms / 1000:.2f} seconds")
            for warning in model.load_warnings:
                st.warning(warning)

    playground, evaluation, reference = st.tabs(["Try a request", "Automatic evaluation", "Actions & policy"])
    with playground:
        examples = ["Custom input", "Show the system logs", "Change my password",
                    "Create a software alarm rule in Beta", "Change Bob's password",
                    "Do not change my password", "Show logs and add an email address"]
        example = st.selectbox("Example request", examples)
        if "previous_example" not in st.session_state or example != st.session_state.previous_example:
            st.session_state.request = "" if example == "Custom input" else example
            st.session_state.previous_example = example
        text = st.text_area("Request", key="request", max_chars=600, height=130)
        if st.button("Translate and check permissions", type="primary"):
            with st.spinner("Running Laya locally…"):
                st.session_state.result = Pipeline(model).translate(text, actor, target_user, target_group, threshold)
        if "result" in st.session_state:
            result = st.session_state.result
            st.subheader(f"Result: {result['status']}")
            st.caption(f"Evaluated as {result['actor']}: {result['input']}")
            st.write(result["reason"])
            left, right = st.columns(2)
            left.metric("Predicted action", result["action"] or "None")
            right.metric("Inference time", f"{result['inference_ms']:.1f} ms")
            st.json({"target": result["target"], "scores": result["scores"], "threshold": result["threshold"]})
            with st.expander("Raw model output and full result"):
                st.json(result)
                st.json(model.metadata())

    with evaluation:
        st.write(f"{len(scenarios())} labelled scenarios: three phrasings per action plus permission and edge cases.")
        st.caption("Uses fixed fixture identities, not the sidebar actor/targets. Uses the sidebar threshold. Models can fail these tests; failures remain visible.")
        if st.button("Run base evaluation"):
            bar = st.progress(0.0, text="Loading and warming model…")
            with st.spinner("Evaluating the real local model…"):
                st.session_state.report = evaluate(model, threshold, progress=lambda done, total: bar.progress(done/total, text=f"Case {done}/{total}"))
        if "report" in st.session_state:
            report = st.session_state.report
            if report["load_error"]:
                st.error(report["load_error"])
            st.json(report["summary"])
            st.bar_chart(report["per_action_accuracy"])
            st.dataframe([{"case": row["name"], "input": row["input"], "expected_action": row["expected"]["action"],
                           "actual_action": row["actual"]["action"], "expected_status": row["expected"]["status"],
                           "actual_status": row["actual"]["status"], "passed": row["passed"]} for row in report["results"]], width="stretch")
            st.download_button("Download JSON report", report_json(report), "evaluation.json", "application/json")
            st.download_button("Download HTML report", report_html(report), "evaluation.html", "text/html")
            with st.expander("Confusion matrix and run metadata"):
                st.json({"matrix": report["confusion_matrix"], "metadata": report["metadata"]})
    with reference:
        st.markdown("""| Capability | Regular | Manager | Superuser |
|---|---|---|---|
| Manuals / video | Yes | Yes | Yes |
| Group reads | Own group | Own group | Any group |
| Personal edits | Self | Self | Any user |
| Email-group edits / additions | No | Own group | Any group |""")
        st.dataframe([{"action": key, "meaning": value.description, "scope": value.scope} for key, value in ACTIONS.items()], width="stretch")
        st.write("Unsupported: deleting, promoting users, moving users between groups, and executing multiple actions. No passwords or complete action payloads are extracted.")


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        from evaluation import main
        sys.argv.remove("--evaluate")
        raise SystemExit(main())
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is not None:
        app()
    else:
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
                                         "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false", *sys.argv[1:]]))
