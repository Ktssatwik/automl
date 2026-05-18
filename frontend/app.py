import json
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Agentic AutoML POC", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
    .title-card {
      border: 1px solid #d9e3ee;
      border-radius: 12px;
      padding: 14px 16px;
      background: #f8fbff;
      margin-bottom: 12px;
    }
    .section-head {
      font-size: 1.05rem;
      font-weight: 650;
      margin-top: 8px;
      margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-card">
      <h2 style="margin:0;">Agentic AutoML POC</h2>
      <p style="margin:6px 0 0 0; color:#52606d;">Upload CSV, run pipeline, inspect every step result, predict, and download model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "job_ids" not in st.session_state:
    st.session_state.job_ids = []
if "selected_job_id" not in st.session_state:
    st.session_state.selected_job_id = ""
if "last_uploaded_df" not in st.session_state:
    st.session_state.last_uploaded_df = None
if "manual_job_input" not in st.session_state:
    st.session_state.manual_job_input = ""


def api_get(url: str, timeout: int = 60) -> Dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def render_step_output(step_name: str, data: Dict[str, Any], show_raw: bool = False) -> None:
    if not isinstance(data, dict):
        st.write(data)
        return

    if step_name == "csv_understanding":
        a, b, c = st.columns(3)
        shape = data.get("dataset_shape", {})
        a.metric("Rows", shape.get("rows", "-"))
        b.metric("Columns", shape.get("columns", "-"))
        c.metric("Selected Target", data.get("selected_target", "-"))
        st.write(f"**Reasoning:** {data.get('reasoning', '-')}")
        cols_profile = data.get("columns_profile", [])
        if cols_profile:
            st.dataframe(pd.DataFrame(cols_profile), width="stretch")

    elif step_name == "problem_type_detection":
        a, b = st.columns(2)
        a.metric("Problem Type", data.get("problem_type", "-"))
        a.metric("Unique Count", data.get("target_unique_count", "-"))
        b.metric("Unique Ratio", data.get("target_unique_ratio", "-"))
        b.metric("Cleaning Applied", str(data.get("label_cleaning_applied", False)))
        st.write(f"**Reasoning:** {data.get('reasoning', '-')}")
        if data.get("label_mapping"):
            st.write("**Label Mapping:**")
            st.dataframe(
                pd.DataFrame(
                    [{"from": k, "to": v} for k, v in data.get("label_mapping", {}).items()]
                ),
                width="stretch",
            )

    elif step_name in {"domain_understanding", "preprocessing", "train_test_split", "model_training", "cross_validation", "metrics_evaluation", "model_selection"}:
        top_fields = {k: v for k, v in data.items() if k not in {"llm_decision", "raw_decision", "metrics", "cv_results"}}
        st.write(top_fields)
        if "metrics" in data and isinstance(data["metrics"], dict):
            st.write("**Metrics Table**")
            metrics_rows = []
            for model_name, vals in data["metrics"].items():
                row = {"model": model_name}
                row.update(vals if isinstance(vals, dict) else {"value": vals})
                metrics_rows.append(row)
            st.dataframe(pd.DataFrame(metrics_rows), width="stretch")
        if "cv_results" in data and isinstance(data["cv_results"], dict):
            st.write("**CV Results**")
            cv_rows = []
            for model_name, vals in data["cv_results"].items():
                row = {"model": model_name}
                row.update(vals if isinstance(vals, dict) else {"value": vals})
                cv_rows.append(row)
            st.dataframe(pd.DataFrame(cv_rows), width="stretch")

    elif step_name == "eda":
        ds = data.get("dataset_summary", {})
        a, b, c = st.columns(3)
        a.metric("Rows", ds.get("rows", "-"))
        b.metric("Columns", ds.get("columns", "-"))
        c.metric("Numeric Features", len(ds.get("numeric_columns", [])))
        st.write(f"**LLM EDA Summary:** {data.get('llm_eda_summary', '-')}")
        miss = data.get("missing_summary", {})
        if miss:
            miss_df = pd.DataFrame(
                [{"column": col, **vals} for col, vals in miss.items()]
            ).sort_values("null_pct", ascending=False)
            st.write("**Missing Values (Top):**")
            st.dataframe(miss_df.head(20), width="stretch")
        corr = data.get("correlation_summary", {}).get("top_correlations", [])
        if corr:
            st.write("**Top Correlations:**")
            st.dataframe(pd.DataFrame(corr), width="stretch")

    elif step_name == "report_generation":
        st.write(f"**Summary:** {data.get('summary', '-')}")
        report = data.get("report", {})
        if isinstance(report, dict):
            st.write("**Final Report Snapshot:**")
            st.write({
                "selected_target": report.get("selected_target"),
                "problem_type": report.get("problem_type"),
                "domain": report.get("domain"),
                "best_model": report.get("best_model"),
                "best_metric": report.get("best_metric"),
            })

    elif step_name == "step_explanation":
        st.write("**Overview:**", data.get("overview", "-"))
        explanations = data.get("step_explanations", [])
        if explanations:
            exp_df = pd.DataFrame(
                [{"step": x.get("step"), "what_we_did": x.get("what_we_did")} for x in explanations]
            )
            st.dataframe(exp_df, width="stretch")
        if data.get("detailed_eda_output"):
            with st.expander("Detailed EDA Output"):
                st.json(data.get("detailed_eda_output"))

    else:
        st.write(data)

    if show_raw:
        st.divider()
        st.caption("Raw JSON")
        st.json(data)


backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000").rstrip("/")

side_a, side_b = st.sidebar.columns(2)
if side_a.button("Health Check"):
    try:
        health = api_get(f"{backend_url}/health", timeout=10)
        st.sidebar.success(f"Backend OK: {health}")
    except Exception as exc:
        st.sidebar.error(f"Health failed: {exc}")
if side_b.button("Refresh"):
    st.rerun()

st.markdown("<div class='section-head'>1) Upload CSV</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        preview_df = pd.read_csv(uploaded_file)
        st.session_state.last_uploaded_df = preview_df
        st.dataframe(preview_df.head(20), width="stretch")
    except Exception as exc:
        st.error(f"Failed to preview CSV: {exc}")

if st.button("Upload CSV to Backend", type="primary"):
    if uploaded_file is None:
        st.warning("Please choose a CSV file first.")
    else:
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{backend_url}/upload-csv", files=files, timeout=60)
            response.raise_for_status()
            body = response.json()
            job_id = body["job_id"]
            if job_id not in st.session_state.job_ids:
                st.session_state.job_ids.insert(0, job_id)
            st.session_state.selected_job_id = job_id
            st.session_state.manual_job_input = job_id
            st.success(f"Uploaded successfully. New job_id: {job_id}")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

st.markdown("<div class='section-head'>2) Select Job</div>", unsafe_allow_html=True)
manual_job_id = st.text_input("Or paste job_id", key="manual_job_input")
if manual_job_id:
    st.session_state.selected_job_id = manual_job_id.strip()

if st.session_state.job_ids:
    default_index = 0
    if st.session_state.selected_job_id in st.session_state.job_ids:
        default_index = st.session_state.job_ids.index(st.session_state.selected_job_id)
    selected_from_list = st.selectbox("Recent uploaded jobs", options=st.session_state.job_ids, index=default_index)
    if selected_from_list != st.session_state.selected_job_id:
        st.session_state.selected_job_id = selected_from_list
        st.session_state.manual_job_input = selected_from_list

job_id = st.session_state.selected_job_id
if job_id:
    st.info(f"Current job_id: {job_id}")

st.markdown("<div class='section-head'>3) Run AutoML</div>", unsafe_allow_html=True)
if st.button("Run AutoML"):
    if not job_id:
        st.warning("Please upload CSV or provide a job_id.")
    else:
        try:
            result = api_post(f"{backend_url}/run-automl", {"job_id": job_id}, timeout=30)
            st.success(result.get("message", "AutoML triggered."))
        except Exception as exc:
            st.error(f"Run failed: {exc}")

if job_id:
    st.markdown("<div class='section-head'>4) Pipeline Status and Step Results</div>", unsafe_allow_html=True)
    try:
        status = api_get(f"{backend_url}/pipeline-status/{job_id}")
        outputs_resp = api_get(f"{backend_url}/job-outputs/{job_id}")
        outputs = outputs_resp.get("data", {}) if outputs_resp.get("status") == "ready" else {}

        c1, c2, c3 = st.columns(3)
        steps = status.get("steps", [])
        completed = sum(1 for s in steps if s.get("status") == "Completed")
        c1.metric("Job Status", status.get("job_status", "unknown"))
        c2.metric("Completed Steps", f"{completed}/{len(steps)}")
        c3.metric("Last Updated", status.get("updated_at", "-"))
        show_raw_json = st.checkbox("Show raw JSON for each step", value=False)

        if status.get("error"):
            st.error(f"Pipeline Error: {status['error']}")

        for step in steps:
            step_name = step.get("name", "unknown")
            step_status = step.get("status", "Pending")
            step_message = step.get("message", "")
            icon = "✅" if step_status == "Completed" else "🟡" if step_status == "Running" else "❌" if step_status == "Failed" else "⚪"

            with st.expander(f"{icon} {step_name} [{step_status}]", expanded=(step_status == "Running")):
                if step_message:
                    st.write(f"Message: {step_message}")
                if step_name in outputs:
                    render_step_output(step_name, outputs[step_name], show_raw=show_raw_json)
                else:
                    st.caption("No result payload yet for this step.")

    except Exception as exc:
        st.error(f"Failed to load pipeline status/results: {exc}")

    st.markdown("<div class='section-head'>5) Reports and Prediction</div>", unsafe_allow_html=True)
    tab_eda, tab_model, tab_pred, tab_dl = st.tabs(["EDA Report", "Model Results", "Predict", "Download"])

    with tab_eda:
        try:
            eda = api_get(f"{backend_url}/eda-report/{job_id}")
            st.write(f"EDA Status: **{eda.get('status', 'unknown')}**")
            st.json(eda.get("data", {}))
        except Exception as exc:
            st.error(f"Failed to fetch EDA report: {exc}")

    with tab_model:
        try:
            model_results = api_get(f"{backend_url}/model-results/{job_id}")
            st.write(f"Model Results Status: **{model_results.get('status', 'unknown')}**")
            st.json(model_results.get("data", {}))
        except Exception as exc:
            st.error(f"Failed to fetch model results: {exc}")

    with tab_pred:
        default_record: Dict[str, Any] = {}
        try:
            feature_columns = outputs.get("preprocessing", {}).get("feature_columns", [])
            if isinstance(feature_columns, list) and feature_columns:
                default_record = {str(col): None for col in feature_columns}
            else:
                raw_cols = [c.get("column") for c in outputs.get("csv_understanding", {}).get("columns_profile", [])]
                target_col = outputs.get("csv_understanding", {}).get("selected_target")
                default_record = {str(col): None for col in raw_cols if col and col != target_col}
        except Exception:
            default_record = {"Age": None, "Fare": None}

        prediction_input_raw = st.text_area(
            "Enter prediction input as JSON object",
            value=json.dumps(default_record, indent=2),
            height=220,
        )

        if st.button("Predict"):
            try:
                record = json.loads(prediction_input_raw)
                payload = {"record": record}
                pred = api_post(f"{backend_url}/predict/{job_id}", payload, timeout=120)
                st.success("Prediction completed.")
                st.json(pred)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with tab_dl:
        if st.button("Prepare Model Download"):
            try:
                resp = requests.get(f"{backend_url}/download-model/{job_id}", timeout=120)
                resp.raise_for_status()
                st.download_button(
                    label="Download Trained Model (.joblib)",
                    data=resp.content,
                    file_name=f"{job_id}_pipeline.joblib",
                    mime="application/octet-stream",
                    width="stretch",
                )
            except Exception as exc:
                st.error(f"Model download failed: {exc}")

st.caption("Tip: Same CSV uploaded again creates a new job_id. Run AutoML for that new job_id separately.")
