import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

REPORTS_DIR = Path(__file__).parent / "reports" / "public"


def metric_key(name: str) -> str:
    name = name.lower()

    if "answer" in name and "relev" in name:
        return "Answer Relevancy"
    if "faith" in name:
        return "Faithfulness"
    if "context" in name and "relev" in name:
        return "Context Relevancy"
    if "correct" in name:
        return "Correctness"

    return name.title()


def get_value(data: dict, *keys, default=None):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def normalise_metric_name(name: str) -> str:
    name = name.lower().replace("_", " ")

    if "answer" in name and "relev" in name:
        return "Answer Relevancy"
    if "faith" in name:
        return "Faithfulness"
    if "context" in name and "relev" in name:
        return "Context Relevancy"
    if "correct" in name:
        return "Correctness"

    return name.title()


def to_score(value):
    if isinstance(value, dict):
        value = value.get("score")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_context(context):
    if isinstance(context, list):
        return "\n\n".join(str(item) for item in context)

    return str(context or "")


def load_report(path: Path) -> pd.DataFrame:
    raw_text = path.read_text(encoding="utf-8").strip()

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    if isinstance(raw, dict) and "testCases" in raw:
        cases = raw["testCases"]
    elif isinstance(raw, list):
        cases = raw
    else:
        cases = [raw]

    rows = []

    for index, case in enumerate(cases, start=1):
        metrics = {}

        # Format used by evaluated_results1.jsonl
        for metric in case.get("metricsData", []):
            metric_name = normalise_metric_name(metric.get("name", ""))
            metrics[metric_name] = to_score(metric.get("score"))

        # Format used by results_v2.jsonl
        direct_metrics = {
            "answer_relevancy": "Answer Relevancy",
            "answer_relevance": "Answer Relevancy",
            "faithfulness": "Faithfulness",
            "context_relevancy": "Context Relevancy",
            "contextual_relevancy": "Context Relevancy",
            "correctness": "Correctness",
        }

        for field, label in direct_metrics.items():
            if field in case:
                metrics[label] = to_score(case[field])

        scores = [score for score in metrics.values() if score is not None]

        passed = get_value(case, "success", "passed")

        if passed is None:
            passed = bool(scores) and all(score >= 0.7 for score in scores)

        rows.append(
            {
                "Test Case": get_value(
                    case,
                    "name",
                    "test_case",
                    "test_case_name",
                    default=f"Case {index}",
                ),
                "Question": get_value(case, "input", "question", default=""),
                "Answer": get_value(
                    case,
"actualOutput",
"actual_output",
"actual_answer",
"answer",
                    default="",
                ),
                "Expected Answer": get_value(
                    case,
                    "expectedOutput",
                    "expected_output",
                    "expected_answer",
                    default="",
                ),
                "Retrieved Context": format_context(
                    get_value(
                        case,
                        "retrievalContext",
                        "retrieval_context",
                        "context",
                        default=[],
                    )
                ),
                "Passed": bool(passed),
                "Latency (seconds)": to_score(
                    get_value(case, "runDuration", "run_duration", "latency", default=0)
                ) or 0,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


report_files = sorted(
    list(REPORTS_DIR.glob("*.json")) + list(REPORTS_DIR.glob("*.jsonl"))
)

st.title("RAG Evaluation Dashboard")
st.caption("Measuring retrieval quality, grounding, and answer correctness before deployment.")

if not report_files:
    st.error("No reports found. Add V1, V2, or V3 JSON files inside reports/public/.")
    st.stop()

reports = {file.stem.upper(): load_report(file) for file in report_files}
selected_version = st.sidebar.selectbox("Choose experiment", list(reports))
df = reports[selected_version].copy()

st.sidebar.markdown("---")
show_failures_only = st.sidebar.checkbox("Show failed cases only")

if show_failures_only:
    df = df[df["Passed"] == False]

total = len(df)
passed = int(df["Passed"].sum())
pass_rate = round((passed / total) * 100, 1) if total else 0

metric_columns = [
    column
    for column in [
        "Answer Relevancy",
        "Faithfulness",
        "Context Relevancy",
        "Correctness",
    ]
    if column in df.columns
]

average_latency = round(df["Latency (seconds)"].mean(), 2) if total else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Cases", total)
col2.metric("Pass Rate", f"{pass_rate}%")
col3.metric("Average Latency", f"{average_latency}s")
col4.metric("Failed Cases", total - passed)

st.markdown("---")

st.subheader(f"{selected_version} Quality Scores")

if metric_columns:
    metric_data = (
        df[metric_columns]
        .mean()
        .reset_index()
        .rename(columns={"index": "Metric", 0: "Average Score"})
    )

    chart = px.bar(
        metric_data,
        x="Metric",
        y="Average Score",
        range_y=[0, 1],
        color="Average Score",
        color_continuous_scale=["#ef4444", "#fbbf24", "#5eead4"],
    )

    st.plotly_chart(chart, use_container_width=True)
else:
    st.warning("No recognised metric fields found in this report.")

st.subheader("Experiment Comparison")

comparison_rows = []

for version, version_df in reports.items():
    row = {
        "Version": version,
        "Test Cases": len(version_df),
        "Pass Rate": round(version_df["Passed"].mean() * 100, 1),
        "Average Latency": round(version_df["Latency (seconds)"].mean(), 2),
    }

    for metric in metric_columns:
        if metric in version_df.columns:
            row[metric] = round(version_df[metric].mean(), 2)

    comparison_rows.append(row)

comparison = pd.DataFrame(comparison_rows)
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.subheader("Test Case Explorer")

visible_columns = [
    column
    for column in [
        "Test Case",
        "Question",
        "Passed",
        "Answer Relevancy",
        "Faithfulness",
        "Context Relevancy",
        "Correctness",
        "Latency (seconds)",
    ]
    if column in df.columns
]

st.dataframe(df[visible_columns], use_container_width=True, hide_index=True)

if not df.empty:
    selected_case = st.selectbox("Inspect one test case", df["Test Case"].tolist())
    case = df[df["Test Case"] == selected_case].iloc[0]

    with st.expander("Question and answer", expanded=True):
        st.markdown("**Question**")
        st.write(case["Question"])

        st.markdown("**Model answer**")
        st.write(case["Answer"])

        st.markdown("**Expected answer**")
        st.write(case["Expected Answer"])

    with st.expander("Retrieved context"):
        st.write(case["Retrieved Context"])

    with st.expander("Metric scores"):
        score_data = {
            metric: case[metric]
            for metric in metric_columns
            if pd.notna(case.get(metric))
        }
        st.json(score_data)

st.caption("Built by Jeevan Parihar · RAG Evaluation Harness")