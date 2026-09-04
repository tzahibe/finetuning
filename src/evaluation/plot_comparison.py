import json
from pathlib import Path

import plotly.graph_objects as go

BEFORE_PATH = "data/reports/baseline_qwen2.5-coder-7b_report.json"
AFTER_PATH = "data/reports/qlora_pilot_150_20ex_report.json"
OUTPUT_HTML = "data/reports/qlora_pilot_150_before_after.html"
OUTPUT_PNG = "data/reports/qlora_pilot_150_before_after.png"

# (report key, display label, is_percentage)
METRICS = [
    ("json_validity_rate", "JSON validity", True),
    ("schema_validity_rate", "Schema validity", True),
    ("room_count_accuracy_mean", "Room count accuracy", True),
    ("hard_constraint_satisfaction_mean", "Hard constraint satisfaction", True),
    ("relationship_f1_mean", "Relationship F1", True),
]

AREA_METRICS = [
    ("area_abs_error_m2_mean", "Area absolute error (m²)"),
    ("area_rel_error_pct_mean", "Area relative error (%)"),
]


def build_figure(before: dict, after: dict) -> go.Figure:
    b = before["metrics"]
    a = after["metrics"]

    labels = [m[1] for m in METRICS]
    before_vals = [round((b[m[0]] or 0) * 100, 1) for m in METRICS]
    after_vals = [round((a[m[0]] or 0) * 100, 1) for m in METRICS]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Before (zero-shot base)",
            x=labels,
            y=before_vals,
            marker_color="#6B7280",
            text=[f"{v}%" for v in before_vals],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="After (QLoRA, 150 examples)",
            x=labels,
            y=after_vals,
            marker_color="#2563EB",
            text=[f"{v}%" for v in after_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Architect AI - Before vs After QLoRA Pilot (150 examples)",
        yaxis_title="% (of examples / of extracted facts)",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
    )
    return fig


def build_area_figure(before: dict, after: dict) -> go.Figure:
    b = before["metrics"]
    a = after["metrics"]
    labels = [m[1] for m in AREA_METRICS]
    before_vals = [b[m[0]] for m in AREA_METRICS]
    after_vals = [a[m[0]] for m in AREA_METRICS]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=labels, y=before_vals, marker_color="#F59E0B", text=before_vals, textposition="outside"))
    fig.add_trace(go.Bar(name="After (QLoRA-150)", x=labels, y=after_vals, marker_color="#16A34A", text=after_vals, textposition="outside"))
    fig.update_layout(
        title="Architect AI - Area Error, Before vs After",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
    )
    return fig


if __name__ == "__main__":
    before = json.loads(Path(BEFORE_PATH).read_text())
    after = json.loads(Path(AFTER_PATH).read_text())

    fig1 = build_figure(before, after)
    fig2 = build_area_figure(before, after)

    from plotly.subplots import make_subplots

    combined = make_subplots(rows=2, cols=1, subplot_titles=("Quality metrics (%)", "Area error"), vertical_spacing=0.15)
    for trace in fig1.data:
        combined.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        trace.showlegend = False
        combined.add_trace(trace, row=2, col=1)
    combined.update_layout(
        title="Architect AI QLoRA Pilot (150 examples) - Before vs After",
        barmode="group",
        template="plotly_white",
        height=800,
        font=dict(size=13),
    )

    combined.write_html(OUTPUT_HTML)
    try:
        combined.write_image(OUTPUT_PNG, scale=2)
    except Exception as e:
        print("PNG export skipped:", e)
    print("Wrote", OUTPUT_HTML)
