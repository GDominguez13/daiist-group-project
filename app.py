"""ParcelPilot Gradio interface; keeps required serve() and /run API."""
from __future__ import annotations

import os
import time

import gradio as gr

import env
import logbook
import monitor
import pipeline

env.load_env()
LOG_TARGET = logbook.start()
TITLE = "ParcelPilot | Privacy-first parcel-support triage"
DESCRIPTION = (
    "**Experimental assistant:** a trained *tabular* risk classifier passes its label "
    "to a tool-using Gemini policy assistant. Staff retain all decisions. "
    "This is a course prototype, **not a live customer-support service**."
)


def serve(raw_input: str) -> dict:
    """Keep this signature: check.py and the hosted /run endpoint rely on it."""
    started = time.perf_counter()
    result = pipeline.run(raw_input)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


def on_run(raw_input: str):
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Enter an aggregate case. You can click an example first.")
    record = serve(raw_input)
    a, b = record["outputs"]["component_a"], record["outputs"]["component_b"]
    status = "⚠️ **Staff review required**" if record["review"] else "✅ **Routine draft (not an action)**"
    mode = b.get("mode", "unknown")
    if mode != "gemini_agent":
        status += " — **No Gemini LLM was used**; output is a rule-based demonstration or fallback."
    if a["label"] == "out_of_scope":
        status = "⛔ **Input rejected.** Only non-identifying aggregate fields are accepted."
    status += (f"\n\nModel: `{a['model']}` · Policy assistant mode: `{mode}` · "
               f"Prompt variant: `{b.get('prompt_variant', 'none')}` · "
               f"Latency: {record['latency_s']:.3f} s · "
               f"Tokens: {record['tokens']} · Est. cost: €{record['cost_eur']:.6f}")
    if mode == "api_error_fallback":
        status += f"\n\n⚠️ Gemini error: `{b.get('error_message', 'Unknown error')}`"
    return a, b["reply"], status, record["id"]


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request before rating its output.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Feedback saved to the logs. This does not override staff decisions.")


def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "Reference missing. Run `DEMO_MODE=1 uv run python main.py reference` and commit `monitoring/reference_features.csv`."
    elif fired:
        alert_text = "### 🚨 Alerts\n" + "\n".join(f"- {message}" for message in fired)
    else:
        alert_text = "### ✅ No alerts (thresholds and reference are provisional)"
    if reference is None:
        drift_table = None
    elif len(logs) >= monitor.MIN_WINDOW:
        drift_table = monitor.drift(reference, monitor.live_features(logs))
    else:
        drift_table = monitor.drift(reference, reference.head(0))
    return alert_text, monitor.summary(logs), monitor.quality(logs), drift_table, monitor.prompt_feedback(logs)


EXAMPLES = [
    [pipeline.EXAMPLE],
    ["issue=damaged_item; days_late=2; previous_contacts=2; item_value_eur=48; damaged=1; refund_requested=0"],
    ["issue=missing_parcel; days_late=11; previous_contacts=3; item_value_eur=250; damaged=0; refund_requested=1"],
    ["issue=medical_advice; days_late=0; previous_contacts=0; item_value_eur=0; damaged=0; refund_requested=0"],
]

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")
    gr.Markdown(
        "**Privacy rule:** do not enter names, addresses, emails, phone numbers, tracking IDs, "
        "free text, or real customer data. This form accepts only six aggregate fields. "
        "The policy library is a demo, not an actual merchant's terms."
    )
    if not os.environ.get("GEMINI_API_KEY") or os.environ.get("DEMO_MODE") == "1":
        gr.Markdown("⚠️ **Offline demonstration active.** Set `GEMINI_API_KEY` (and unset `DEMO_MODE`) to use the genuine LLM agent.")
    if missing := env.missing():
        gr.Markdown(f"⚠️ Required environment variables missing: {', '.join(missing)}. See SETUP.md.")
    with gr.Tab("App"):
        request_id = gr.State("")
        raw_input = gr.Textbox(
            label="Aggregate case (semicolon-separated fields; no PII)",
            lines=3,
            value=pipeline.EXAMPLE,
        )
        gr.Examples(EXAMPLES, inputs=[raw_input], label="Demo cases, not a graded test set")
        run_button = gr.Button("Classify + draft reply", variant="primary")
        with gr.Row():
            out_a = gr.JSON(label="Component A · tabular logistic regression")
            out_b = gr.Textbox(label="Component B · policy-tool agent draft", lines=8)
        status = gr.Markdown()
        with gr.Row():
            up = gr.Button("👍 Helpful")
            down = gr.Button("👎 Unhelpful")
        run_button.click(on_run, raw_input, [out_a, out_b, status, request_id], api_name="run")
        up.click(lambda rid: on_feedback(rid, 1), request_id, None, api_name=False)
        down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)
    with gr.Tab("Monitoring"):
        gr.Markdown("These metrics are based on **actual app logs**. Demo thresholds have not been calibrated.")
        refresh = gr.Button("Refresh monitoring")
        alert_box = gr.Markdown()
        summary_table = gr.Dataframe(label="Operational summary", interactive=False)
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", label="Quality over time", height=300)
        drift_table = gr.Dataframe(
            label=f"Input drift: PSI over last {monitor.WINDOW} requests against reference", interactive=False
        )
        bandit_table = gr.Dataframe(label="Optional RL: prompt variant feedback from actual live agent calls", interactive=False)
        gr.Markdown(f"<small>{LOG_TARGET}</small>")
        monitored = [alert_box, summary_table, quality_plot, drift_table, bandit_table]
        refresh.click(on_refresh, inputs=None, outputs=monitored, api_name=False)
        demo.load(on_refresh, inputs=None, outputs=monitored, api_name=False)

if __name__ == "__main__":
    demo.launch()
