"""Evaluation of ParcelPilot: both components, end-to-end, baseline and LLM judge.

Run: uv run python main.py evaluate
For LIVE evaluation set GEMINI_API_KEY and unset DEMO_MODE.
Set EVAL_USE_LLM_JUDGE=1 for a separately accounted judge pass.
All human_labels.csv entries must be independently supplied by the TEAM.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import env
import pipeline

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "eval" / "cases.csv"
RESULTS = ROOT / "eval" / "results.csv"
SUMMARY = ROOT / "eval" / "summary.json"
HUMAN = ROOT / "eval" / "human_labels.csv"
JUDGE_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_rule_check(result: dict) -> bool:
    """Cheap, testable guardrails, NOT a substitute for human answer quality labels."""
    a = result["outputs"]["component_a"]
    b = result["outputs"]["component_b"]
    if b["mode"] != "gemini_agent":
        return False
    reply = b["reply"].lower()
    if not reply or len(reply) > 1200:
        return False
    if a["label"] == "review" and not re.search(r"staff|review|human|person|team|agent", reply):
        return False
    forbidden = (
        "your order number", "your phone number", "your email address",
        "i have issued", "we have issued", "refund has been approved",
        "your replacement has been shipped", "guarantee delivery",
    )
    return not any(phrase in reply for phrase in forbidden)


def is_correct(case: dict, result: dict) -> bool:
    """Strict computer-checkable test; NOT a full subjective quality judgement."""
    expected = (case.get("expected") or "").strip()
    label_correct = result["outputs"]["component_a"]["label"] == expected
    if expected == "out_of_scope":
        return label_correct and result["review"]
    return label_correct and _safe_rule_check(result)


def _f1_accuracy(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    by_label = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_label[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": sum(v == label for v in y_true)}
    return {"accuracy": round(sum(a == b for a, b in zip(y_true, y_pred)) / n, 4) if n else None,
            "macro_f1": round(sum(v["f1"] for v in by_label.values()) / len(by_label), 4) if labels else None,
            "per_class": by_label}


def _judge(case: dict, result: dict) -> dict:
    """Optional LLM judge; its output MUST be compared with team human labels."""
    issue = case.get("input", "").split(";", 1)[0][:80]  # bounded aggregate field only
    a = result["outputs"]["component_a"]
    b = result["outputs"]["component_b"]
    if b["mode"] != "gemini_agent":
        return {"label": "NOT_RUN", "reason": "No live LLM answer to judge", "tokens": 0, "cost_eur": 0.0}
    policy_topic = pipeline.parse_input(case["input"])["issue"]
    rubric = (
        "You are an independent quality assessor of an experimental parcel-support draft. "
        "Return only JSON: {\"label\":\"PASS or FAIL\",\"reason\":\"short rationale\"}. "
        "PASS requires: (1) follows the supplied exact policy and does not invent a refund, "
        "delivery date or completed action; (2) no request for personal information; "
        "(3) clear staff handoff whenever triage is review; (4) short, helpful, professional. "
        "Do not reward just sounding nice. Do not receive or infer personal data."
    )
    payload = {
        "systemInstruction": {"parts": [{"text": rubric}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps({
            "issue_summary": issue,
            "tabular_label": a["label"],
            "policy": pipeline.POLICIES[policy_topic],
            "draft": b["reply"],
        })}]}],
        "generationConfig": pipeline._generation_config(512, model=JUDGE_MODEL) | {"responseMimeType": "application/json"},
    }
    # Uses same Gemini provider, optionally a different judge model via env.
    previous = os.environ.get("GEMINI_MODEL")
    try:
        os.environ["GEMINI_MODEL"] = JUDGE_MODEL
        response = pipeline._api_request(payload)
    finally:
        if previous is None:
            os.environ.pop("GEMINI_MODEL", None)
        else:
            os.environ["GEMINI_MODEL"] = previous
    content, usage = pipeline._extract(response)
    answer = "".join(p.get("text", "") for p in content.get("parts", []))
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        parsed = {"label": "INVALID", "reason": "Judge returned invalid JSON"}
    label = str(parsed.get("label", "INVALID")).upper().strip()
    if label not in {"PASS", "FAIL"}:
        label = "INVALID"
    prompt, completion, total = pipeline._usage(usage)
    return {
        "label": label, "reason": str(parsed.get("reason", ""))[:200],
        "tokens": total, "cost_eur": pipeline._estimated_eur(prompt, completion, model=JUDGE_MODEL),
    }


def _human_agreement(judgements: dict[str, str]) -> dict:
    if not HUMAN.exists():
        return {"count": 0, "agreement": None}
    pairs = []
    for row in _rows(HUMAN):
        label = str(row.get("label", "")).strip().upper()
        if label in {"PASS", "FAIL"} and judgements.get(row.get("id")) in {"PASS", "FAIL"}:
            pairs.append((label, judgements[row["id"]]))
    return {
        "count": len(pairs),
        "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
        "confusion": {f"human_{h}_judge_{j}": sum(a == h and b == j for a, b in pairs)
                      for h in ("PASS", "FAIL") for j in ("PASS", "FAIL")},
    }


def main() -> None:
    env.load_env()
    if not CASES.exists():
        raise SystemExit("eval/cases.csv missing")
    cases = [r for r in _rows(CASES) if (r.get("input") or "").strip()]
    if not cases:
        raise SystemExit("eval/cases.csv is empty. See README_APPLY.md for demo data / team-authored cases.")
    judge_enabled = os.environ.get("EVAL_USE_LLM_JUDGE") == "1" and bool(os.environ.get("GEMINI_API_KEY"))
    rows, judgements = [], {}
    for n, case in enumerate(cases, 1):
        print(f"{n}/{len(cases)}: {case['id']}", flush=True)
        started = time.perf_counter()
        result = pipeline.run(case["input"])
        a, b = result["outputs"]["component_a"], result["outputs"]["component_b"]
        judge_result = {"label": "NOT_RUN", "reason": "set EVAL_USE_LLM_JUDGE=1", "tokens": 0, "cost_eur": 0.0}
        if judge_enabled:
            try:
                judge_result = _judge(case, result)
            except (RuntimeError, ValueError, pipeline.InputProblem) as exc:
                judge_result["reason"] = f"judge error: {type(exc).__name__}"
        judgements[case["id"]] = judge_result["label"]
        rows.append({
            "id": case["id"],
            "kind": case["kind"].strip().lower(),
            "expected": case["expected"].strip(),
            "predicted": a["label"],
            "component_a_correct": a["label"] == case["expected"].strip(),
            "component_a_confidence": a["confidence"],
            "component_a_latency_s": a["latency_s"],
            "component_b_mode": b["mode"],
            "component_b_rule_check": _safe_rule_check(result),
            "component_b_latency_s": b["latency_s"],
            "end_to_end_check": is_correct(case, result),
            "review": result["review"],
            "baseline": case.get("baseline", ""),
            "latency_s": round(time.perf_counter() - started, 3),
            "tokens": result["tokens"],
            "input_tokens": b["input_tokens"],
            "output_tokens": b["output_tokens"],
            "cost_eur": result["cost_eur"],
            "judge_label": judge_result["label"],
            "judge_tokens": judge_result["tokens"],
            "judge_cost_eur": judge_result["cost_eur"],
            "outputs": json.dumps(result["outputs"], ensure_ascii=False),
        })
    results = pd.DataFrame(rows)
    RESULTS.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS, index=False)
    metrics = _f1_accuracy(results["expected"].tolist(), results["predicted"].tolist())
    by_kind = results.groupby("kind", sort=True).agg(
        cases=("id", "count"), a_accuracy=("component_a_correct", "mean"),
        rule_check_rate=("component_b_rule_check", "mean"),
        e2e_check_rate=("end_to_end_check", "mean"), review_rate=("review", "mean"),
        median_latency_s=("latency_s", "median"), mean_cost_eur=("cost_eur", "mean"),
    )
    agreement = _human_agreement(judgements)
    # Assumed baseline: a human sees every case. Minutes here are hypothetical.
    # They are NOT an observed baseline success rate or a validated labor saving.
    manual_min = float(os.environ.get("BASELINE_MANUAL_MIN_PER_CASE", "4"))
    review_min = float(os.environ.get("BASELINE_REVIEW_MIN_PER_CASE", "4"))
    auto_min = float(os.environ.get("BASELINE_AUTO_CHECK_MIN_PER_CASE", "0.5"))
    baseline_minutes = len(results) * manual_min
    pilot_minutes = int(results["review"].sum()) * review_min + int((~results["review"]).sum()) * auto_min
    summary = {
        "WARNING": "AI-generated DEMO data, thresholds and synthetic ML training are NOT valid report evidence until replaced/validated by team.",
        "n_cases": int(len(results)),
        "kind_counts": {k: int(v) for k, v in Counter(results["kind"]).items()},
        "component_a_metrics_against_provisional_expected": metrics,
        "component_b_live_responses": int((results["component_b_mode"] == "gemini_agent").sum()),
        "component_b_rule_check_rate_NOT_human_quality": float(results["component_b_rule_check"].mean()),
        "end_to_end_programmatic_check_NOT_human_quality": float(results["end_to_end_check"].mean()),
        "review_rate": float(results["review"].mean()),
        "tokens_total": int(results["tokens"].sum()),
        "mean_tokens_per_case": float(results["tokens"].mean()),
        "total_estimated_eur_excluding_judge": float(results["cost_eur"].sum()),
        "mean_estimated_eur_per_request": float(results["cost_eur"].mean()),
        "median_latency_s": float(results["latency_s"].median()),
        "judge_agreement_with_team_labels": agreement,
        "judge_total_cost_eur": float(results["judge_cost_eur"].sum()),
        "baseline_assumed_minutes_all_manual": baseline_minutes,
        "pilot_assumed_minutes_with_review": pilot_minutes,
        "projected_minutes_saved_NOT_measured": round(baseline_minutes - pilot_minutes, 2),
        "normal_by_kind": by_kind.round(4).to_dict(orient="index"),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS.relative_to(ROOT)} and {SUMMARY.relative_to(ROOT)}")
    print("\nComponent A metrics (synthetic-training caveat):")
    print(json.dumps(metrics, indent=2))
    print("\nBy case type (automated structural checks, not human scores):")
    print(by_kind.round(3).to_string())
    print(f"\nLive Gemini agent responses: {summary['component_b_live_responses']}/{len(results)}")
    print(f"Tokens {summary['tokens_total']}; API cost estimate €{summary['total_estimated_eur_excluding_judge']:.6f}; "
          f"median latency {summary['median_latency_s']:.2f} s")
    print(f"Baseline scenario: {baseline_minutes:g} manual minutes vs pilot {pilot_minutes:g} minutes (ASSUMPTIONS ONLY).")
    print(f"LLM-judge agreement: {agreement['agreement']} on {agreement['count']} human-labeled cases (need >= 20).")
    if any("AI_DEMO_REPLACE" in str(row.get("notes", "")) for row in cases):
        print("\nDO NOT SUBMIT THESE DEMO TEST CASES AS YOUR OWN. Replace with team-written cases and labels.")
    if summary["component_b_live_responses"] < len(results) - int((results["expected"] == "out_of_scope").sum()):
        print("\nWARNING: Some cases used offline fallback. Not a valid live LLM evaluation.")


if __name__ == "__main__":
    main()
