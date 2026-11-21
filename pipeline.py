"""ParcelPilot: tabular ML triage -> Gemini tool-using policy assistant.

AI-generated starter code. The model is trained on SYNTHETIC, rule-labeled
examples and MUST be replaced/validated with team-owned data for submission.
No customer identifiers are accepted or logged; use invented aggregate cases.

Contract preserved for the course's main.py, app.py, check.py and logger.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

ISSUES = ("late_delivery", "damaged_item", "missing_parcel", "refund_request")
FIELD_ORDER = ("issue", "days_late", "previous_contacts", "item_value_eur", "damaged", "refund_requested")
FIELDS = set(FIELD_ORDER)
TRAINING_FILE = Path(__file__).resolve().parent / "data" / "train.csv"
MODEL_VERSION = "team-labeled-tabular-logreg-v1" if TRAINING_FILE.is_file() else "synthetic-logreg-v1-NOT-VALIDATED"
PROMPT_VERSION = "gemini-tools-policy-v1"
PROMPT_VARIANTS = {
    "concise": "Write a short direct reply, about 2 sentences.",
    "empathetic": "Use a warm, respectful tone while remaining factual, about 3 sentences.",
    "steps": "Use a short numbered description of the next steps without promising any action.",
}
BANDIT_EPSILON = 0.10  # PROVISIONAL exploration policy; justify in REPORT.md.


def choose_prompt_variant() -> str:
    """Optional epsilon-greedy bandit using actual 👍/👎 in app logs.

    Enabled only by ENABLE_BANDIT=1. Unrated and offline-mode examples provide
    NO reward. The next request reads latest log feedback and changes the policy.
    """
    if os.environ.get("ENABLE_BANDIT", "0") != "1":
        return "concise"
    try:
        import logbook  # lazy: avoids a dependency for headless pipeline testing
        logs = logbook.load()
    except Exception:  # empty or unavailable log store: safe cold start
        logs = None
    scores = {name: [0, 0] for name in PROMPT_VARIANTS}
    if logs is not None and not logs.empty:
        for rec in logs.to_dict(orient="records"):
            rated = rec.get("feedback")
            version_info = rec.get("versions")
            response_info = rec.get("outputs")
            if not isinstance(version_info, dict) or not isinstance(response_info, dict):
                continue
            variant = version_info.get("prompt_variant")
            component = response_info.get("component_b") or {}
            if variant not in scores or component.get("mode") != "gemini_agent":
                continue
            try:
                reward = int(rated)
            except (ValueError, TypeError):
                continue
            if reward in (-1, 1):
                scores[variant][0] += 1 if reward == 1 else 0
                scores[variant][1] += 1
    rng = random.SystemRandom()
    if rng.random() < BANDIT_EPSILON:
        return rng.choice(list(PROMPT_VARIANTS))
    return max(PROMPT_VARIANTS, key=lambda name: ((scores[name][0] + 1) / (scores[name][1] + 2),
                                                  -list(PROMPT_VARIANTS).index(name)))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
VERSIONS = {"component_a": MODEL_VERSION, "component_b": f"{GEMINI_MODEL}/{PROMPT_VERSION}"}
# These are DEMONSTRATION choices, not academically justified thresholds.
REVIEW_THRESHOLD = 0.60
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.40, 0.60
MAX_TEXT_LENGTH = 500

# Never request private data, order numbers, photos, or account details.
EXAMPLE = "issue=late_delivery; days_late=4; previous_contacts=1; item_value_eur=35; damaged=0; refund_requested=0"

# Only this tiny, static policy KB can be retrieved by the agent. It performs no transactions.
POLICIES = {
    "late_delivery": (
        "If a parcel is late, explain that delivery status must be checked by a staff member. "
        "Do not promise a delivery date or refund. Escalate repeat contacts or significant delays."
    ),
    "damaged_item": (
        "Ask staff to check the applicable damage and return policies; no photos or identifying "
        "details should be submitted to this demo. Do not promise a replacement or refund."
    ),
    "missing_parcel": (
        "Missing parcels always require staff verification. Do not claim an investigation "
        "has started, or that the item is located. Do not promise compensation."
    ),
    "refund_request": (
        "Refund eligibility depends on account and purchase policy checks performed "
        "outside this demo. No refund may be approved or promised by the assistant."
    ),
}


class InputProblem(ValueError):
    """The input is incomplete, unsafe, or outside the accepted data schema."""


def parse_input(raw_input: str) -> dict:
    """Accept ONLY predefined aggregate fields; never ingest free-text PII."""
    if not isinstance(raw_input, str) or not raw_input.strip():
        raise InputProblem("Use the example format and fill in all six fields.")
    if len(raw_input) > MAX_TEXT_LENGTH:
        raise InputProblem("Input is too long. No names, contact details or order IDs are accepted.")
    values = {}
    for piece in raw_input.split(";"):
        if not piece.strip():
            continue
        if "=" not in piece:
            raise InputProblem("Enter six semicolon-separated key=value fields. Do not add free text.")
        key, value = (part.strip().lower() for part in piece.split("=", 1))
        if key not in FIELDS or key in values:
            raise InputProblem("Unknown or duplicate field. Only the six example fields are allowed.")
        # Reject content containing personal data or malicious prompt content by using strict enums/ranges.
        if not re.fullmatch(r"[a-z_]+|[0-9]+(?:\.[0-9]{1,2})?", value, re.ASCII):
            raise InputProblem(f"Invalid value in '{key}'. Use an allowed category or number only.")
        values[key] = value
    if set(values) != FIELDS:
        raise InputProblem("Missing: " + ", ".join(sorted(FIELDS - set(values))) + ".")
    if values["issue"] not in ISSUES:
        raise InputProblem("'issue' is outside the supported process. Choose: " + ", ".join(ISSUES))

    result = {"issue": values["issue"]}
    limits = {
        "days_late": (0, 60),
        "previous_contacts": (0, 10),
        "item_value_eur": (0, 2000),
        "damaged": (0, 1),
        "refund_requested": (0, 1),
    }
    for key, (low, high) in limits.items():
        try:
            number = float(values[key])
        except ValueError as exc:
            raise InputProblem(f"'{key}' must be a number.") from exc
        if not low <= number <= high:
            raise InputProblem(f"'{key}' must be between {low} and {high}.")
        if key != "item_value_eur" and not number.is_integer():
            raise InputProblem(f"'{key}' must be a whole number.")
        result[key] = round(number, 2) if key == "item_value_eur" else int(number)
    return result


def _feature_array(row: dict) -> np.ndarray:
    """One row of explicitly tabular, numerical features for a logistic model."""
    return np.array([
        row["days_late"] / 30.0,
        row["previous_contacts"] / 5.0,
        row["item_value_eur"] / 500.0,
        float(row["damaged"]),
        float(row["refund_requested"]),
        *[float(row["issue"] == issue) for issue in ISSUES],
    ], dtype=np.float64)


def _synthetic_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Temporary illustration ONLY: synthetic features, synthetic labels.

    The rules below embed assumptions rather than ground truth. Replace with
    team-authored labeled table and document the source/licence in REPORT.md.
    """
    rng = np.random.default_rng(2026)
    inputs, targets = [], []
    for i in range(2400):
        routine_pool = i % 2 == 0
        issue_choices = ISSUES[:1] + ISSUES[1:2] + ISSUES[3:] if routine_pool else ISSUES
        issue = issue_choices[int(rng.integers(0, len(issue_choices)))]
        row = {
            "issue": issue,
            "days_late": int(rng.integers(0, 6 if routine_pool else 32)),
            "previous_contacts": int(rng.integers(0, 2 if routine_pool else 6)),
            "item_value_eur": float(rng.uniform(5, 130 if routine_pool else 450)),
            "damaged": int(issue == "damaged_item" or (not routine_pool and rng.random() < 0.08)),
            "refund_requested": int(issue == "refund_request" or (not routine_pool and rng.random() < 0.18)),
        }
        # Synthetic routing policy: not a validated business rule.
        risky = (row["days_late"] >= 8 or row["previous_contacts"] >= 3
                 or row["item_value_eur"] >= 220 or issue == "missing_parcel"
                 or (row["damaged"] and row["refund_requested"]))
        label = int(risky)
        if rng.random() < 0.04:  # add label noise to avoid pretending perfect labels
            label = 1 - label
        inputs.append(_feature_array(row))
        targets.append(label)
    return np.stack(inputs), np.array(targets, dtype=np.float64)


def _labeled_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Read optional anonymous TEAM-labeled training file; never use eval cases here.

    Required `data/train.csv` columns are the six input fields plus
    `requires_review` (0/1). Use 100+ independent annotated cases where possible.
    """
    with TRAINING_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 40:
        raise ValueError("data/train.csv needs at least 40 independently labeled cases")
    xs, ys = [], []
    for r in rows:
        try:
            label = str(r["requires_review"]).strip()
            if label not in {"0", "1"}:
                raise ValueError("requires_review must be 0 or 1")
            case = parse_input("; ".join(f"{name}={r[name]}" for name in FIELD_ORDER))
        except (KeyError, InputProblem) as exc:
            raise ValueError("data/train.csv has invalid columns or row values") from exc
        xs.append(_feature_array(case))
        ys.append(int(label))
    if len(set(ys)) < 2:
        raise ValueError("data/train.csv must include both 0 and 1 labels")
    return np.stack(xs), np.asarray(ys, dtype=np.float64)


@lru_cache(maxsize=1)
def _fit_model() -> tuple[np.ndarray, float]:
    """Fit a regularized binary logistic regression by gradient descent."""
    x, y = _labeled_training_data() if TRAINING_FILE.is_file() else _synthetic_training_data()
    w, bias = np.zeros(x.shape[1]), 0.0
    for _ in range(1400):
        z = np.clip(x @ w + bias, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w -= 0.17 * (x.T @ (p - y) / len(x) + 0.005 * w)
        bias -= 0.17 * float(np.mean(p - y))
    return w, bias


def component_a(row: dict) -> dict:
    """Model scores whether staff review is needed (uncalibrated probability)."""
    started = time.perf_counter()
    w, bias = _fit_model()
    score = float(np.dot(_feature_array(row), w) + bias)
    p_review = 1 / (1 + np.exp(-np.clip(score, -30, 30)))
    # Critical cases are ALWAYS escalated, regardless of ML output.
    forced = row["issue"] == "missing_parcel"
    uncertain = UNCERTAIN_LOW <= p_review <= UNCERTAIN_HIGH
    review = forced or uncertain or p_review >= REVIEW_THRESHOLD
    return {
        "label": "review" if review else "routine",
        "confidence": round(float(max(p_review, 1 - p_review)), 4),
        "review_probability": round(p_review, 4),
        "forced_review": bool(forced),
        "uncertain": bool(uncertain),
        "latency_s": round(time.perf_counter() - started, 4),
        "model": MODEL_VERSION,
    }


def _api_request(payload: dict) -> dict:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    model = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    # Retry transient provider failures (overload, rate limit, timeouts) with
    # exponential backoff. Permanent errors (bad key, bad model, bad request)
    # fail immediately. In the 2026-11-19 evaluation, 17/39 live calls failed
    # with HTTP 503 or a timeout and no retry, so every one fell back to review.
    attempts = max(1, int(os.environ.get("GEMINI_MAX_ATTEMPTS", "3")))
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_HTTP_CODES and attempt < attempts:
                time.sleep(_backoff_seconds(attempt))
                continue
            # Never expose the raw provider response: it might contain sensitive text.
            raise RuntimeError(f"Gemini API HTTP {exc.code}; check key, quota and model name") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < attempts:
                time.sleep(_backoff_seconds(attempt))
                continue
            raise RuntimeError("Gemini API connection timed out or is unavailable") from exc
    raise RuntimeError("Gemini API connection timed out or is unavailable")


RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _backoff_seconds(attempt: int) -> float:
    """2 s, 4 s, 8 s ... plus up to 1 s of jitter so parallel users do not retry in lockstep."""
    return 2 ** attempt + random.random()


def _extract(response: dict) -> tuple[dict, dict]:
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no response (possible safety block)")
    content = candidates[0].get("content") or {}
    return content, response.get("usageMetadata") or {}


def _usage(metadata: dict) -> tuple[int, int, int]:
    prompt = int(metadata.get("promptTokenCount", 0))
    completion = int(metadata.get("candidatesTokenCount", 0)) + int(metadata.get("thoughtsTokenCount", 0))
    total = int(metadata.get("totalTokenCount", prompt + completion))
    return prompt, completion, total


def _estimated_eur(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    """Paid *standard-tier* token cost estimate (not a real invoice).

    Rates from the public Google Gemini pricing page, checked Nov 2026;
    values can be overridden. The illustrative FX factor is NOT live.
    """
    current_model = model or os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
    published_rates = {
        "gemini-3.5-flash-lite": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
    }
    default_input, default_output = published_rates.get(current_model, (0.30, 2.50))
    usd_input = float(os.environ.get("GEMINI_USD_PER_M_INPUT", str(default_input)))
    usd_output = float(os.environ.get("GEMINI_USD_PER_M_OUTPUT", str(default_output)))
    usd_to_eur = float(os.environ.get("USD_TO_EUR", "0.90"))
    return round((input_tokens * usd_input + output_tokens * usd_output) / 1e6 * usd_to_eur, 7)


def _safe_reply(text: str) -> str:
    """Defense-in-depth: remove any contact details echoed by a model."""
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "[redacted]", text)
    text = re.sub(r"https?://\S+", "[redacted URL]", text)
    text = re.sub(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)", "[redacted number]", text)
    return text[:1200]


def _offline_reply(row: dict, a: dict, variant: str = "concise") -> dict:
    """Visibly flagged FALLBACK, not a replacement for the LLM evaluation."""
    label = "Staff review required." if a["label"] == "review" else "Routine case (demo only)."
    return {
        "reply": f"[OFFLINE DEMONSTRATION — NOT AN LLM OUTPUT] {label} " + POLICIES[row["issue"]],
        "tools_used": ["local_policy_template"],
        "mode": "offline_demo",
        "prompt_variant": variant,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens": 0,
        "cost_eur": 0.0,
        "latency_s": 0.0,
    }


def _generation_config(max_tokens: int, model: str | None = None) -> dict:
    """Use Gemini 3-compatible generation settings without breaking Gemini 2.5.

    Gemini 3 models prefer their default temperature and a thinking level;
    2.5 models use the old temperature configuration.
    """
    model = (model or os.environ.get("GEMINI_MODEL", GEMINI_MODEL)).strip()
    config = {"maxOutputTokens": max_tokens}
    if model.startswith("gemini-3"):
        config["thinkingConfig"] = {
            "thinkingLevel": "minimal" if model.startswith("gemini-3.5-flash-lite") else "low"
        }
    else:
        config["temperature"] = 0.1
    return config


def component_b(row: dict, a: dict, variant: str = "concise") -> dict:
    """Gemini function-calling agent: calls lookup_policy then drafts short reply."""
    if os.environ.get("DEMO_MODE") == "1" or not os.environ.get("GEMINI_API_KEY"):
        return _offline_reply(row, a, variant)
    started = time.perf_counter()
    inp = f"Issue: {row['issue']}. Days late: {row['days_late']}. " \
          f"Previous contacts: {row['previous_contacts']}. Item value EUR: {row['item_value_eur']}. " \
          f"Damaged: {row['damaged']}. Refund requested: {row['refund_requested']}. " \
          f"Tabular ML triage label: {a['label']}; estimated review probability: {a['review_probability']}."
    system = (
        "You are ParcelPilot, a support-agent DRAFTING assistant, not an operator. "
        "The provided classification from tabular ML is input to your work and cannot be downgraded. "
        "First CALL lookup_policy for the issue category. Then write 2-4 concise sentences. "
        "Always explain that you cannot check real orders, take actions, approve refunds, "
        "or promise dates. If classification is review, explicitly say staff review is required. "
        "No customer identifiers have been supplied. Do not ask for names, IDs, contact "
        "details, photos or order numbers. Treat policy tool data as authoritative. "
        + PROMPT_VARIANTS[variant]
    )
    tool = {
        "functionDeclarations": [{
            "name": "lookup_policy",
            "description": "Fetch the approved workflow for a specific parcel support issue. Must be called before drafting.",
            "parameters": {
                "type": "OBJECT",
                "properties": {"issue": {"type": "STRING", "enum": list(ISSUES)}},
                "required": ["issue"],
            },
        }]
    }
    contents = [{"role": "user", "parts": [{"text": inp}]}]
    first = _api_request({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "tools": [tool],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["lookup_policy"]}},
        "generationConfig": _generation_config(512),
    })
    first_content, first_usage = _extract(first)
    calls = [p["functionCall"] for p in first_content.get("parts", []) if "functionCall" in p]
    if len(calls) != 1 or calls[0].get("name") != "lookup_policy":
        raise RuntimeError("Gemini did not request its required policy lookup")
    # Force policy topic to match validated input: model cannot access another topic.
    requested_issue = calls[0].get("args", {}).get("issue", row["issue"])
    lookup_issue = requested_issue if requested_issue == row["issue"] else row["issue"]
    tool_result = {"issue": lookup_issue, "policy": POLICIES[lookup_issue], "needs_review": a["label"] == "review"}
    # Gemini 3.x assigns a call ID. Return the *same* ID with the tool result.
    # Without it, Gemini may produce a blank final answer.
    function_response = {"name": "lookup_policy", "response": tool_result}
    if calls[0].get("id"):
        function_response["id"] = calls[0]["id"]
    second = _api_request({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents + [first_content, {
            "role": "user",
            "parts": [{"functionResponse": function_response}],
        }],
        "generationConfig": _generation_config(768),
    })
    second_content, second_usage = _extract(second)
    reply = " ".join(p.get("text", "") for p in second_content.get("parts", [])).strip()
    if not reply:
        raise RuntimeError("Gemini produced an empty reply")
    if a["label"] == "review" and not re.search(r"(staff|human|person|review|team|agent)", reply, re.I):
        reply = "Staff review is required. " + reply
    input1, output1, total1 = _usage(first_usage)
    input2, output2, total2 = _usage(second_usage)
    return {
        "reply": _safe_reply(reply),
        "tools_used": ["lookup_policy"],
        "mode": "gemini_agent",
        "prompt_variant": variant,
        "input_tokens": input1 + input2,
        "output_tokens": output1 + output2,
        "tokens": total1 + total2,
        "cost_eur": _estimated_eur(input1 + input2, output1 + output2),
        "latency_s": round(time.perf_counter() - started, 3),
    }


def run(raw_input: str) -> dict:
    """Public contract read by app.py, evaluate.py, main.py and check.py."""
    try:
        row = parse_input(raw_input)
    except InputProblem as exc:
        # Never log potentially identifying raw input, including rejected values.
        a = {"label": "out_of_scope", "confidence": 1.0, "review_probability": 1.0,
             "forced_review": True, "uncertain": False, "latency_s": 0.0, "model": "schema_guard-v1"}
        b = {"reply": f"Cannot process: {str(exc)[:180]}. No data was retained.",
             "tools_used": [], "mode": "not_called", "input_tokens": 0,
             "output_tokens": 0, "tokens": 0, "cost_eur": 0.0, "latency_s": 0.0}
        return {"input_summary": "REJECTED: schema or out of scope", "features": {
                    "days_late": 0.0, "previous_contacts": 0.0, "item_value_eur": 0.0,
                    "damaged": 0.0, "refund_requested": 0.0, "review_probability": 1.0,
                    "out_of_scope": 1.0,
                }, "outputs": {"component_a": a, "component_b": b},
                "versions": dict(VERSIONS), "review": True, "tokens": 0, "cost_eur": 0.0}

    a = component_a(row)
    variant = choose_prompt_variant()
    try:
        b = component_b(row, a, variant)
    except (RuntimeError, ValueError) as exc:
        # Fail closed: do not present API outages as an automated successful answer.
        b = _offline_reply(row, a, variant)
        b["mode"] = "api_error_fallback"
        b["reply"] = "[Gemini temporarily unavailable; staff review required.] " + b["reply"]
        b["error_type"] = type(exc).__name__  # never record API key or provider response
        b["error_message"] = str(exc)[:180]  # only locally constructed safe errors
    should_review = a["label"] == "review" or b["mode"] != "gemini_agent"
    features = {
        "days_late": float(row["days_late"]),
        "previous_contacts": float(row["previous_contacts"]),
        "item_value_eur": float(row["item_value_eur"]),
        "damaged": float(row["damaged"]),
        "refund_requested": float(row["refund_requested"]),
        "issue_code": float(ISSUES.index(row["issue"])),
        "review_probability": float(a["review_probability"]),
        "confidence": float(a["confidence"]),
        "out_of_scope": 0.0,
    }
    return {
        "input_summary": "; ".join(f"{key}={row[key]}" for key in FIELD_ORDER),
        "features": features,
        "outputs": {"component_a": a, "component_b": b},
        "versions": {"component_a": MODEL_VERSION, "component_b": f"{os.environ.get('GEMINI_MODEL', GEMINI_MODEL)}/{PROMPT_VERSION}",
                     "prompt_variant": variant, "bandit": "epsilon-greedy-v1" if os.environ.get("ENABLE_BANDIT") == "1" else "off"},
        "review": bool(should_review),
        "tokens": int(b["tokens"]),
        "cost_eur": float(b["cost_eur"]),
    }


if __name__ == "__main__":
    print(json.dumps(run(EXAMPLE), indent=2))
