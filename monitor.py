"""Operations dashboard: request quality, business-rule alerts and PSI drift.

ALL THRESHOLDS BELOW ARE PROVISIONAL and must be chosen and justified by team
using an independent reference, acceptable false alarm rates and a drift demo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE_FILE = ROOT / "monitoring" / "reference_features.csv"
PSI_BINS = 5
PSI_ALERT = 0.25
NOISE_MULTIPLE = 2
WINDOW = 100
MIN_WINDOW = 20
THUMBS_DOWN_ALERT = 0.4
MIN_RATED = 10
REVIEW_RATE_ALERT = 0.75


def _bin(values: np.ndarray, reference: np.ndarray, bins: int) -> np.ndarray:
    unique = np.unique(reference)
    if len(unique) <= bins:
        index = np.searchsorted(unique, values)
        found = (index < len(unique)) & (unique[np.minimum(index, len(unique) - 1)] == values)
        return np.where(found, index, len(unique))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def psi(reference, current, bins: int = PSI_BINS) -> float:
    """Population stability index; features must be numeric, non-PII."""
    a, b = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    ia, ib = _bin(a, a, bins), _bin(b, a, bins)
    n = max(int(ia.max()), int(ib.max())) + 1
    pa = np.bincount(ia, minlength=n) / len(a)
    pb = np.bincount(ib, minlength=n) / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())


def psi_noise(n_reference: int, n_current: int, bins: int = PSI_BINS) -> float:
    return (bins - 1) * (1 / max(n_reference, 1) + 1 / max(n_current, 1))


def load_reference() -> pd.DataFrame | None:
    return pd.read_csv(REFERENCE_FILE) if REFERENCE_FILE.exists() else None


def drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in reference.columns:
        if col not in current:
            continue
        value = psi(reference[col], current[col])
        n_ref, n_live = int(reference[col].notna().sum()), int(current[col].notna().sum())
        threshold = max(PSI_ALERT, NOISE_MULTIPLE * psi_noise(n_ref, n_live))
        rows.append({"feature": col, "psi": round(value, 3) if np.isfinite(value) else np.nan,
                     "threshold": round(threshold, 3),
                     "alert": bool(np.isfinite(value) and value > threshold)})
    return pd.DataFrame(rows, columns=["feature", "psi", "threshold", "alert"])


def live_features(logs: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in logs.columns if c.startswith("f_")]
    return logs[cols].tail(WINDOW).rename(columns=lambda s: s[2:])


def quality(logs: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame(columns=["time", "signal", "value"])
    df = logs.set_index("timestamp").sort_index()
    rated = df["feedback"].dropna().astype(float)
    signals = pd.DataFrame({
        "share sent to review": df["review"].astype(float).resample(freq).mean(),
        "share 👎 (rated only)": (rated < 0).astype(float).resample(freq).mean(),
    })
    if "f_confidence" in df:
        signals["mean classifier confidence"] = df["f_confidence"].astype(float).resample(freq).mean()
    return signals.reset_index(names="time").melt("time", var_name="signal", value_name="value").dropna()


def summary(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame({"metric": ["requests", "feedback"], "value": ["0", "0"]})
    rated = logs["feedback"].dropna().astype(float)
    stats = {
        "requests": len(logs),
        "staff review share": f"{logs['review'].mean():.1%}",
        "feedback count": len(rated),
        "negative among rated": f"{(rated < 0).mean():.1%}" if len(rated) else "insufficient",
        "median latency (s)": round(float(logs["latency_s"].median()), 3),
        "mean estimated API cost (€)": round(float(logs["cost_eur"].mean()), 6),
        "total tokens": int(logs["tokens"].sum()),
        "total estimated API cost (€)": round(float(logs["cost_eur"].sum()), 6),
    }
    return pd.DataFrame({"metric": list(stats), "value": [str(x) for x in stats.values()]})


def alerts(logs: pd.DataFrame, reference: pd.DataFrame | None) -> list[str]:
    fired = []
    if logs.empty:
        return fired
    window = logs.tail(WINDOW)
    if reference is not None and len(window) >= MIN_WINDOW:
        report = drift(reference, live_features(window))
        for row in report[report["alert"]].itertuples():
            fired.append(f"Input drift `{row.feature}`: PSI {row.psi} > {row.threshold} ({len(window)} requests).")
        if float(window["review"].mean()) > REVIEW_RATE_ALERT:
            fired.append(f"Staff-review rate {window['review'].mean():.0%} exceeds {REVIEW_RATE_ALERT:.0%}.")
    rated = window["feedback"].dropna().astype(float)
    if len(rated) >= MIN_RATED and float((rated < 0).mean()) > THUMBS_DOWN_ALERT:
        fired.append(f"Negative feedback {((rated < 0).mean()):.0%} exceeds {THUMBS_DOWN_ALERT:.0%} on {len(rated)} rated cases.")
    return fired


def prompt_feedback(logs: pd.DataFrame) -> pd.DataFrame:
    """Observed live thumbs by bandit prompt variant (not causal evaluation)."""
    variants = ("concise", "empathetic", "steps")
    rows = []
    for variant in variants:
        matched = []
        if not logs.empty:
            for rec in logs.to_dict(orient="records"):
                version_info = rec.get("versions")
                output_info = rec.get("outputs")
                if not isinstance(version_info, dict) or not isinstance(output_info, dict):
                    continue
                if version_info.get("prompt_variant") != variant:
                    continue
                if (output_info.get("component_b") or {}).get("mode") != "gemini_agent":
                    continue
                matched.append(rec)
        votes = []
        for rec in matched:
            try:
                vote = int(rec.get("feedback"))
            except (TypeError, ValueError):
                continue
            if vote in (-1, 1):
                votes.append(vote)
        likes = sum(v == 1 for v in votes)
        rows.append({
            "prompt_variant": variant,
            "live_requests": len(matched),
            "rated_requests": len(votes),
            "thumbs_up_share": round(likes / len(votes), 3) if votes else float("nan"),
            "smoothed_bandit_reward": round((likes + 1) / (len(votes) + 2), 3),
        })
    return pd.DataFrame(rows)
