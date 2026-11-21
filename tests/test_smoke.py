"""Run: uv run python -m unittest discover -s tests -v
Unit tests use mocked Gemini HTTP responses; NO API key or network required.
"""
import csv
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import pipeline
import monitor

ROOT = Path(__file__).resolve().parents[1]


class TestParcelPilot(unittest.TestCase):
    def test_contract_without_key(self):
        with patch.dict(os.environ, {"DEMO_MODE": "1"}):
            record = pipeline.run(pipeline.EXAMPLE)
        self.assertEqual(set(record), {"input_summary", "features", "outputs", "versions", "review", "tokens", "cost_eur"})
        self.assertIn("component_a", record["outputs"])
        self.assertIn("component_b", record["outputs"])
        self.assertTrue(record["review"])
        self.assertEqual(record["outputs"]["component_b"]["mode"], "offline_demo")
        self.assertEqual(record["tokens"], 0)
        self.assertTrue(all(type(v) in (int, float) for v in record["features"].values()))

    def test_strict_privacy_guard_rejects_free_text_without_echo(self):
        raw = "contact=jane.doe@example.org; " + pipeline.EXAMPLE
        rec = pipeline.run(raw)
        self.assertEqual(rec["outputs"]["component_a"]["label"], "out_of_scope")
        self.assertTrue(rec["review"])
        self.assertNotIn("jane", str(rec))
        self.assertNotIn("example.org", str(rec))
        self.assertEqual(rec["tokens"], 0)

    def test_real_tabular_training_is_not_fixed_label_template(self):
        a = pipeline.component_a(pipeline.parse_input(pipeline.EXAMPLE))
        b = pipeline.component_a(pipeline.parse_input(
            "issue=missing_parcel; days_late=20; previous_contacts=5; item_value_eur=500; damaged=0; refund_requested=1"))
        self.assertGreater(b["review_probability"], a["review_probability"])
        self.assertEqual(b["label"], "review")
        self.assertIn(a["model"], ("synthetic-logreg-v1-NOT-VALIDATED", "team-labeled-tabular-logreg-v1"))

    def test_mocked_gemini_function_call_actually_feeds_policy_tool(self):
        calls = []

        def fake_api(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {
                    "candidates": [{"content": {"role": "model", "parts": [
                        {"functionCall": {"id": "call_123", "name": "lookup_policy", "args": {"issue": "late_delivery"}}}
                    ]}}],
                    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 12, "totalTokenCount": 62},
                }
            return {
                "candidates": [{"content": {"role": "model", "parts": [
                    {"text": "Our staff can verify the delivery status. We cannot promise a date or refund."}
                ]}}],
                "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 22, "totalTokenCount": 102},
            }

        with patch.dict(os.environ, {"GEMINI_API_KEY": "MOCK_KEY", "DEMO_MODE": "0"}), patch.object(pipeline, "_api_request", side_effect=fake_api):
            result = pipeline.run(pipeline.EXAMPLE)
        self.assertEqual(result["outputs"]["component_b"]["mode"], "gemini_agent")
        self.assertEqual(result["outputs"]["component_b"]["tools_used"], ["lookup_policy"])
        self.assertEqual(len(calls), 2)
        self.assertIn("Tabular ML triage label", calls[0]["contents"][0]["parts"][0]["text"])
        tool_answer = calls[1]["contents"][-1]["parts"][0]["functionResponse"]
        self.assertIn("If a parcel is late", tool_answer["response"]["policy"])
        self.assertEqual(tool_answer["id"], "call_123")
        self.assertEqual(calls[0]["generationConfig"]["thinkingConfig"]["thinkingLevel"], "minimal")
        self.assertEqual(result["tokens"], 164)
        self.assertGreater(result["cost_eur"], 0.0)
        self.assertFalse(result["review"])

    def test_api_error_fails_closed(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "MOCK", "DEMO_MODE": "0"}), patch.object(
            pipeline, "_api_request", side_effect=RuntimeError("Test API outage")
        ):
            result = pipeline.run(pipeline.EXAMPLE)
        self.assertTrue(result["review"])
        self.assertEqual(result["outputs"]["component_b"]["mode"], "api_error_fallback")

    def test_psi_reference_and_shifted_demo(self):
        reference = monitor.load_reference()
        self.assertIsNotNone(reference, "Run `python main.py reference` first")
        def feats(name,kind=None):
            with (ROOT / "eval" / name).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return pd.DataFrame([pipeline.run(r["input"])["features"] for r in rows
                                 if kind is None or r.get("kind") == kind])
        with patch.dict(os.environ, {"DEMO_MODE": "1"}):
            stable = monitor.drift(reference, feats("cases.csv", "normal"))
            shifted = monitor.drift(reference, feats("shifted.csv"))
        self.assertFalse(bool(stable["alert"].any()))
        self.assertTrue(bool(shifted["alert"].any()))

    def test_bandit_uses_feedback(self):
        import logbook
        data = pd.DataFrame([{
            "versions": {"prompt_variant": "concise"},
            "outputs": {"component_b": {"mode": "gemini_agent"}},
            "feedback": -1,
        }])
        with patch.dict(os.environ, {"ENABLE_BANDIT": "1"}), patch.object(logbook, "load", return_value=data), patch(
            "pipeline.random.SystemRandom"
        ) as rng:
            rng.return_value.random.return_value = 0.99
            variant = pipeline.choose_prompt_variant()
        self.assertEqual(variant, "empathetic")


class TestGeminiRetry(unittest.TestCase):
    """Transient provider errors are retried; permanent ones are not."""

    def _http_error(self, code):
        import urllib.error
        return urllib.error.HTTPError("https://example.invalid", code, "error", {}, None)

    def test_retries_503_then_succeeds(self):
        import io, json as _json
        ok = io.BytesIO(_json.dumps({"candidates": []}).encode())
        with patch.dict(os.environ, {"GEMINI_API_KEY": "MOCK_KEY", "GEMINI_MAX_ATTEMPTS": "3"}), \
                patch("urllib.request.urlopen", side_effect=[self._http_error(503), self._http_error(503), ok]) as mocked, \
                patch("time.sleep") as slept:
            self.assertEqual(pipeline._api_request({}), {"candidates": []})
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(slept.call_count, 2)

    def test_does_not_retry_bad_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "MOCK_KEY", "GEMINI_MAX_ATTEMPTS": "3"}), \
                patch("urllib.request.urlopen", side_effect=self._http_error(403)) as mocked, \
                patch("time.sleep"):
            with self.assertRaises(RuntimeError):
                pipeline._api_request({})
        self.assertEqual(mocked.call_count, 1)

    def test_gives_up_after_max_attempts(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "MOCK_KEY", "GEMINI_MAX_ATTEMPTS": "3"}), \
                patch("urllib.request.urlopen", side_effect=TimeoutError()) as mocked, \
                patch("time.sleep"):
            with self.assertRaises(RuntimeError):
                pipeline._api_request({})
        self.assertEqual(mocked.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
