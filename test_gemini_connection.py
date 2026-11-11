"""One live Gemini smoke test with a non-identifying fictional case.

Run from the project root with `uv run python test_gemini_connection.py`.
This makes *two* real Gemini API calls if the API key is present.
Does not write to the course logbook.
"""
from __future__ import annotations

import os

import env

env.load_env()

import pipeline  # noqa: E402  (read .env before importing pipeline)

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise SystemExit("GEMINI_API_KEY missing. Add it to a .env file at the project root.")
    if os.environ.get("DEMO_MODE") == "1":
        raise SystemExit("DEMO_MODE=1 disables Gemini. Set DEMO_MODE=0 in .env and restart.")
    print("Using model:", os.environ.get("GEMINI_MODEL", pipeline.GEMINI_MODEL))
    print("Sending one fictional case; this should make two Gemini API calls.")
    result = pipeline.run(pipeline.EXAMPLE)
    assistant = result["outputs"]["component_b"]
    print("Mode:", assistant.get("mode"))
    print("Tools:", assistant.get("tools_used"))
    print("Tokens:", result["tokens"])
    print("Estimated cost EUR:", result["cost_eur"])
    print("Reply:", assistant.get("reply"))
    if assistant.get("mode") != "gemini_agent":
        raise SystemExit(
            "Gemini was NOT used successfully: "
            + assistant.get("error_message", "check .env, model, quota, and network")
        )
    if "lookup_policy" not in assistant.get("tools_used", []):
        raise SystemExit("Agent did not report a policy lookup.")
    print("SUCCESS: live Gemini response with policy-tool call.")
