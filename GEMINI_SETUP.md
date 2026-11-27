> Historical starter notes. Current deployment instructions are in SETUP.md and current evaluation instructions are in eval/README.md.

# ParcelPilot: exact Gemini setup (November 2026)

This upgrade is for the earlier `daiist_parcelpilot_starter_overlay.zip` app.
It does NOT require `google-genai` or any new Python package. It uses Google's
existing Gemini `generateContent` REST API via Python's built-in `urllib`.

## 1. Save your project, then install upgrade

- In VS Code, stop the app (`Ctrl+C` in the terminal).
- Optionally `git status`, then commit/stash current work.
- Unzip `daiist_gemini_upgrade.zip`.
- Copy **the contents of this ZIP into the root of your existing cloned fork**.
- Overwrite ONLY these existing files: `pipeline.py`, `app.py`, `evaluate.py`, `.env.example`, and `tests/test_smoke.py`.
- ADD the two new files `test_gemini_connection.py` and `GEMINI_SETUP.md`.
- KEEP `main.py`, `check.py`, `env.py`, `logbook.py`, `monitor.py`, `deploy.py`, data files and `uv.lock` unchanged.
- If you already created `.env`, KEEP it. This upgrade does not contain `.env` or any key.

## 2. Create `.env` in the SAME folder as app.py

Get your own key at https://aistudio.google.com/apikey.
Make `.env` by copying `.env.example` (on Windows, use VS Code New File and
name it exactly `.env`, no `.txt` extension). Set these lines:

```dotenv
GEMINI_API_KEY=YOUR_REAL_KEY_HERE
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_JUDGE_MODEL=gemini-3.5-flash-lite
DEMO_MODE=0
GEMINI_USD_PER_M_INPUT=0.30
GEMINI_USD_PER_M_OUTPUT=2.50
USD_TO_EUR=0.90
```

Leave optional HF variables blank for now. `USD_TO_EUR=0.90` is an
*illustrative*, NOT live, conversion rate; update it when writing your report.
Gemini 3.5 Flash-Lite paid standard-tier November-2026 token rates are
$0.30/M input and $2.50/M output (including thinking tokens); free tier may
charge $0. Reference: https://ai.google.dev/gemini-api/docs/pricing/

Do NOT put the secret into `.env.example`, `pipeline.py`, GitHub, screenshots,
assignment reports, or chat. The original `.gitignore` already ignores `.env`.

## 3. Test one LIVE request before opening the app

In VS Code `Terminal > New Terminal`, from the repository root:

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python test_gemini_connection.py
```

Expected last lines:

```text
Mode: gemini_agent
Tools: ['lookup_policy']
Tokens: <positive integer>
Reply: <a fresh drafted message>
SUCCESS: live Gemini response with policy-tool call.
```

The connection test uses ONE invented case and TWO paid/quota-consuming Gemini
API calls. It never prints your key. We have tested the code with mocked
Gemini replies, not with a live key: your test is the live verification.

If it fails, the last line should contain a safe error, for example
`Gemini API HTTP 429; check key, quota and model name`. `401/403` indicates
key or project access, `404` usually an unavailable model/endpoint,
`429` rate limit or quota, `400` request format or unsupported setting.
Check your Google AI Studio project before repeatedly retrying.

## 4. Restart the UI and verify it is live

```bash
uv run python app.py
```

Open http://127.0.0.1:7860 and click `Classify + draft reply` using the
pre-filled fake case. **The status must say `Policy assistant mode: gemini_agent`**
and show a positive token count. If it says `offline_demo` or
`api_error_fallback`, those are NOT Gemini answers. The new UI prints a safe
error reason below the response if the live API call fails.

`Component A` still uses an unvalidated synthetic training baseline unless
YOUR TEAM independently creates `data/train.csv`. The Gemini integration does
not make the synthetic model research-valid.

## 5. Do NOT burn quota on drift demos

`main.py reference`, `main.py drift-demo`, and the full `check.py` invoke
`pipeline.run()` many times. Those need only numeric features; you can disable
Gemini for those feature tests. In VS Code PowerShell (Windows):

```powershell
$env:DEMO_MODE='1'
uv run python main.py reference
uv run python main.py drift-demo
uv run python check.py
Remove-Item Env:DEMO_MODE
```

macOS/Linux (bash/zsh):

```bash
DEMO_MODE=1 uv run python main.py reference
DEMO_MODE=1 uv run python main.py drift-demo
DEMO_MODE=1 uv run python check.py
```

Keep `DEMO_MODE=0` in the actual `.env` so a normal app launch is LIVE.
For full live evaluation (after your team supplies independent cases), run
`uv run python main.py evaluate` WITHOUT DEMO_MODE=1. Set
`EVAL_USE_LLM_JUDGE=1` when ready for additional API calls, and compare judge
labels against **20+ independently supplied human labels**.

## 6. Why the files changed

- `pipeline.py`: defaults to Gemini 3.5 Flash-Lite, preserves returned Gemini
  `functionCall.id` as the matching `functionResponse.id`, uses Gemini 3's
  preferred thinking level instead of temperature, gives larger token budgets,
  updates pricing defaults, and exposes safe fallback errors.
- `app.py`: shows readable error status for failed Gemini calls.
- `evaluate.py`: judge defaults to Gemini 3.5 Flash-Lite and uses the same
  compatible generation config and model-specific cost calculation.
- `.env.example`: marks Gemini's key required for the final LIVE product.
- `tests/test_smoke.py`: checks that a mock Gemini 3 function-call ID is
  returned in the tool response (without any billable real request).

The model name and API features were checked against Google's November 2026
public docs, but availability/quota vary by key and account.

## 7. Optional: manual edits (if not overwriting from ZIP)

Make precisely these replacements in the EARLIER starter source:

### `pipeline.py`

1. Change the line beginning `GEMINI_MODEL =` to:

```python
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
```

2. Just BEFORE `def component_b(`, insert:

```python
def _generation_config(max_tokens: int, model: str | None = None) -> dict:
    model = (model or os.environ.get("GEMINI_MODEL", GEMINI_MODEL)).strip()
    config = {"maxOutputTokens": max_tokens}
    if model.startswith("gemini-3"):
        config["thinkingConfig"] = {
            "thinkingLevel": "minimal" if model.startswith("gemini-3.5-flash-lite") else "low"
        }
    else:
        config["temperature"] = 0.1
    return config

```

3. Within `component_b`, change the FIRST `generationConfig` from the old
`{"temperature": 0.1, "maxOutputTokens": 160}` to:

```python
"generationConfig": _generation_config(512),
```

4. After `tool_result = {"issue": ...}`, insert:

```python
    function_response = {"name": "lookup_policy", "response": tool_result}
    if calls[0].get("id"):
        function_response["id"] = calls[0]["id"]
```

Then change the second `contents` response's `parts` line from:

```python
"parts": [{"functionResponse": {"name": "lookup_policy", "response": tool_result}}],
```

to:

```python
"parts": [{"functionResponse": function_response}],
```

5. In the SECOND `_api_request` payload, change the generation config to:

```python
"generationConfig": _generation_config(768),
```

6. Under `b["error_type"] = ...` in `run()`, add:

```python
        b["error_message"] = str(exc)[:180]
```

7. In `_estimated_eur`, update rates for Gemini 3.5 Flash-Lite and
consider model-specific defaults; the full patch has the function.

### `app.py`

Inside `on_run`, find the final `return a, b["reply"], status, record["id"]`.
Just before it, add:

```python
    if mode == "api_error_fallback":
        status += f"\n\n⚠️ Gemini error: `{b.get('error_message', 'Unknown error')}`"
```

### `evaluate.py`

Replace `JUDGE_MODEL = ...` with:

```python
JUDGE_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")
```

Replace the `generationConfig` line inside `_judge` with:

```python
"generationConfig": pipeline._generation_config(512, model=JUDGE_MODEL) | {"responseMimeType": "application/json"},
```

In the judge's return dict, pass `model=JUDGE_MODEL` to the cost function:

```python
"tokens": total, "cost_eur": pipeline._estimated_eur(prompt, completion, model=JUDGE_MODEL),
```

### `.env.example`

Replace the line for `GEMINI_API_KEY` with:

```dotenv
GEMINI_API_KEY=    # Required for genuine Gemini agent (set in local .env and Space secrets)
```

Keep this `.env.example` file free of secrets. See complete patch for code
comments and automatic per-model cost defaults.
