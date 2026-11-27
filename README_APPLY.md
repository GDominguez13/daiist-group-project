> Historical starter notes. Current deployment instructions are in SETUP.md and current evaluation instructions are in eval/README.md.

# EXACT APPLICATION GUIDE — ParcelPilot starter patch

Source project: https://github.com/ami232/daiist-group-project

**This ZIP is an OVERLAY, not a fork or a complete copy of the instructor's repo.**
Extract/copy its contents into your own fork's root, preserving folder layout.
Every file here is a complete replacement or new file; **do not paste a block at
an arbitrary line number**. All paths below are relative to the directory
containing the course's original `app.py` and `check.py`.

## 0. First fork / clone the instructor's project

1. Open the GitHub link and click **Fork**. Use your own or your team's fork.
2. From your terminal:

   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/daiist-group-project.git
   cd daiist-group-project
   ```

   If already cloned, run `git status`, commit/stash your work, then continue.
3. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
4. Run `uv sync`. This patch uses only libraries ALREADY present in the original
   `pyproject.toml` and `uv.lock` (NumPy, pandas, Gradio, Hugging Face Hub).

## 1. EXACT paths: replace these existing files completely

| File in your fork | What to do | Purpose |
|---|---|---|
| `pipeline.py` | Replace its **entire** contents with patch `pipeline.py` | Strict input parsing, trained tabular logistic regression, Gemini tool call, optional feedback bandit |
| `app.py` | Replace **entire** contents with patch `app.py` | Two Gradio tabs and unchanged `/run` endpoint and `serve()` |
| `evaluate.py` | Replace **entire** contents with patch `evaluate.py` | Component F1, guardrails, judge, human agreement, baseline assumptions, cost |
| `monitor.py` | Replace **entire** contents with patch `monitor.py` | Time plots, PSI and alert rules |
| `.env.example` | Replace **entire** contents with patch `.env.example` | Optional settings and supported keys |
| `REPORT.md` | Replace **entire** contents with patch `REPORT.md` | CLEARLY MARKED provisional outline and worked example, not ready to submit |
| `eval/cases.csv` | Replace **entire** contents with patch version **for smoke testing only** | 44 AI-generated demonstration cases; YOU MUST replace with 30+ TEAM-WRITTEN cases |
| `eval/reference.csv` | Replace **entire** contents with patch version **for smoke testing only** | 60 fictional reference inputs; YOU MUST replace |
| `eval/shifted.csv` | Replace **entire** contents with patch version **for smoke testing only** | 16 fictional shifted inputs; YOU MUST replace |
| `eval/human_labels.csv` | Replace **entire** contents with header-only patch version | Team must independently label 20+ live outputs |
| `monitoring/reference_features.csv` | Copy generated patch version for demo, **then REGENERATE** | Numeric reference for PSI |

Keep these instructor-supplied files **unchanged**: `main.py`, `check.py`,
`deploy.py`, `logbook.py`, `env.py`, `SETUP.md`, `pyproject.toml`, `uv.lock`,
`.gitignore`, the existing `eval/README.md`, and CI workflows.

New files in this patch: `README_APPLY.md`, `PRESENTATION_PLAN.md` and
`eval/README_DEMO.md`. You may commit them; only the real team report and real
team evaluation should be assessed as evidence.

### A simple exact copy command

If the patch ZIP is in your Downloads folder, unpack it, open a terminal in
the **original fork** and copy each filename/folder from the extracted patch
into this root, overwriting files when asked. On macOS/Linux, if the extracted
folder is next to your clone, run:

```bash
cp -R ../daiist_solution_overlay/. ./
```

The **dot** in `/.` matters: it copies hidden `.env.example` too. On Windows,
use File Explorer to drag the extracted folder's **contents** into the root,
then choose **Replace files**. Do not paste the enclosing folder itself into
an extra `daiist_solution_overlay/` directory.

## 2. Run the safe no-key demonstration

From the repo root (no keys required):

```bash
uv sync
uv run python pipeline.py
uv run python main.py reference
uv run python main.py drift-demo
uv run python main.py evaluate
uv run python check.py
uv run python app.py
```

Open http://127.0.0.1:7860. Paste this exact test input (no personal data):

```text
issue=late_delivery; days_late=4; previous_contacts=1; item_value_eur=35; damaged=0; refund_requested=0
```

Try also a missing-parcel case and an unsupported `issue` from the App examples.
Without a Gemini key, **all drafts are correctly identified as offline
fallback**, and the `review` flag is deliberately on. That is enough to test
wiring, logging and drift; it is NOT a two-AI product evaluation.

`eval/summary.json` and `eval/results.csv` are created by `evaluate.py`; the
summary warns you if the case data and components are still demonstrations.

## 3. Use the actual Gemini agent

Get an **allowed** key: https://aistudio.google.com/apikey. In the **repo root**:

```bash
cp .env.example .env
```

Edit `.env` with your own key (DO NOT paste it here or in GitHub):

```dotenv
GEMINI_API_KEY=PASTE_OWN_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash-lite
```

Ensure `DEMO_MODE` is empty or 0. Restart `uv run python app.py` and run a
case. `Component B` should show `mode = gemini_agent` when its **actual two-call
function-calling cycle** succeeds, and `tools_used = ['lookup_policy']`. If an
API error or quota limit occurs, the output says fallback and requires review.
Do not report those fallback requests as Gemini successes.

The Gemini key is read only from the environment. The agent executes no
payments, writes, or genuine customer actions. The policy database is invented
for demonstration: if a partner company is used in your business scenario,
replace `POLICIES` in `pipeline.py` only with team-verified policy text.

Current dollar prices vary: the starter cost function assumes $0.10 per
million input tokens and $0.40 per million output tokens for paid Gemini
2.5 Flash-Lite; `USD_TO_EUR=0.90` is a **placeholder FX rate**. Update all
three inputs with dated sources for REPORT.md. Free tier bills may differ.

## 4. Evaluate LIVE (after team replaces fixture data)

You must write **real team-owned** inputs in `eval/cases.csv`, at least 30,
including >=5 `hard`, >=5 `review`, >=3 `out_of_scope`, plus normal cases.
`expected` must be independent human annotations of routine/review/out_of_scope.
The code's parser accepts only six aggregate fields; design the test cases
accordingly. Include a realistic baseline (this patch has manual review only).

Also write 30+ independent normal `eval/reference.csv` inputs (not the same as
cases), and 10+ deliberate shifted `eval/shifted.csv` inputs. Rerun:

```bash
uv run python main.py reference
uv run python main.py drift-demo
uv run python main.py evaluate
```

**Important:** `main.py reference` and `main.py drift-demo` currently call the
whole `pipeline.run()` for every case. To avoid unnecessary Gemini charges,
set `DEMO_MODE=1` only while running those **drift-feature calculations**;
features and A's model output remain real, only Gemini's text is skipped:

```bash
# macOS/Linux
DEMO_MODE=1 uv run python main.py reference
DEMO_MODE=1 uv run python main.py drift-demo

# PowerShell
$env:DEMO_MODE='1'; uv run python main.py reference; uv run python main.py drift-demo; Remove-Item Env:DEMO_MODE
```

For end-to-end answer evaluation, **turn DEMO_MODE off**; with an actual key:

```bash
EVAL_USE_LLM_JUDGE=1 uv run python main.py evaluate
```

Above command calls Gemini for every accepted case, plus an extra LLM judge;
it may consume API quota. To measure judge agreement, first have the team read
20+ LIVE drafts and independently enter its own labels, BEFORE seeing the judge:

```csv
id,label,comment
n01,PASS,Specific rationale based on the actual generated reply
n02,FAIL,Specific factual or safety error
```

Those two rows are **syntax examples only**, not real labels. Use actual case
IDs and a minimum of 20 defensible entries. `eval/summary.json` then reports
agreement and confusion counts. The judge is not a substitute for human labels.

## 5. Optional reinforcement-learning bonus

To activate the epsilon-greedy three-prompt-variant bandit, set:

```dotenv
ENABLE_BANDIT=1
```

Generate a real live response using Gemini, click 👍/👎 in the app, then make
more requests. `pipeline.choose_prompt_variant()` reads the actual persisted
logbook feedback on each new request, updates per-variant success estimates,
and uses 10% exploration. Examine `versions.prompt_variant` in the JSON logs.
For the bonus, demonstrate in a real session that feedback actually changes
variant preferences, and document exploration cost and feedback bias. Simulated
traffic alone does **not** earn the bonus. Offline demos are excluded from rewards.

## 6. Deployment and final grading checks

One member creates a public Gradio Hugging Face Space and a **PRIVATE** dataset
for logs. In `.env` (do not commit):

```dotenv
HF_TOKEN=YOUR_WRITE_TOKEN
HF_SPACE=your-user/your-space
HF_LOG_REPO=your-user/your-private-logs-dataset
GEMINI_API_KEY=YOUR_KEY
```

Then:

```bash
uv run python deploy.py --secrets
uv run python check.py --space https://huggingface.co/spaces/your-user/your-space
```

The instructor's `deploy.py` exports dependencies from the unmodified lockfile,
and deploys these new Python, Markdown and CSV files. Do not edit files directly
in the Space. The private HF dataset is required for **persistent** monitoring
across Space restarts.

Before submitting on **Monday 30 November 2026 at 23:59**, the team must:

- replace AI demo cases and reference/shifted data with team-authored cases;
- supply 20+ team labels for live free-text outputs and check judge agreement;
- replace synthetic training with independently labeled, genuinely predictive
  data or rigorously explain/validate its limitation;
- justify its own product, metrics, thresholds, review rule, euro assumptions
  and decision in `REPORT.md` using real outputs, not the draft;
- document names, commits and contributions in `CONTRIBUTIONS.md`;
- verify the public Space plus private durable logs, rehearse a 6-minute live
  demo, rerun `uv run python check.py`, and tag `final` only when ready;
- disclose generative AI use for coding and drafting in the final report.

Then (once per group):

```bash
git add .
git status
git commit -m "Implement ParcelPilot two-stage prototype and evaluation"
git push
git tag final
git push --tags
```

Open a pull request from your fork to the original `ami232/daiist-group-project`
repository and submit the PR link, Space URL and backup ZIP on Blackboard.

## 7. What has and has NOT been verified

**Verified locally** (environment with Python 3.13, NumPy, pandas and Gradio):
Python syntax, offline pipeline, numerical model fitting, 44 synthetic inputs,
PSI alert silent on synthetic normal inputs and firing on synthetically shifted
inputs. The exact original teacher checker and actual Gemini API/hub deployment
MUST be rerun with your own keys/machine. They cannot be verified in this
isolated build environment. The bundle makes no claim to a passing course grade.

### Replacing synthetic ML training without changing Python

There is an additional `data/README.md` in the patch. Create your own
`data/train.csv` with **six non-PII input columns + `requires_review`**.
Once you have >=40 independent team-labeled rows (ideally >=100), next run
will automatically fit the logistic model on that table instead of synthesized
rules. Keep all evaluation cases separate from those training rows. The model
version changes to `team-labeled-tabular-logreg-v1` in your request logs.
