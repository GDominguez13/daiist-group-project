# Contributions

*Fill this in as you go, not on the last day. It is checked against the commits and against the peer assessment. A first name and a GitHub username are enough: this repo is public.*

## Members

| First name | GitHub username | Main areas |
|---|---|---|
| Alejandra | alegova210 | Evaluation cases, human labels, drift reference/shifted data, Gemini retries, LLM-judge analysis |
| | | |
| | | |
| | | |
| | | |

## Log

*One line per piece of work: the date, who did it (more than one name is fine), and what. Code done in pair programming counts for both, even if only one of you committed it.*

| Date | Who | What |
|---|---|---|
| 2026-11-19 | Alejandra | Wrote 46 evaluation cases in eval/cases.csv (24 normal, 8 hard, 7 review, 7 out_of_scope) replacing the demo cases |
| 2026-11-19 | Alejandra | First live evaluation with Gemini; found 17/39 provider failures (HTTP 503 and timeouts) |
| 2026-11-19 | Alejandra | Added retries with backoff for transient Gemini errors in pipeline.py, plus 3 unit tests (with AI assistance) |
| 2026-11-21 | Alejandra | Second live evaluation: 33/39 live replies; classifier misses all value/delay/damage+refund boundaries in hard cases |
| 2026-11-21 | Alejandra | New eval/reference.csv (36 normal inputs) and eval/shifted.csv (Christmas-peak scenario), both sampled by script from our normal cases; drift demo quiet on normal, fires on shifted |
| 2026-11-21 | Alejandra | Labelled 33 live Gemini replies PASS/FAIL in eval/human_labels.csv (6 PASS, 27 FAIL) |
| 2026-11-21 | Alejandra | Added `evaluate.py --judge-only` to judge the already-labelled replies; judge agreement 50% (too lenient: passes 14 of 22 human FAILs) |
| 2026-11-23 | Emilia | Grouped the 27 FAIL human labels into 4 root causes: staff review promised in routine cases, no next step for the customer, internal jargon, misread facts (delay, previous contacts, wrong review reason) |
| 2026-11-23 | Emilia | Rewrote the drafting prompt (v2) in pipeline.py: plain-language case facts, review reasons that match the routing rule, separate instructions for routine vs review, customer next steps in the policies (with AI assistance) |
| 2026-11-23 | Emilia | Added `quality_flags` to component B (internal_jargon, staff_promise_in_routine, missing_handoff); on the old replies they catch 13 of 27 human FAILs with 0 false alarms |
