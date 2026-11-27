> Historical starter notes. Current deployment instructions are in SETUP.md and current evaluation instructions are in eval/README.md.

# Critical: AI-generated DEMONSTRATION fixtures, NOT your submission evidence

`cases.csv`, `reference.csv`, `shifted.csv`, and `monitoring/reference_features.csv`
were written by an AI as temporary **fictional** code-test fixtures. The course
explicitly reserves case authorship, reference data, monitoring thresholds,
business decisions and the report's analysis to the **team**. Rewrite these
files from scratch as a group before submission, remove `AI_DEMO_REPLACE`
markers only after replacing each corresponding case, and rerun all commands.

`human_labels.csv` has only a header. **Do not invent 20 labels.** Have at least
20 independent team judgments of *live LLM responses*, before looking at the
LLM judge's labels, then write `id,label,comment` with `PASS` or `FAIL`.

The built-in logistic model is also trained on rule-labeled synthetic examples.
Do not report its output as independently validated machine-learning performance.
Replace its training examples with real, anonymous team-labeled cases and
hold out a separate test set before making deployment decisions.
