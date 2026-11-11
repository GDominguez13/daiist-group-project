# Replace synthetic training with independently team-labeled tabular data

The starter trains its numeric classifier on synthetic data. For genuine team
training, create `data/train.csv` (not `train.example.csv`) with 100+ distinct
anonymous cases if possible. Required columns (order not important):

```csv
issue,days_late,previous_contacts,item_value_eur,damaged,refund_requested,requires_review
```

`requires_review=1` means independently adjudicated staff review, `0` routine.
The inputs must follow the six strict parser fields and numeric limits. It is
valid to have more than 100 rows, but never include any PII. Have team members
label cases using a documented independent review rubric. Do not copy rows from
`eval/cases.csv` or `eval/reference.csv`: they are **held-out** evaluation or
drift data, not training data. Save an additional held-out tabular validation
split to measure calibration, false negative review rate and uncertainty.

The model checks for `data/train.csv` when `pipeline.py` starts, switches to
`team-labeled-tabular-logreg-v1` and trains on that file automatically. If file
contains <40 rows or labels/columns are invalid, it raises a descriptive
error instead of silently replacing your data with the demo synthetic rows.
Keep team labeling decisions and provenance in REPORT.md; don't claim that an
AI generated them.
