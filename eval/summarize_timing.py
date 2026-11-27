"""Summarize the team's paired staff-work observations in seconds."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    with args.csv_file.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        parser.error("Record at least one paired observation first.")
    differences = []
    seen = set()
    for line, row in enumerate(rows, start=2):
        try:
            case = row["case_id"].strip()
            manual = float(row["manual_seconds"])
            assisted = float(row["assisted_seconds"])
        except (KeyError, TypeError, ValueError, AttributeError):
            parser.error(f"Invalid or missing fields on line {line}.")
        if not case or case in seen:
            parser.error(f"Missing or duplicate case ID on line {line}.")
        if not all(math.isfinite(t) and t >= 0 for t in (manual, assisted)):
            parser.error(f"Expected finite nonnegative times on line {line}.")
        seen.add(case)
        differences.append(manual - assisted)
    print(f"Paired cases: {len(differences)}")
    print(f"Mean staff seconds saved per observed case: {mean(differences):.2f}")
    print(f"Total observed staff minutes saved: {sum(differences) / 60:.2f}")


if __name__ == "__main__":
    main()
