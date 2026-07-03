"""Run the aggregated evaluation report and print it (docs/11 §4; T5.2).

    python -m scripts.run_eval            # human-readable
    python -m scripts.run_eval --json     # machine-readable, for storage

Every §1 metric produces a number; sections are labelled by the dataset they ran
on (`synthetic` = harness smoke, not a clinical result), and unresolved coverage
is listed under GAPS with its blocker. Numbers here are reproducible because the
harnesses are deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from ai.eval_report import build_report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation harnesses and report.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    report = build_report()

    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"Eval report  (versions: {report.versions})")
    print(f"generated_at: {report.generated_at}\n")
    for section in report.sections:
        print(f"[{section.name}]  dataset={section.dataset}")
        for metric, value in section.metrics.items():
            print(f"    {metric:>24}: {value:.4f}")
        print()
    print("GAPS (numbers not yet on real offline data / blocked):")
    for gap in report.gaps:
        print(f"  - {gap.metric}  [{gap.blocker.value}]")
        print(f"      {gap.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
