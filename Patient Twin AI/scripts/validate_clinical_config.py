"""Validate the clinician-filled clinical config stubs (T8.3).

    python -m scripts.validate_clinical_config          # human-readable
    python -m scripts.validate_clinical_config --json    # machine-readable

Exit code 0 when every present config is well-formed (all-unset stubs included);
exit code 2 with a precise message when a value is present but malformed. Meant for
CI and pre-deploy gating so a clinical-config mistake fails loud, never silent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from core.clinical_config import ClinicalConfigError, validate_clinical_configs


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate clinical config stubs.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--dir", type=Path, default=None, help="config/clinical dir override")
    args = parser.parse_args(argv)

    try:
        report = validate_clinical_configs(args.dir)
    except ClinicalConfigError as exc:
        if args.json:
            json.dump({"ok": False, "error": str(exc)}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"MALFORMED: {exc}", file=sys.stderr)
        return 2

    if args.json:
        json.dump(
            {
                "ok": True,
                "all_unset": report.all_unset,
                "sections": [asdict(s) for s in report.sections],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    print("Clinical config validation: OK")
    for section in report.sections:
        state = (
            "missing"
            if not section.present
            else (f"{section.n_set} set" if section.n_set else "UNSET")
        )
        print(f"  {section.name:>20}: {state}")
    if report.all_unset:
        print("\nAll sections are unset stubs (fail-safe) — awaiting clinical input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
