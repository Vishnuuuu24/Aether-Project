"""Fetch the auto-downloadable datasets listed in docs/13_Datasets.md.

Deliberately uses `huggingface_hub` (already a project dependency) instead of the
`datasets` library: this repo has a top-level `datasets/` folder that shadows the
`datasets` pip package as a namespace import when Python runs with the repo root
on `sys.path` (e.g. `python -c` or `python -m` from the repo root). Run this file
directly (`python scripts/fetch_datasets.py ...`) so its own directory, not the
repo root, is `sys.path[0]`.

Human-required datasets (MIMIC-IV Notes, SNOMED/UMLS/LOINC/RxNorm, MESA/SHHS,
WESAD's stale origin, the unconfirmed Stanford Long-COVID dataset) are
intentionally NOT in this script — see the per-dataset READMEs under `datasets/`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"

# (folder name, HF dataset repo id, optional allow-patterns to limit what's pulled)
HF_DATASETS: dict[str, tuple[str, list[str] | None]] = {
    "guidelines": ("epfl-llm/guidelines", None),
    "medrag-textbooks": ("MedRAG/textbooks", None),
    "medqa": ("GBaker/MedQA-USMLE-4-options", None),
    "medmcqa": ("openlifescienceai/medmcqa", None),
    "pubmedqa": ("qiaojin/PubMedQA", None),
    "nfcorpus": ("BeIR/nfcorpus", None),
    "pmc-patients": ("zhengyun21/PMC-Patients", None),
    "augmented-clinical-notes": ("AGBonnet/augmented-clinical-notes", None),
    "mmlu-clinical": (
        "cais/mmlu",
        [
            "anatomy/*",
            "clinical_knowledge/*",
            "college_medicine/*",
            "medical_genetics/*",
            "professional_medicine/*",
        ],
    ),
}

# folder name -> destination sub-directory under datasets/
HF_DEST = {
    "guidelines": "guidelines",
    "medrag-textbooks": "MedRAG-textbooks",
    "medqa": "MedQA-USMLE",
    "medmcqa": "MedMCQA",
    "pubmedqa": "PubMedQA",
    "nfcorpus": "nfcorpus",
    "pmc-patients": "PMC-Patients",
    "augmented-clinical-notes": "augmented-clinical-notes",
    "mmlu-clinical": "MMLU-clinical",
}

ICD10CM_URL = "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip"
MIRAGE_REPO = "https://github.com/Teddy-XiongGZ/MIRAGE.git"
# Single-zip endpoint (confirmed via HEAD: 8.1GB, ODC-BY license) — avoids depending
# on `wget -r` (not installed on this Mac) to mirror the per-record file listing.
SLEEP_EDF_ZIP_URL = "https://physionet.org/content/sleep-edfx/get-zip/1.0.0/"

ALL_KEYS = [*HF_DATASETS.keys(), "icd10cm", "mirage", "sleep-edf"]


def fetch_hf(key: str) -> None:
    from huggingface_hub import snapshot_download

    repo_id, allow_patterns = HF_DATASETS[key]
    dest = DATASETS_DIR / HF_DEST[key]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{key}] downloading {repo_id} -> {dest}", flush=True)
    # Never let the HF repo's own README.md land here — this repo keeps a
    # hand-written, git-tracked README.md per dataset folder (provenance/license/
    # fetch instructions); the raw dataset card would silently clobber it.
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        ignore_patterns=["README.md", ".gitattributes"],
    )
    if key == "nfcorpus":
        _fetch_nfcorpus_qrels(dest)
    print(f"[{key}] done", flush=True)


def _fetch_nfcorpus_qrels(dest: Path) -> None:
    """BeIR/nfcorpus ships only corpus+queries; its human relevance judgements live in
    the sibling `BeIR/nfcorpus-qrels` repo. The Sprint 11 reranker training + IR eval
    need these graded qrels, so fetch train/dev/test into `<dest>/qrels/`."""
    from huggingface_hub import hf_hub_download

    qrels_dir = dest / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        src = hf_hub_download(
            "BeIR/nfcorpus-qrels", f"{split}.tsv", repo_type="dataset"
        )
        shutil.copyfile(src, qrels_dir / f"{split}.tsv")
    print("[nfcorpus] qrels (train/dev/test) fetched", flush=True)


def fetch_icd10cm() -> None:
    dest = DATASETS_DIR / "ICD-10-CM"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "icd10cm_order.zip"
    print(f"[icd10cm] downloading {ICD10CM_URL}", flush=True)
    urllib.request.urlretrieve(ICD10CM_URL, zip_path)  # noqa: S310 (fixed https CMS URL)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print("[icd10cm] done", flush=True)


def fetch_mirage() -> None:
    # Clone into a `repo/` subdir, not the dataset folder root — the upstream
    # repo has its own README.md that would otherwise clobber ours.
    dest = DATASETS_DIR / "MIRAGE" / "repo"
    if (dest / ".git").exists():
        print("[mirage] already cloned, skipping", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mirage] cloning {MIRAGE_REPO}", flush=True)
    subprocess.run(  # noqa: S603
        ["git", "clone", "--depth", "1", MIRAGE_REPO, str(dest)],
        check=True,
    )
    print("[mirage] done", flush=True)


def fetch_sleep_edf() -> None:
    dest = DATASETS_DIR / "Sleep-EDF"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "sleep-edf-database-expanded-1.0.0.zip"
    if not zip_path.exists():
        print(
            f"[sleep-edf] downloading {SLEEP_EDF_ZIP_URL} (8.1GB, this takes a while)",
            flush=True,
        )
        subprocess.run(  # noqa: S603
            ["curl", "-L", "--fail", "-o", str(zip_path), SLEEP_EDF_ZIP_URL],
            check=True,
        )
    print("[sleep-edf] extracting...", flush=True)
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(dest)], check=True)  # noqa: S603, S607
    print("[sleep-edf] done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=ALL_KEYS,
        help="Dataset key to fetch (repeatable). Default: all.",
    )
    args = parser.parse_args()
    keys = args.dataset or ALL_KEYS

    for key in keys:
        if key == "icd10cm":
            fetch_icd10cm()
        elif key == "mirage":
            fetch_mirage()
        elif key == "sleep-edf":
            fetch_sleep_edf()
        else:
            fetch_hf(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
