"""Subject-held-out splitting (docs/16 Sprint 10 — "no subject leakage").

A HR model that sees any window from subject S in training must not be scored on
another window from S: wrist-PPG windows within a subject are highly correlated, so
a random split leaks and inflates the score. We split on WHOLE subjects instead —
the honest test of "does this generalise to a person it has never seen?".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def subject_held_out_split(
    subject_ids: Sequence[str],
    *,
    val_subjects: Sequence[str] | None = None,
    val_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Partition window indices so no subject appears in both train and val.

    Returns `(train_idx, val_idx, train_subjects, val_subjects)`. If `val_subjects`
    is given it is used verbatim; otherwise a seeded shuffle picks a `val_fraction`
    share of the distinct subjects (at least one) as the held-out set.
    """
    ids = np.asarray(subject_ids)
    # distinct subjects in first-seen order (stable, provenance-friendly)
    seen: dict[str, None] = {}
    for sid in subject_ids:
        seen.setdefault(sid, None)
    unique = list(seen)
    if len(unique) < 2 and val_subjects is None:
        raise ValueError(
            f"subject-held-out split needs >= 2 distinct subjects, got {len(unique)}"
        )

    if val_subjects is None:
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(len(unique)))
        n_val = max(1, int(round(len(unique) * val_fraction)))
        n_val = min(n_val, len(unique) - 1)  # always leave >= 1 train subject
        val_set = {unique[i] for i in order[:n_val]}
    else:
        val_set = set(val_subjects)
        unknown = val_set - set(unique)
        if unknown:
            raise ValueError(f"val_subjects not present in data: {sorted(unknown)}")
        if not (set(unique) - val_set):
            raise ValueError("val_subjects covers every subject; no training data left")

    val_mask = np.array([sid in val_set for sid in ids], dtype=bool)
    val_idx = np.nonzero(val_mask)[0]
    train_idx = np.nonzero(~val_mask)[0]
    train_subjects = tuple(s for s in unique if s not in val_set)
    val_subjects_out = tuple(s for s in unique if s in val_set)
    return train_idx, val_idx, train_subjects, val_subjects_out
