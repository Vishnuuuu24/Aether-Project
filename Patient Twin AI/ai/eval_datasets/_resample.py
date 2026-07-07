"""Signal resampling for the PaPaGei-S pretrained-input contract (docs/16 Sprint 10).

PaPaGei-S was pretrained on 125 Hz / 10 s segments; our wrist BVP is 64 Hz. To feed it
in-distribution we resample 64 -> 125 Hz with **polyphase resampling** (`resample_poly`)
— the exact method PaPaGei's own `resample_batch_signal` uses — rather than linear
interpolation, so the frequency content the encoder relies on is preserved faithfully.

Kept tiny and shared so both the PPG-DaLiA and WESAD adapters resample identically.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def resample_poly_to(signal: FloatArray, fs_in: float, fs_out: float) -> FloatArray:
    """Polyphase-resample a 1-D signal from `fs_in` to `fs_out` Hz (PaPaGei's method).

    Reduces the rate ratio to integer up/down factors via a rational approximation, then
    applies `scipy.signal.resample_poly`. A no-op when the rates already match.
    """
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(signal, dtype=np.float64)
    r_in = Fraction(fs_in).limit_denominator()
    r_out = Fraction(fs_out).limit_denominator()
    lcm_den = np.lcm(r_in.denominator, r_out.denominator)
    in_scaled = int(r_in * lcm_den)
    out_scaled = int(r_out * lcm_den)
    g = gcd(in_scaled, out_scaled)
    up = out_scaled // g
    down = in_scaled // g
    resampled = resample_poly(np.asarray(signal, dtype=np.float64), up, down)
    return np.asarray(resampled, dtype=np.float64)


def resample_labels_nearest(labels: np.ndarray, out_len: int) -> np.ndarray:
    """Nearest-neighbour resample a piecewise-constant label stream to `out_len`.

    Used to carry per-sample condition codes onto a resampled signal grid. Within a
    condition span this is exact; only boundary samples are ambiguous, and windowing
    (which requires every sample in a window to share the label) drops those windows.
    """
    n_in = labels.shape[0]
    if n_in == out_len:
        return labels
    idx = np.clip(np.round(np.arange(out_len) * (n_in / out_len)).astype(np.intp), 0, n_in - 1)
    return labels[idx]
