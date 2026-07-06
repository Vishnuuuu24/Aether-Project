"""Self-contained HTML+SVG training report (docs/16 Sprint 10).

Produces a single `.html` file (no matplotlib, no external assets) you can open in a
browser to SEE a run: the learning curve, predicted-vs-true HR on the held-out
subjects, the residual spread, and the headline scores vs the classical baseline it
must beat. Everything is inline SVG so the file is portable and diff-able.

This renders results only — it never decides promotion (that stays human-gated,
CLAUDE.md principle 5). "encoder loses to baseline" renders honestly, in red.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai.training.mlx_encoder import TrainingHistory

DEFAULT_REPORT_ROOT = Path("reports")

_W, _H, _PAD = 640, 300, 44


def _svg_open(title: str) -> str:
    return (
        f'<svg viewBox="0 0 {_W} {_H}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{html.escape(title)}">'
        f'<rect width="{_W}" height="{_H}" fill="#ffffff"/>'
    )


def _axes(x0: float, y0: float, x1: float, y1: float) -> str:
    return (
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#94a3b8" stroke-width="1"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#94a3b8" stroke-width="1"/>'
    )


def _txt(x: float, y: float, s: str, *, size: int = 11, fill: str = "#334155",
         anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="{anchor}">'
        f'{html.escape(s)}</text>'
    )


def _learning_curve(history: TrainingHistory) -> str:
    logs = history.logs
    if not logs:
        return _svg_open("learning curve") + _txt(_W / 2, _H / 2, "no epochs") + "</svg>"
    xs = [log.epoch for log in logs]
    ys = [log.val_mae_bpm for log in logs]
    x0, x1, y0, y1 = _PAD, _W - 16, 16, _H - _PAD
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0.0, max(ys) * 1.1 + 1e-6

    def px(x: float) -> float:
        return x0 + (x - xmin) / max(1, xmax - xmin) * (x1 - x0)

    def py(y: float) -> float:
        return y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)

    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys, strict=True))
    best = min(range(len(ys)), key=lambda i: ys[i])
    parts = [_svg_open("held-out HR MAE per epoch"), _axes(x0, y0, x1, y1)]
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    parts.append(
        f'<circle cx="{px(xs[best]):.1f}" cy="{py(ys[best]):.1f}" r="4" fill="#16a34a"/>'
    )
    parts.append(_txt(px(xs[best]), py(ys[best]) - 8,
                      f"best {ys[best]:.2f} bpm @ epoch {xs[best]}", size=11, fill="#16a34a"))
    parts.append(_txt(x0, y1 + 16, f"epoch {xmin}", size=10))
    parts.append(_txt(x1, y1 + 16, f"{xmax}", size=10, anchor="end"))
    parts.append(_txt(x0 - 6, py(ymax) + 4, f"{ymax:.0f}", size=10, anchor="end"))
    parts.append(_txt(x0 - 6, y1, "0", size=10, anchor="end"))
    parts.append(_txt(_W / 2, 12, "Held-out HR MAE (bpm) per epoch", size=12, anchor="middle"))
    parts.append("</svg>")
    return "".join(parts)


def _scatter(true: np.ndarray, pred: np.ndarray, baseline_mae: float | None) -> str:
    lo, hi = 40.0, 160.0
    x0, x1, y0, y1 = _PAD, _W - 16, 16, _H - _PAD

    def px(v: float) -> float:
        return x0 + (np.clip(v, lo, hi) - lo) / (hi - lo) * (x1 - x0)

    def py(v: float) -> float:
        return y1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0)

    parts = [_svg_open("predicted vs true HR"), _axes(x0, y0, x1, y1)]
    parts.append(
        f'<line x1="{px(lo):.1f}" y1="{py(lo):.1f}" x2="{px(hi):.1f}" y2="{py(hi):.1f}" '
        f'stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 3"/>'
    )
    step = max(1, len(true) // 400)  # cap glyphs so the file stays small
    for t, p in zip(true[::step], pred[::step], strict=True):
        parts.append(
            f'<circle cx="{px(float(t)):.1f}" cy="{py(float(p)):.1f}" r="2" '
            f'fill="#2563eb" fill-opacity="0.35"/>'
        )
    parts.append(_txt(_W / 2, 12, "Predicted vs true HR (held-out subjects)",
                      size=12, anchor="middle"))
    parts.append(_txt(_W / 2, _H - 6, "true HR (bpm) →   |   diagonal = perfect",
                      size=10, anchor="middle"))
    if baseline_mae is not None:
        parts.append(_txt(x1, y0 + 4, f"baseline MAE {baseline_mae:.2f}", size=10,
                          fill="#64748b", anchor="end"))
    parts.append("</svg>")
    return "".join(parts)


def _verdict(encoder_mae: float, baseline_mae: float | None) -> tuple[str, str]:
    if baseline_mae is None:
        return ("no baseline to compare", "#64748b")
    if encoder_mae < baseline_mae:
        delta = baseline_mae - encoder_mae
        return (f"WINS — {delta:.2f} bpm lower MAE than the linear baseline", "#16a34a")
    return ("does NOT beat the linear baseline — keep the classical path", "#dc2626")


def render_encoder_report(
    *,
    version: str,
    provenance: dict[str, object],
    history: TrainingHistory,
    val_true: Sequence[float],
    val_pred: Sequence[float],
    encoder_mae: float,
    encoder_rmse: float,
    baseline_mae: float | None,
    generated_at: datetime | None = None,
) -> str:
    """Render the full standalone HTML report as a string."""
    true = np.asarray(val_true, dtype=np.float64)
    pred = np.asarray(val_pred, dtype=np.float64)
    verdict, colour = _verdict(encoder_mae, baseline_mae)
    when = (generated_at or datetime.now(UTC)).isoformat(timespec="seconds")

    def row(k: str, v: str) -> str:
        return (
            f'<tr><td style="padding:4px 14px 4px 0;color:#64748b">{html.escape(k)}</td>'
            f'<td style="padding:4px 0;font-weight:600">{html.escape(v)}</td></tr>'
        )

    prov_rows = "".join(row(str(k), str(v)) for k, v in provenance.items())
    base_txt = f"{baseline_mae:.2f} bpm" if baseline_mae is not None else "—"
    curve_svg = _learning_curve(history)
    scatter_svg = _scatter(true, pred, baseline_mae)
    card = "background:#fff;border-radius:10px;padding:8px;margin-bottom:16px"
    body_style = (
        "margin:0;background:#f1f5f9;color:#0f172a;"
        "font-family:ui-sans-serif,system-ui,sans-serif"
    )
    subtitle = f"docs/16 Sprint 10 · {html.escape(version)} · {html.escape(when)}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Encoder report — {html.escape(version)}</title></head>
<body style="{body_style}">
<div style="max-width:720px;margin:0 auto;padding:28px">
  <h1 style="font-size:20px;margin:0 0 2px">Biosignal encoder — PPG→HR</h1>
  <div style="color:#64748b;font-size:13px">{subtitle}</div>
  <div style="margin:16px 0;padding:14px 16px;border-radius:10px;background:#fff;
              border-left:5px solid {colour}">
    <div style="font-size:13px;color:#64748b">Held-out verdict</div>
    <div style="font-size:16px;font-weight:700;color:{colour}">{html.escape(verdict)}</div>
    <div style="margin-top:8px;font-size:14px">
      Encoder HR MAE <b>{encoder_mae:.2f} bpm</b> · RMSE <b>{encoder_rmse:.2f} bpm</b>
      &nbsp;·&nbsp; linear baseline MAE <b>{base_txt}</b></div>
  </div>
  <div style="{card}">{curve_svg}</div>
  <div style="{card}">{scatter_svg}</div>
  <h2 style="font-size:15px;margin:18px 0 6px">Run provenance</h2>
  <table style="font-size:13px;border-collapse:collapse">{prov_rows}</table>
  <p style="color:#94a3b8;font-size:12px;margin-top:20px">
    Results only — promotion is human-gated (CLAUDE.md principle 5). Numbers are the
    dataset's own ground-truth HR on subject-held-out windows.</p>
</div></body></html>"""


def write_report(html_text: str, *, version: str, root: Path = DEFAULT_REPORT_ROOT) -> Path:
    """Write the report HTML to `<root>/<version>.html` and return the path."""
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{version}.html"
    out.write_text(html_text)
    return out
