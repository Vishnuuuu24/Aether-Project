"""Sprint 9 — training harness foundation tests (docs/16).

Components that need neither MLX nor a dataset are tested unconditionally (config +
seed determinism, backend selection + CUDA refusal, checkpoint write/version/reload,
data shaping, eval-hook scoring). The actual training smoke is skip-guarded on MLX
(and on the PPG-DaLiA dataset for the real-data variant), mirroring the DB/Qdrant
and WESAD skip-guards elsewhere in the suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ai.eval_datasets.ppg_dalia import (
    FEATURE_NAMES,
    PpgDaliaLayoutError,
    load_ppg_dalia_hr_windows,
    ppg_dalia_available,
)
from ai.training.backends import (
    CudaQloraBackend,
    MlxBackend,
    TrainBackend,
    TrainBackendUnavailable,
    TrainedHead,
    select_backend,
)
from ai.training.checkpoints import (
    load_head,
    register_checkpoint_version,
    write_checkpoint,
)
from ai.training.config import DEFAULT_SEED, TrainConfig, set_global_seed
from ai.training.eval_hook import score_head
from ai.training.smoke import run_smoke, synthetic_hr_windows

_MLX = importlib.util.find_spec("mlx") is not None
_PPG_DALIA_ROOT = Path("datasets/PPG-DaLiA")
requires_mlx = pytest.mark.skipif(not _MLX, reason="MLX not installed (Apple-silicon only)")


# --- config + seeding -------------------------------------------------------------


def test_train_config_validates() -> None:
    with pytest.raises(ValueError):
        TrainConfig(val_fraction=0.0)
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=0.0)


def test_set_global_seed_is_deterministic() -> None:
    set_global_seed(DEFAULT_SEED)
    a = np.random.random(5)
    set_global_seed(DEFAULT_SEED)
    b = np.random.random(5)
    assert np.array_equal(a, b)


# --- backend seam -----------------------------------------------------------------


def test_select_backend_resolves_known_and_rejects_unknown() -> None:
    assert isinstance(select_backend("mlx"), MlxBackend)
    assert isinstance(select_backend("cuda_qlora"), CudaQloraBackend)
    assert isinstance(select_backend("mlx"), TrainBackend)
    with pytest.raises(ValueError):
        select_backend("tensorflow")


def test_cuda_backend_refuses_without_cuda() -> None:
    # No CUDA on the Mac → a clear refusal, not a silent degrade.
    with pytest.raises(TrainBackendUnavailable):
        CudaQloraBackend().fit(
            np.zeros((4, 2)), np.zeros(4), TrainConfig(backend="cuda_qlora", epochs=1)
        )


# --- checkpoint writer + version registry ----------------------------------------


def _dummy_head() -> TrainedHead:
    return TrainedHead(
        weights=np.array([1.0, -2.0]),
        bias=0.5,
        feature_mean=np.array([0.0, 0.0]),
        feature_std=np.array([1.0, 1.0]),
        feature_names=("a", "b"),
        backend="mlx",
    )


def test_write_checkpoint_is_content_addressed_and_reloadable(tmp_path: Path) -> None:
    cfg = TrainConfig(epochs=10)
    prov = {"dataset": "unit", "n_windows": 4}
    h1 = write_checkpoint(_dummy_head(), name="t", config=cfg, provenance=prov, root=tmp_path)
    h2 = write_checkpoint(_dummy_head(), name="t", config=cfg, provenance=prov, root=tmp_path)
    # Deterministic id: same run identity → same version (created_at excluded).
    assert h1.version == h2.version
    assert h1.version.startswith("t@")
    assert (h1.path / "head.npz").exists() and (h1.path / "manifest.json").exists()

    reloaded = load_head(h1)
    assert reloaded.feature_names == ("a", "b")
    assert reloaded.bias == 0.5
    assert np.array_equal(reloaded.weights, np.array([1.0, -2.0]))


def test_checkpoint_version_differs_by_config(tmp_path: Path) -> None:
    prov = {"dataset": "unit"}
    a = write_checkpoint(_dummy_head(), name="t", config=TrainConfig(seed=1),
                         provenance=prov, root=tmp_path)
    b = write_checkpoint(_dummy_head(), name="t", config=TrainConfig(seed=2),
                         provenance=prov, root=tmp_path)
    assert a.version != b.version


def test_register_checkpoint_version_stamps_without_mutating() -> None:
    from core.versioning import VersionRegistry

    base = VersionRegistry.from_env()
    stamped = register_checkpoint_version("enc@abc123", base=base)
    assert stamped.current().baseline_engine == "enc@abc123"
    # Base registry is untouched (human-gated release model, no closed loop).
    assert base.current().baseline_engine != "enc@abc123"


# --- data shaping + eval hook -----------------------------------------------------


def test_synthetic_windows_shape_and_names() -> None:
    w = synthetic_hr_windows(n=50, seed=3)
    assert w.features.shape == (50, len(FEATURE_NAMES))
    assert w.targets.shape == (50,)
    assert w.feature_names == FEATURE_NAMES


def test_score_head_reports_error_metrics() -> None:
    head = _dummy_head()
    feats = np.array([[1.0, 1.0], [2.0, 2.0]])
    # pred = z·w + b with z standardised by baked-in (mean=0,std=1): [1-2+0.5, 2-4+0.5]
    targets = head.predict(feats)  # zero-error case
    m = score_head(head, feats, targets)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["n"] == 2.0


# --- PPG-DaLiA loader (validation is dataset-free; real load is skip-guarded) ------


def test_ppg_dalia_missing_root_raises() -> None:
    assert not ppg_dalia_available(Path("datasets/does-not-exist"))
    with pytest.raises(PpgDaliaLayoutError):
        load_ppg_dalia_hr_windows(Path("datasets/does-not-exist"))


@pytest.mark.skipif(not ppg_dalia_available(_PPG_DALIA_ROOT), reason="PPG-DaLiA not on disk")
def test_ppg_dalia_real_load_shapes() -> None:
    w = load_ppg_dalia_hr_windows(_PPG_DALIA_ROOT, max_subjects=1, max_windows_per_subject=50)
    assert len(w) > 0
    assert w.features.shape[1] == len(FEATURE_NAMES)
    assert w.features.shape[0] == w.targets.shape[0]
    assert np.all(w.targets > 0.0)  # GT HR is positive


# --- end-to-end smoke (the Sprint 9 DoD; needs MLX) -------------------------------


@requires_mlx
def test_smoke_end_to_end_on_synthetic(tmp_path: Path) -> None:
    result = run_smoke(
        synthetic_hr_windows(n=300, seed=0),
        config=TrainConfig(epochs=300, learning_rate=0.1),
        checkpoint_root=tmp_path,
    )
    assert result.n_train + result.n_val == 300
    assert (result.handle.path / "manifest.json").exists()
    assert result.version.startswith("linear-hr-smoke@")
    # A linear head must fit a linear target well below the raw-target spread.
    assert result.metrics["mae"] < 5.0


@requires_mlx
@pytest.mark.skipif(not ppg_dalia_available(_PPG_DALIA_ROOT), reason="PPG-DaLiA not on disk")
def test_smoke_end_to_end_on_ppg_dalia(tmp_path: Path) -> None:
    windows = load_ppg_dalia_hr_windows(
        _PPG_DALIA_ROOT, max_subjects=1, max_windows_per_subject=400
    )
    result = run_smoke(windows, checkpoint_root=tmp_path)
    assert result.metrics["n"] > 0
    assert (result.handle.path / "head.npz").exists()
