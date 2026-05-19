"""Missing-modality synthesis for the cleaned cohort.

Some cases arrive with one or more of {t1, t1ce, t2, flair} absent. The
preprocessing pipeline already keeps those cases with the missing
channel zero-filled — this module replaces those zero channels with a
plausible synthesis driven by the donor pool of cases that *do* carry
the full 4-modality set.

We deliberately avoid a GPU-trained CNN at this stage and ship three
zero-training strategies that map directly onto the radiomics literature:

    * "atlas_mean"        — replace the missing channel with the mean of
                            that channel across donor cases, after
                            grid-aligning each donor to the target.
                            Baseline. Smooth, blurry, but unbiased.

    * "linear_regression" — fit a *voxel-wise* linear model that
                            predicts every donor's missing modality from
                            its observed modalities, then apply it to
                            the target. Captures cross-modality
                            correlation; far sharper than atlas_mean.

    * "patch_knn"         — for each voxel position, look up the donor
                            voxel whose observed-modality intensity tuple
                            is closest to the target's, and copy that
                            donor's missing-modality value. The most
                            faithful of the three for sharp boundaries
                            but the most expensive; defaults to a small
                            k and a subsample of donors.

A learned-CNN hook (``LearnedSynth``) is also defined so a model trained
later can be dropped in without touching the calling code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CANONICAL_MODALITIES = ("t1", "t1ce", "t2", "flair")
DEFAULT_CHANNEL_INDEX = {m: i for i, m in enumerate(CANONICAL_MODALITIES)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_missing_channels(image: np.ndarray, eps: float = 1e-6) -> List[str]:
    """A channel is treated as missing when it carries no non-zero voxel.

    The cleaning pipeline guarantees missing channels are exact zero, so
    a simple max-abs check is sufficient.
    """
    out: List[str] = []
    for m, idx in DEFAULT_CHANNEL_INDEX.items():
        if image[idx].size == 0 or float(np.abs(image[idx]).max()) < eps:
            out.append(m)
    return out


def _resize_to(arr: np.ndarray, target_shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    from scipy.ndimage import zoom

    if tuple(arr.shape) == tuple(target_shape):
        return arr
    factors = [t / s for t, s in zip(target_shape, arr.shape)]
    return zoom(arr, factors, order=order)


def _brain_mask_from_image(image: np.ndarray) -> np.ndarray:
    return np.any(image != 0, axis=0)


# ---------------------------------------------------------------------------
# Donor pool
# ---------------------------------------------------------------------------

@dataclass
class DonorPool:
    """A lightweight donor index over a folder of complete cleaned ``.npz``
    cases. Donor volumes are lazily loaded and resized to a target shape
    on demand."""
    paths: List[str] = field(default_factory=list)
    cache_grid: Optional[Tuple[int, int, int]] = None
    cache_images: Optional[np.ndarray] = None  # (n_donors, 4, H, W, D)
    cache_brain_masks: Optional[np.ndarray] = None  # (n_donors, H, W, D)

    @classmethod
    def from_dir(cls, donor_dir: str, max_donors: Optional[int] = None) -> "DonorPool":
        paths = sorted(str(p) for p in Path(donor_dir).glob("*.npz"))
        if max_donors is not None:
            paths = paths[:max_donors]
        pool = cls(paths=paths)
        pool._validate()
        return pool

    def _validate(self) -> None:
        if not self.paths:
            raise FileNotFoundError("Donor pool is empty.")

    def _is_complete(self, image: np.ndarray) -> bool:
        return not detect_missing_channels(image)

    def materialise_on_grid(self, target_shape: Tuple[int, int, int],
                            max_donors: Optional[int] = None) -> None:
        """Load all donor images, drop incomplete ones, resize to the
        target's grid, cache. Idempotent for a given grid."""
        if (self.cache_grid == tuple(target_shape)
                and self.cache_images is not None):
            return

        complete: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for p in self.paths:
            try:
                d = np.load(p)
                img = d["image"]
            except Exception:
                continue
            if not self._is_complete(img):
                continue
            resized = np.stack(
                [_resize_to(img[c], target_shape, order=1) for c in range(img.shape[0])],
                axis=0,
            ).astype(np.float32)
            complete.append(resized)
            masks.append(_brain_mask_from_image(resized))
            if max_donors is not None and len(complete) >= max_donors:
                break
        if not complete:
            raise RuntimeError("No complete donor cases found in the pool.")
        self.cache_grid = tuple(target_shape)
        self.cache_images = np.stack(complete, axis=0)
        self.cache_brain_masks = np.stack(masks, axis=0)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@dataclass
class SynthesisReport:
    case_id: str
    strategy: str
    missing_channels: List[str]
    donors_used: int
    notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "strategy": self.strategy,
            "missing_channels": self.missing_channels,
            "donors_used": self.donors_used,
            "notes": self.notes,
        }


def synth_atlas_mean(
    image: np.ndarray,
    missing: Sequence[str],
    donors: DonorPool,
) -> Tuple[np.ndarray, SynthesisReport]:
    target_shape = image.shape[1:]
    donors.materialise_on_grid(target_shape)
    out = image.copy()
    for m in missing:
        ch = DEFAULT_CHANNEL_INDEX[m]
        # Average the missing channel over donors, restricted to the
        # target's brain mask so background stays exactly zero.
        donor_stack = donors.cache_images[:, ch, ...]  # (n, H, W, D)
        out[ch] = donor_stack.mean(axis=0)
    brain = _brain_mask_from_image(image)
    for m in missing:
        ch = DEFAULT_CHANNEL_INDEX[m]
        out[ch][~brain] = 0.0
    return out, SynthesisReport(
        case_id="", strategy="atlas_mean",
        missing_channels=list(missing),
        donors_used=int(donors.cache_images.shape[0]),
    )


def synth_linear_regression(
    image: np.ndarray,
    missing: Sequence[str],
    donors: DonorPool,
    ridge: float = 1.0,
    subsample_voxels: int = 200000,
    seed: int = 442,
) -> Tuple[np.ndarray, SynthesisReport]:
    """Per-target voxel-wise ridge regression from observed → missing.

    We fit on (voxels of donor cases) where observed-modality intensities
    play the predictor role and the missing modality plays the response,
    then apply to the target voxel-wise.

    The fit is global (one model per missing modality), not per spatial
    location, but the donors have already been brought to the target's
    grid so spatial structure leaks in naturally through the observed
    channels themselves.
    """
    target_shape = image.shape[1:]
    donors.materialise_on_grid(target_shape)
    n_donors = int(donors.cache_images.shape[0])

    rng = np.random.default_rng(seed)
    observed = [m for m in CANONICAL_MODALITIES if m not in missing]
    obs_idx = [DEFAULT_CHANNEL_INDEX[m] for m in observed]
    if not obs_idx:
        # No observed modality — fall back to atlas mean.
        out, report = synth_atlas_mean(image, missing, donors)
        report.strategy = "atlas_mean(fallback_no_observed)"
        return out, report

    brain_target = _brain_mask_from_image(image)
    brain_all_donors = np.all(donors.cache_brain_masks, axis=0) & brain_target

    coords = np.argwhere(brain_all_donors)
    if coords.shape[0] == 0:
        return synth_atlas_mean(image, missing, donors)
    if coords.shape[0] > subsample_voxels:
        pick = rng.choice(coords.shape[0], size=subsample_voxels, replace=False)
        coords = coords[pick]

    # Build (X, y) from donors at these voxel coordinates
    rows = []
    targets: Dict[str, List[np.ndarray]] = {m: [] for m in missing}
    for d in range(n_donors):
        donor_img = donors.cache_images[d]
        X_d = np.stack([donor_img[i][coords[:, 0], coords[:, 1], coords[:, 2]]
                        for i in obs_idx], axis=1)  # (n_vox, n_obs)
        rows.append(X_d)
        for m in missing:
            ch = DEFAULT_CHANNEL_INDEX[m]
            targets[m].append(
                donor_img[ch][coords[:, 0], coords[:, 1], coords[:, 2]]
            )
    X = np.concatenate(rows, axis=0)
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)

    out = image.copy()
    notes: Dict[str, str] = {}
    for m in missing:
        y = np.concatenate(targets[m], axis=0)
        # Ridge:  w = (X^T X + lambda I)^-1 X^T y
        A = X_aug.T @ X_aug + ridge * np.eye(X_aug.shape[1], dtype=X_aug.dtype)
        b = X_aug.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        # Predict at every voxel inside target's brain mask
        ch_pred = DEFAULT_CHANNEL_INDEX[m]
        X_full = np.stack([image[i][brain_target] for i in obs_idx], axis=1)
        X_full_aug = np.concatenate([X_full, np.ones((X_full.shape[0], 1),
                                                     dtype=X_full.dtype)], axis=1)
        pred = X_full_aug @ w
        synthesized = np.zeros(target_shape, dtype=np.float32)
        synthesized[brain_target] = pred.astype(np.float32)
        out[ch_pred] = synthesized
        # Report a residual standard deviation as a quality estimate
        residual = y - X_aug @ w
        notes[f"{m}_rmse"] = f"{float(np.sqrt(np.mean(residual ** 2))):.4f}"
        notes[f"{m}_weights"] = json.dumps([float(v) for v in w.tolist()])
    return out, SynthesisReport(
        case_id="", strategy="linear_regression",
        missing_channels=list(missing),
        donors_used=n_donors,
        notes=notes,
    )


def synth_patch_knn(
    image: np.ndarray,
    missing: Sequence[str],
    donors: DonorPool,
    k: int = 5,
    voxels_per_chunk: int = 50000,
    max_donor_voxels: int = 200000,
    seed: int = 442,
) -> Tuple[np.ndarray, SynthesisReport]:
    """Voxel-wise k-NN imputation in observed-modality intensity space.

    For each target voxel inside the brain mask we find the ``k`` donor
    voxels (across all donors, restricted to a random subsample of
    ``max_donor_voxels`` voxels for tractability) whose observed-modality
    intensities are closest in Euclidean distance, then average their
    missing-modality intensity. This is *not* a learnt model but a
    nonparametric lookup; it is surprisingly competitive on co-registered
    data after z-scoring.
    """
    target_shape = image.shape[1:]
    donors.materialise_on_grid(target_shape)

    observed = [m for m in CANONICAL_MODALITIES if m not in missing]
    obs_idx = [DEFAULT_CHANNEL_INDEX[m] for m in observed]
    if not obs_idx:
        out, report = synth_atlas_mean(image, missing, donors)
        report.strategy = "atlas_mean(fallback_no_observed)"
        return out, report

    rng = np.random.default_rng(seed)
    n_donors = int(donors.cache_images.shape[0])
    brain_target = _brain_mask_from_image(image)
    brain_all_donors = np.all(donors.cache_brain_masks, axis=0) & brain_target

    donor_coords = np.argwhere(brain_all_donors)
    if donor_coords.shape[0] > max_donor_voxels:
        pick = rng.choice(donor_coords.shape[0], size=max_donor_voxels, replace=False)
        donor_coords = donor_coords[pick]

    # Gather (donor_idx, voxel_coord) → observed intensity tuple + missing intensity
    donor_obs = []
    donor_targets: Dict[str, List[np.ndarray]] = {m: [] for m in missing}
    for d in range(n_donors):
        di = donors.cache_images[d]
        obs_d = np.stack([di[i][donor_coords[:, 0], donor_coords[:, 1], donor_coords[:, 2]]
                          for i in obs_idx], axis=1)  # (n_vox, n_obs)
        donor_obs.append(obs_d)
        for m in missing:
            ch = DEFAULT_CHANNEL_INDEX[m]
            donor_targets[m].append(
                di[ch][donor_coords[:, 0], donor_coords[:, 1], donor_coords[:, 2]]
            )
    donor_obs = np.concatenate(donor_obs, axis=0)  # (N, n_obs)
    target_obs_full = np.stack([image[i][brain_target] for i in obs_idx], axis=1)

    out = image.copy()

    n_target = target_obs_full.shape[0]
    for m in missing:
        donor_y = np.concatenate(donor_targets[m], axis=0)
        pred = np.zeros(n_target, dtype=np.float32)
        for start in range(0, n_target, voxels_per_chunk):
            stop = min(start + voxels_per_chunk, n_target)
            tx = target_obs_full[start:stop]
            # Squared distance to every donor voxel — vectorised but memory
            # intensive, hence the chunking.
            d2 = ((tx[:, None, :] - donor_obs[None, :, :]) ** 2).sum(axis=-1)
            nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            pred[start:stop] = donor_y[nn].mean(axis=1)
        ch_pred = DEFAULT_CHANNEL_INDEX[m]
        synthesized = np.zeros(target_shape, dtype=np.float32)
        synthesized[brain_target] = pred
        out[ch_pred] = synthesized

    return out, SynthesisReport(
        case_id="", strategy=f"patch_knn(k={k})",
        missing_channels=list(missing),
        donors_used=n_donors,
    )


# ---------------------------------------------------------------------------
# Learned-CNN hook (placeholder)
# ---------------------------------------------------------------------------

@dataclass
class LearnedSynth:
    """Stub for a learned modality translator.

    Concrete implementations should provide ``predict(observed_image,
    missing_modalities) -> synthesized_image`` that returns the full 4-channel
    array. Wiring is intentionally framework-agnostic so we can drop in a
    PyTorch model (e.g. a small 3-D U-Net) or a 2-D pix2pix without
    touching the calling code.
    """
    checkpoint_path: str

    def predict(self, image: np.ndarray, missing: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(
            "LearnedSynth is a hook only; train a translator and override "
            "predict() before calling. See src/models/network.py for a "
            "starting backbone."
        )


# ---------------------------------------------------------------------------
# Cohort-level driver
# ---------------------------------------------------------------------------

STRATEGIES = {
    "atlas_mean": synth_atlas_mean,
    "linear_regression": synth_linear_regression,
    "patch_knn": synth_patch_knn,
}


def _save_diff_panel(case_id: str, before: np.ndarray, after: np.ndarray,
                     missing: List[str], viz_dir: Path) -> Optional[str]:
    """Side-by-side before/after panel for a synthesised case.

    The figure has one row per missing channel. Left column shows the
    target's existing observed channels (the most informative one,
    typically T1ce or T1), middle shows the missing channel before
    synthesis (zeros), right shows the synthesised channel. Saved into
    the visualization output directory if one is given.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = after.shape[-1] // 2
    observed = [m for m in CANONICAL_MODALITIES if m not in missing]
    ref_idx = DEFAULT_CHANNEL_INDEX[observed[0]] if observed else 0
    rows = max(len(missing), 1)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    for i, m in enumerate(missing):
        ch = DEFAULT_CHANNEL_INDEX[m]
        axes[i, 0].imshow(after[ref_idx, :, :, z].T, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"observed ref: {observed[0] if observed else '-'}")
        axes[i, 1].imshow(before[ch, :, :, z].T, cmap="gray",
                          origin="lower", vmin=-3, vmax=3)
        axes[i, 1].set_title(f"{m}: BEFORE (missing)")
        axes[i, 2].imshow(after[ch, :, :, z].T, cmap="gray",
                          origin="lower", vmin=-3, vmax=3)
        axes[i, 2].set_title(f"{m}: AFTER synthesis")
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(f"case {case_id} — modality synthesis", y=1.01)
    fig.tight_layout()
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / f"{case_id}_synth_panel.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def synthesize_one(
    npz_path: str,
    donors: DonorPool,
    strategy: str = "linear_regression",
    out_dir: Optional[str] = None,
    skip_if_complete: bool = True,
    viz_dir: Optional[str] = None,
    **kwargs,
) -> SynthesisReport:
    data = np.load(npz_path)
    image = data["image"]
    missing = detect_missing_channels(image)
    if not missing and skip_if_complete:
        return SynthesisReport(
            case_id=Path(npz_path).stem,
            strategy="noop_complete",
            missing_channels=[],
            donors_used=0,
        )

    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"unknown synthesis strategy: {strategy}")

    new_image, report = fn(image, missing, donors, **kwargs)
    report.case_id = Path(npz_path).stem

    target_dir = Path(out_dir) if out_dir else Path(npz_path).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{Path(npz_path).stem}.npz"
    payload = dict(data)
    payload["image"] = new_image.astype(np.float32)
    np.savez_compressed(out_path, **payload)

    side = target_dir / f"{Path(npz_path).stem}_synth.json"
    side.write_text(json.dumps(report.to_dict(), indent=2))

    if viz_dir is not None:
        try:
            _save_diff_panel(report.case_id, image, new_image, missing, Path(viz_dir))
        except Exception as exc:
            report.notes["viz_error"] = str(exc)
    return report


def synthesize_cohort(
    in_dir: str,
    donors: DonorPool,
    strategy: str = "linear_regression",
    out_dir: Optional[str] = None,
    viz_dir: Optional[str] = None,
    **kwargs,
) -> List[SynthesisReport]:
    reports: List[SynthesisReport] = []
    for p in sorted(Path(in_dir).glob("*.npz")):
        reports.append(synthesize_one(
            str(p), donors=donors, strategy=strategy,
            out_dir=out_dir, viz_dir=viz_dir, **kwargs,
        ))
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "synthesis_report.json").write_text(
            json.dumps([r.to_dict() for r in reports], indent=2)
        )
    return reports
