"""Pre-render T1/T2 mid-axial slices to PNG (8-bit gray) for the
synthesis demo. The JS then loads them as ImageData and runs the
recipe directly in canvas.

Usage:
    python web/data/_make_synth_slices.py
"""
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image


def to_uint8(arr):
    fg = arr > 0
    if not fg.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[fg], [1, 99])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-8)
    return (arr * 255).astype(np.uint8)


def main():
    base = Path("web/data/cases/N_005")
    out_dir = Path("web/data/synth")
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = nib.load(base / "t1.nii.gz").get_fdata()
    t2 = nib.load(base / "t2.nii.gz").get_fdata()

    # Pick the axial slice with the most foreground area
    fg = (t1 > 0).sum(axis=(0, 1))
    z = int(np.argmax(fg))
    print(f"using axial slice z={z}")

    # Export 13 slices around the chosen one so the JS can scrub
    z0 = max(0, z - 6)
    z1 = min(t1.shape[2], z + 7)
    for i, zi in enumerate(range(z0, z1)):
        t1_slice = np.rot90(t1[:, :, zi])
        t2_slice = np.rot90(t2[:, :, zi])
        Image.fromarray(to_uint8(t1_slice)).save(out_dir / f"t1_{i:02d}.png")
        Image.fromarray(to_uint8(t2_slice)).save(out_dir / f"t2_{i:02d}.png")
    print(f"wrote {z1 - z0} pairs to {out_dir}")


if __name__ == "__main__":
    main()
