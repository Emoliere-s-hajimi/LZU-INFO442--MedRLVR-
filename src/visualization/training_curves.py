"""Plot training/validation curves from a CSV log produced by `src.train`."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def _safe_import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(csv_path: str, columns: Sequence[str], out_path: Optional[str] = None):
    """Read a CSV with at least an `epoch` column plus the requested metric
    columns, and plot one panel per metric."""
    import pandas as pd

    plt = _safe_import_plt()
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3))
    axes = axes if hasattr(axes, "__iter__") else [axes]
    for ax, col in zip(axes, columns):
        if col not in df:
            ax.set_title(f"{col} (missing)")
            continue
        ax.plot(df["epoch"], df[col], label=col)
        ax.set_xlabel("epoch")
        ax.set_ylabel(col)
        ax.set_title(col)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
