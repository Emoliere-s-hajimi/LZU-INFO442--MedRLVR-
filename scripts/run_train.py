"""Thin wrapper so the train command line is consistent with run_clean / run_eda."""
from __future__ import annotations

import sys

from src.train import main


if __name__ == "__main__":
    main()
