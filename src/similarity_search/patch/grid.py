"""Grid and index conversion helpers for patch maps.

This module contains row/column to linear-index conversion helpers and index
clamping utilities for navigating patch grids safely.
"""

import numpy as np


def rc_to_idx(row: int, col: int, cols: int) -> int:
    """Convert row and column indices to a single patch index."""
    return int(row) * int(cols) + int(col)


def idx_to_rc(index: int, cols: int) -> tuple[int, int]:
    """Convert a single patch index to row and column indices."""
    return int(index) // int(cols), int(index) % int(cols)


def clamp_idx(index: int, rows: int, cols: int) -> int:
    """Clamp a patch index to the valid range."""
    max_idx = int(rows) * int(cols) - 1
    return int(np.clip(index, 0, max_idx))
