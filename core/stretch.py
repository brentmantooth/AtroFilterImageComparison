from __future__ import annotations

import numpy as np


def normalize_unit_interval(data: np.ndarray) -> np.ndarray:
    """Normalise raw ADU data to [0, 1].

    Uses dtype integer max for integer arrays (preserves absolute black
    level); falls back to per-image min/max for float arrays.
    """
    if np.issubdtype(data.dtype, np.integer):
        imax = float(np.iinfo(data.dtype).max)
        return data.astype(np.float64) / imax
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    if dmax <= dmin:
        return np.zeros_like(data, dtype=np.float64)
    return (data.astype(np.float64) - dmin) / (dmax - dmin)


def mtf(x: np.ndarray, m: float) -> np.ndarray:
    """PixInsight-compatible midtone transfer function."""
    if m == 0.0:
        return np.where(x == 0.0, 0.0, 1.0).astype(np.float64)
    if m == 1.0:
        return np.where(x == 1.0, 1.0, 0.0).astype(np.float64)
    x = np.asarray(x, dtype=np.float64)
    denom = ((2 * m - 1) * x - m)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            denom == 0,
            0.0,
            (m - 1) * x / denom
        )
    return np.clip(result, 0.0, 1.0)


def stf_stretch(data: np.ndarray,
                shadow_clip: float = -2.8,
                target_bkg: float = 0.25) -> np.ndarray:
    """Screen Transfer Function stretch (PixInsight-compatible).

    Normalises to [0, 1], clips shadows at median + shadow_clip × 1.4826 × MAD,
    then applies the midtone transfer function so the background median maps to
    target_bkg.
    """
    norm = normalize_unit_interval(data)

    # Shadow clipping: median - |shadow_clip| * 1.4826 * MAD
    median = float(np.median(norm))
    mad = float(np.median(np.abs(norm - median)))
    shadows = median + shadow_clip * 1.4826 * mad   # shadow_clip is negative → lowers black
    shadows = float(np.clip(shadows, 0.0, 1.0))

    # Clip shadows and rescale to [0, 1]
    clipped = np.clip(norm - shadows, 0.0, None)
    max_val = float(np.max(clipped))
    if max_val <= 0:
        return np.zeros_like(norm)
    clipped /= max_val

    # Midpoint for MTF: place background median at target_bkg
    bg_after_clip = float(np.median(clipped))
    if bg_after_clip <= 0 or bg_after_clip >= 1:
        m = target_bkg
    else:
        # Solve MTF(m, bg) = target_bkg  →  m = target_bkg*(bg - 1) / ((2*target_bkg - 1)*bg - target_bkg)
        bg = bg_after_clip
        denom = (2 * target_bkg - 1) * bg - target_bkg
        if abs(denom) < 1e-12:
            m = target_bkg
        else:
            m = float(np.clip(target_bkg * (bg - 1) / denom, 0.0, 1.0))

    return mtf(clipped, m)


def normalize_for_display(data: np.ndarray,
                           stretch: bool = True) -> np.ndarray:
    """Return a uint8 array suitable for display.

    stretch=True  — STF nonlinear stretch (default; adapts black level per-image)
    stretch=False — linear min-max normalisation
    """
    if data is None or data.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if stretch:
        scaled = stf_stretch(data)
    else:
        scaled = normalize_unit_interval(data)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)
