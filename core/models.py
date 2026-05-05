from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

# === CONSTANTS ===

MIN_STAR_SNR = 30.0
HALO_MIN_STAR_SNR = 200.0
ISOLATION_RADIUS_FWHM = 5.0
HALO_FIT_RADIUS_PX = 80
GHOST_SEARCH_RADIUS_PX = 200
SATURATION_FRACTION = 0.90
DEFAULT_PIXEL_SCALE = 1.0       # arcsec/px fallback
SEEING_WARN_FWHM_ARCS = 3.0
EDGE_ROI_HALF_WIDTH = 30
POWER_SPECTRUM_NPIX = 256

STD_KERNEL_SIZES = (5, 15, 31)
LOG_SIGMAS = (1.5, 3.0, 6.0)
WAVELET_NAME = "db4"
WAVELET_LEVELS = 4


# === DATA CLASSES ===

@dataclass
class AnalysisResult:
    label: str
    psf_metrics: dict | None = None
    halo_metrics: dict | None = None
    ghost_metrics: dict | None = None
    edge_metrics: dict | None = None
    power_metrics: dict | None = None
    spatial_metrics: dict | None = None
    warnings: list[str] = field(default_factory=list)
    figures: dict[str, "matplotlib.figure.Figure"] = field(default_factory=dict)
