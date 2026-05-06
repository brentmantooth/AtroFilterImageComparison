from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData, StdDevUncertainty
from photutils.background import Background2D, SExtractorBackground, MedianBackground

from core.models import DEFAULT_PIXEL_SCALE


# FITS keywords tried in priority order for pixel scale derivation
_PIXEL_SCALE_KEYWORDS = ["CDELT1", "CD1_1", "PIXSCALE", "SCALE"]

_DTYPE_LABELS: dict[str, str] = {
    "uint8":   "8-bit unsigned int",
    "int16":   "16-bit signed int",
    "uint16":  "16-bit unsigned int",
    "int32":   "32-bit signed int",
    "uint32":  "32-bit unsigned int",
    "float32": "32-bit float",
    "float64": "64-bit float",
}


def _dtype_label(dtype: np.dtype) -> str:
    return _DTYPE_LABELS.get(dtype.name, str(dtype))


def statistical_stretch(data: np.ndarray,
                         blackpoint_sigma: float = 5.0,
                         target_median: float = 0.25) -> np.ndarray:
    """SETIAstroSuite MTF-based statistical stretch for display only."""
    median = np.median(data)
    lower = data[data < median]
    if lower.size == 0:
        return np.clip(data, 0, 1)
    mad = np.median(np.abs(lower - median))
    blackpoint = max(0.0, float(median) - blackpoint_sigma * 1.4826 * float(mad))

    denom = 1.0 - blackpoint
    if denom <= 0:
        return np.clip(data, 0, 1)
    r = np.clip((data - blackpoint) / denom, 0.0, 1.0)

    m = target_median
    t = 0.5
    denom2 = m * (t + r - 1.0) - t * r
    # Guard against division by zero
    safe = np.abs(denom2) > 1e-12
    out = np.where(safe, ((m - 1.0) * t * r) / np.where(safe, denom2, 1.0), r)
    return np.clip(out, 0.0, 1.0)


class AstroImage:
    """Wraps a single FITS or XISF file for analysis."""

    def __init__(self, path: str, label: str = ""):
        self.path = Path(path)
        self.label = label or self.path.stem
        self.data: np.ndarray | None = None
        self.header: fits.Header | None = None
        self.pixel_scale: float = DEFAULT_PIXEL_SCALE
        self.pixel_scale_is_estimated: bool = False
        self.bandwidth_nm: float | None = None
        self.original_dtype: np.dtype | None = None   # dtype before float64 conversion
        self.background: Background2D | None = None
        self.background_rms: np.ndarray | None = None
        self._load_error: str | None = None

        # Extracted metadata for display
        self.meta: dict = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        suffix = self.path.suffix.lower()
        if suffix in (".fits", ".fit", ".fts"):
            self._load_fits()
        elif suffix == ".xisf":
            self._load_xisf()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        if self.data is not None:
            self.original_dtype = self.data.dtype   # capture before float64 conversion
            self.data = self.data.astype(np.float64)
            self.pixel_scale = self._extract_pixel_scale()
            self.bandwidth_nm = self._extract_bandwidth()
            self._extract_metadata()

    def _load_fits(self) -> None:
        with fits.open(self.path) as hdul:
            for hdu in hdul:
                if (hdu.data is not None
                        and hdu.data.ndim == 2
                        and max(hdu.data.shape) > 100):
                    self.data = hdu.data.copy()
                    self.header = hdu.header.copy()
                    return
        raise ValueError(f"No valid 2D image data found in {self.path.name}")

    def _load_xisf(self) -> None:
        try:
            import xisf as xisf_lib  # type: ignore
        except ImportError:
            raise ImportError("xisf package not installed. Run: pip install xisf")
        x = xisf_lib.XISF(str(self.path))
        img = x.read_image(0)
        if img is None:
            raise ValueError(f"No image data in {self.path.name}")
        # xisf returns (H, W) or (H, W, C); take first channel if colour
        if img.ndim == 3:
            img = img[:, :, 0]
        self.data = img   # float64 conversion happens in load() after dtype is captured
        # Build a minimal header-like dict from XISF metadata
        meta_list = x.get_images_metadata()
        if meta_list:
            self.header = fits.Header()
            raw = meta_list[0]
            # Map common XISF FITSKeyword entries into the header
            fk = raw.get("FITSKeywords", {})
            for key, entries in fk.items():
                if entries:
                    self.header[key] = entries[0].get("value", "")

    # ------------------------------------------------------------------
    # Pixel scale
    # ------------------------------------------------------------------

    def _extract_pixel_scale(self) -> float:
        if self.header is None:
            self.pixel_scale_is_estimated = True
            return DEFAULT_PIXEL_SCALE

        # CDELT1 in degrees/px
        if "CDELT1" in self.header:
            return abs(float(self.header["CDELT1"])) * 3600.0

        # CD matrix
        if "CD1_1" in self.header:
            return abs(float(self.header["CD1_1"])) * 3600.0

        # Direct arcsec/px keywords
        for kw in ("PIXSCALE", "SCALE"):
            if kw in self.header:
                return float(self.header[kw])

        # Derive from focal length + pixel size
        if "FOCALLEN" in self.header and "XPIXSZ" in self.header:
            focallen_mm = float(self.header["FOCALLEN"])
            xpixsz_um = float(self.header["XPIXSZ"])
            if focallen_mm > 0:
                return (xpixsz_um / focallen_mm) * 206.265

        self.pixel_scale_is_estimated = True
        return DEFAULT_PIXEL_SCALE

    # ------------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------------

    def _extract_bandwidth(self) -> float | None:
        if self.header is None:
            return None
        for kw in ("BANDWID", "FWHM", "BANDWIDTH"):
            if kw in self.header:
                try:
                    return float(self.header[kw])
                except (ValueError, TypeError):
                    pass
        return None

    # ------------------------------------------------------------------
    # Metadata for GUI display
    # ------------------------------------------------------------------

    def _extract_metadata(self) -> None:
        # Bit depth — available regardless of header
        if self.original_dtype is not None:
            self.meta["Bit depth"] = _dtype_label(self.original_dtype)

        if self.header is None:
            return
        mapping = {
            "Telescope":    ["TELESCOP"],
            "Camera":       ["INSTRUME"],
            "Filter":       ["FILTER"],
            "Focal length": ["FOCALLEN"],
            "Pixel size":   ["XPIXSZ"],
            "Exposure":     ["EXPTIME", "EXPOSURE"],
            "Date":         ["DATE-OBS"],
            "Gain":         ["GAIN"],
            "Binning":      ["XBINNING"],
            "Bandwidth":    ["BANDWID", "FWHM", "BANDWIDTH"],
        }
        for display_key, keywords in mapping.items():
            for kw in keywords:
                if kw in self.header:
                    self.meta[display_key] = str(self.header[kw]).strip()
                    break

    # ------------------------------------------------------------------
    # Background estimation
    # ------------------------------------------------------------------

    def estimate_background(self, box_size: int = 64) -> None:
        if self.data is None:
            raise RuntimeError("Image not loaded")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.background = Background2D(
                self.data,
                box_size=box_size,
                filter_size=3,
                bkg_estimator=SExtractorBackground(),
                bkgrms_estimator=MedianBackground(),
            )
        self.background_rms = self.background.background_rms

    def background_subtracted(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("Image not loaded")
        if self.background is None:
            return self.data.copy()
        return self.data - self.background.background

    def saturation_threshold(self) -> float:
        if self.data is None:
            return 65535.0
        if self.header is not None and "DATAMAX" in self.header:
            try:
                return float(self.header["DATAMAX"]) * 0.90
            except (ValueError, TypeError):
                pass
        return float(np.max(self.data)) * 0.90

    def nddata(self) -> NDData:
        """Return NDData with uncertainty plane for photutils PSF tools."""
        bgsub = self.background_subtracted()
        if self.background_rms is not None:
            uncertainty = StdDevUncertainty(self.background_rms)
        else:
            uncertainty = None
        return NDData(bgsub, uncertainty=uncertainty)

    # ------------------------------------------------------------------
    # Display stretch
    # ------------------------------------------------------------------

    def display_image(self,
                      stretch: bool = True,
                      blackpoint_sigma: float = 5.0,
                      target_median: float = 0.25) -> np.ndarray:
        """Return uint8 array suitable for Qt display.

        stretch=True  — SETIAstroSuite statistical stretch (default)
        stretch=False — percentile-clipped linear scale (0.1%–99.9%)
        """
        if self.data is None:
            raise RuntimeError("Image not loaded")
        if stretch:
            scaled = statistical_stretch(self.data, blackpoint_sigma, target_median)
        else:
            lo, hi = np.percentile(self.data, [0.1, 99.9])
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((self.data - lo) / (hi - lo), 0.0, 1.0)
        return (scaled * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        shape = self.data.shape if self.data is not None else "not loaded"
        return f"AstroImage({self.label!r}, shape={shape}, scale={self.pixel_scale:.3f}\"/px)"
