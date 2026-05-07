from __future__ import annotations

import base64
import io
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import zoom as _ndimage_zoom
from PIL import Image as _PILImage

from core.models import AnalysisResult, HALO_FIT_RADIUS_PX
from core.astro_image import AstroImage

_TEST_IMAGE_PATH = Path(__file__).parent.parent / "resources" / "ContrastTestImage.png"


# ── CSS ──────────────────────────────────────────────────────────────────────

_CSS = """
body { font-family: Segoe UI, Arial, sans-serif; max-width: 960px;
       margin: 0 auto; padding: 20px; color: #222; background: #fafafa; }
h1 { background: #1a3a5c; color: white; padding: 14px 18px;
     border-radius: 6px; margin-bottom: 4px; }
h2 { background: #2d6da3; color: white; padding: 8px 14px;
     border-radius: 4px; margin-top: 28px; }
h3 { color: #1a3a5c; border-bottom: 2px solid #2d6da3;
     padding-bottom: 4px; margin-top: 20px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th { background: #2d6da3; color: white; padding: 8px 12px; text-align: left; }
td { padding: 7px 12px; border-bottom: 1px solid #dde; }
tr:nth-child(even) { background: #f0f4fa; }
.better { background: #d4edda !important; font-weight: bold; }
.worse  { background: #f8d7da !important; }
.warn-box { background: #fff3cd; border: 1px solid #ffc107;
            border-radius: 4px; padding: 10px 14px; margin: 10px 0; }
.info-box { background: #d1ecf1; border: 1px solid #bee5eb;
            border-radius: 4px; padding: 10px 14px; margin: 10px 0; }
.bw-warn { background: #f8d7da; border: 1px solid #f5c6cb;
           border-radius: 4px; padding: 12px 16px; margin: 14px 0;
           font-size: 1.05em; }
.metric-label-ok   { color: #155724; font-weight: bold; }
.metric-label-warn { color: #856404; font-weight: bold; }
img { max-width: 100%; height: auto; border: 1px solid #ccc;
      border-radius: 4px; margin: 8px 0; }
.caption { font-style: italic; color: #555; font-size: 0.92em;
           margin-top: -4px; margin-bottom: 12px; }
.error-box { background: #f8d7da; border: 1px solid #f5c6cb;
             border-radius: 4px; padding: 10px 14px; margin: 10px 0;
             font-family: monospace; white-space: pre-wrap; font-size: 0.9em; }
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_tag(fig: plt.Figure, alt: str = "") -> str:
    if fig is None:
        return ""
    return f'<img src="data:image/png;base64,{_fig_to_b64(fig)}" alt="{alt}">'


def _arr_to_b64_png(arr: np.ndarray) -> str:
    """Convert a uint8 H×W (gray) or H×W×3 (RGB) array to base64 PNG at native resolution."""
    if arr.ndim == 2:
        img = _PILImage.fromarray(arr, mode="L")
    else:
        img = _PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _arr_img_tag(arr: np.ndarray, alt: str = "") -> str:
    """Inline <img> at native (1:1) pixel resolution from a uint8 numpy array."""
    return f'<img src="data:image/png;base64,{_arr_to_b64_png(arr)}" alt="{alt}" style="max-width:100%;display:block;">'


def _val(v, fmt=".3f", fallback="—") -> str:
    if v is None:
        return fallback
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def _error_box(metric_key: str, ra: AnalysisResult, rb: AnalysisResult) -> str:
    """Return an error box HTML if the metric failed, else empty string."""
    err = ra.errors.get(metric_key) or rb.errors.get(metric_key)
    if not err:
        return ""
    return f'<div class="error-box">⚠ <strong>Analysis failed:</strong> {err}</div>'


def _better_worse_class(val_a, val_b, higher_is_better: bool = True) -> tuple[str, str]:
    if val_a is None or val_b is None:
        return "", ""
    if higher_is_better:
        return ("better", "worse") if val_a >= val_b else ("worse", "better")
    return ("better", "worse") if val_a <= val_b else ("worse", "better")


# ── Main class ────────────────────────────────────────────────────────────────

class ReportBuilder:
    """Generate a self-contained HTML comparison report."""

    def generate(self, image_a: AstroImage, image_b: AstroImage,
                  result_a: AnalysisResult, result_b: AnalysisResult,
                  output_dir: str | Path,
                  open_browser: bool = True) -> Path:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{result_a.label}_{result_b.label}_{ts}.html".replace(" ", "_")
        out_path = output_dir / filename

        bw_a = image_a.bandwidth_nm
        bw_b = image_b.bandwidth_nm
        bw_differ = (bw_a is not None and bw_b is not None and
                     abs(bw_a - bw_b) > 0.1)

        sections = [
            self._section_header(image_a, image_b, result_a, result_b, bw_differ),
            self._section_observation(result_a, result_b),
            self._section_psf(result_a, result_b),
            self._section_halo(result_a, result_b, image_a, image_b),
            self._section_ghost(result_a, result_b),
            self._section_edge(result_a, result_b, bw_differ),
            self._section_power(result_a, result_b),
            self._section_spatial(result_a, result_b),
            self._section_summary(result_a, result_b, bw_differ),
        ]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Filter Comparison: {result_a.label} vs {result_b.label}</title>
  <style>{_CSS}</style>
</head>
<body>
{"".join(sections)}
<p style="color:#999;font-size:0.85em;margin-top:40px;">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by FilterImageCompare
</p>
</body>
</html>"""

        out_path.write_text(html, encoding="utf-8")
        if open_browser:
            webbrowser.open(out_path.as_uri())
        return out_path

    # ── Section 1: Header ─────────────────────────────────────────────────────

    def _section_header(self, img_a: AstroImage, img_b: AstroImage,
                         result_a: AnalysisResult, result_b: AnalysisResult,
                         bw_differ: bool) -> str:
        bw_warn = ""
        if bw_differ:
            bw_warn = (f'<div class="bw-warn">⚠ <strong>Bandwidth warning:</strong> '
                       f'Filters have different bandwidths '
                       f'({img_a.bandwidth_nm:.1f} nm vs {img_b.bandwidth_nm:.1f} nm). '
                       f'Metrics marked <span class="metric-label-warn">⚠</span> are '
                       f'sensitive to this difference and should be interpreted with caution. '
                       f'Metrics marked <span class="metric-label-ok">✓</span> are '
                       f'bandwidth-independent.</div>')

        def meta_rows(img: AstroImage, result: AnalysisResult) -> str:
            rows = ""
            for key, val in img.meta.items():
                rows += f"<tr><td><strong>{key}</strong></td><td>{val}</td></tr>"
            if img.pixel_scale_is_estimated:
                rows += ("<tr><td><strong>Pixel scale</strong></td>"
                         f"<td>{img.pixel_scale:.3f} \"/px (estimated)</td></tr>")
            else:
                rows += ("<tr><td><strong>Pixel scale</strong></td>"
                         f"<td>{img.pixel_scale:.3f} \"/px</td></tr>")
            if img.bandwidth_nm:
                rows += ("<tr><td><strong>Bandwidth</strong></td>"
                         f"<td>{img.bandwidth_nm:.1f} nm</td></tr>")
            n_total = (result.psf_metrics or {}).get("n_stars_total")
            if n_total is not None:
                rows += (f"<tr><td><strong>Stars detected</strong></td>"
                         f"<td>{n_total}</td></tr>")
            sl = getattr(img, "starless_image", None)
            if sl is not None:
                rows += (f"<tr><td><strong>Starless</strong></td>"
                         f"<td>{sl.path.name}</td></tr>")
            return rows

        # Compute linear stretch limits from main image data for starless thumbnails
        lo_a, hi_a = np.percentile(img_a.data, [0.1, 99.9])
        lo_b, hi_b = np.percentile(img_b.data, [0.1, 99.9])
        sl_a = getattr(img_a, "starless_image", None)
        sl_b = getattr(img_b, "starless_image", None)

        thumb_a = _img_tag(self._thumbnail_fig(img_a), f"Preview {img_a.label}")
        thumb_b = _img_tag(self._thumbnail_fig(img_b), f"Preview {img_b.label}")
        thumb_sl_a = _img_tag(self._thumbnail_fig(sl_a, lo=lo_a, hi=hi_a),
                               f"Starless {img_a.label}") if sl_a else ""
        thumb_sl_b = _img_tag(self._thumbnail_fig(sl_b, lo=lo_b, hi=hi_b),
                               f"Starless {img_b.label}") if sl_b else ""

        sl_cap_a = ('<p class="caption">Starless (linear stretch, same scale)</p>'
                    if sl_a else "")
        sl_cap_b = ('<p class="caption">Starless (linear stretch, same scale)</p>'
                    if sl_b else "")

        hist_tag = _img_tag(self._plot_image_histograms(img_a, img_b), "Pixel histograms")

        return f"""
<h1>Filter Image Comparison Report</h1>
<p><strong>{img_a.label}</strong> vs <strong>{img_b.label}</strong></p>
{bw_warn}
<h2>1. Image Metadata</h2>
<div style="display:flex;gap:20px;">
  <div style="flex:1;">
    <h3>{img_a.label}</h3>
    {thumb_a}
    {thumb_sl_a}{sl_cap_a}
    <table><tbody>{meta_rows(img_a, result_a)}</tbody></table>
  </div>
  <div style="flex:1;">
    <h3>{img_b.label}</h3>
    {thumb_b}
    {thumb_sl_b}{sl_cap_b}
    <table><tbody>{meta_rows(img_b, result_b)}</tbody></table>
  </div>
</div>
<h3>Pixel Histograms</h3>
{hist_tag}
<p class="caption">Log-scale pixel value distributions. Dotted vertical lines mark the median of each image.</p>"""

    def _plot_image_histograms(self, img_a: AstroImage, img_b: AstroImage) -> plt.Figure | None:
        """Combined log-scale histogram of both images with median markers."""
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = {"a": "steelblue", "b": "tomato"}

            for img, key, label in [(img_a, "a", img_a.label), (img_b, "b", img_b.label)]:
                pixels = img.data.ravel().astype(float)
                positive = pixels[pixels > 0]
                if len(positive) == 0:
                    continue
                lo, hi = np.percentile(positive, [0.01, 99.99])
                if lo <= 0:
                    lo = positive.min()
                if hi <= lo:
                    hi = lo * 10
                bins = np.geomspace(lo, hi, 256)
                counts, edges = np.histogram(positive, bins=bins)
                centers = np.sqrt(edges[:-1] * edges[1:])
                color = colors[key]
                ax.step(centers, counts, where="mid", color=color,
                        alpha=0.85, linewidth=1.4, label=label)
                median_val = float(np.median(positive))
                ax.axvline(median_val, color=color, linestyle=":", linewidth=1.5)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Pixel value")
            ax.set_ylabel("Count")
            ax.set_title("Pixel value histogram")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25, which="both")
            fig.tight_layout()
            return fig
        except Exception:
            return None

    def _thumbnail_fig(self, img: AstroImage,
                        lo: float | None = None,
                        hi: float | None = None) -> plt.Figure | None:
        """Return a small matplotlib figure with a stretched preview of the image.

        If lo/hi are provided, apply a manual linear stretch using those clip limits
        (used for starless thumbnails so the scale matches the main image).
        """
        if img is None or img.data is None:
            return None
        try:
            if lo is not None and hi is not None:
                hi_eff = hi if hi > lo else lo + 1.0
                arr_f = np.clip((img.data.astype(float) - lo) / (hi_eff - lo), 0.0, 1.0)
                max_dim = max(arr_f.shape[:2])
                if max_dim > 400:
                    step = max_dim // 400 + 1
                    arr_f = arr_f[::step, ::step]
                arr = (arr_f * 255).astype(np.uint8)
            else:
                arr = img.display_image(stretch=True)
                max_dim = max(arr.shape[:2])
                if max_dim > 400:
                    step = max_dim // 400 + 1
                    arr = arr[::step, ::step]
            fig, ax = plt.subplots(figsize=(4, 4 * arr.shape[0] / arr.shape[1]))
            ax.imshow(arr, origin="upper", cmap="gray", interpolation="bilinear",
                      aspect="auto")
            ax.axis("off")
            fig.tight_layout(pad=0)
            return fig
        except Exception:
            return None

    # ── Section 2: Observation context ────────────────────────────────────────

    def _section_observation(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        psf_a = ra.psf_metrics or {}
        psf_b = rb.psf_metrics or {}
        seeing_warn = ""
        if psf_a.get("seeing_dominated") or psf_b.get("seeing_dominated"):
            seeing_warn = (
                '<div class="warn-box">⚠ <strong>Seeing warning:</strong> '
                'FWHM exceeds 3″ in one or both images. PSF and MTF differences '
                'between filters may reflect atmospheric seeing variation rather than '
                'filter optical quality. For the most valid comparisons, images should '
                'be taken on the same night under similar conditions.</div>'
            )

        all_warnings = list(set(ra.warnings + rb.warnings))
        warn_html = ""
        if all_warnings:
            items = "".join(f"<li>{w}</li>" for w in all_warnings)
            warn_html = f'<div class="warn-box"><ul>{items}</ul></div>'

        return f"""
<h2>2. Observation Context</h2>
{seeing_warn}
{warn_html}
<div class="info-box">PSF/MTF comparisons are most meaningful when both images were
captured on the same night under similar atmospheric conditions. DATE-OBS values are
shown in the metadata table above.</div>"""

    # ── Section 3: PSF / MTF ──────────────────────────────────────────────────

    def _section_psf(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        err = _error_box("psf", ra, rb)
        pa = ra.psf_metrics or {}
        pb = rb.psf_metrics or {}
        ca, cb = _better_worse_class(pa.get("fwhm_px"), pb.get("fwhm_px"), higher_is_better=False)
        ma, mb = _better_worse_class(pa.get("mtf50_cycles_per_px"), pb.get("mtf50_cycles_per_px"))

        fig_mtf = None
        if "figures" in pa and "mtf" in pa["figures"] and "mtf" in (pb.get("figures") or {}):
            fig_mtf = self._overlay_mtf(pa["figures"]["mtf"], pb["figures"]["mtf"],
                                        ra.label, rb.label)

        img_mtf = _img_tag(fig_mtf, "MTF comparison")
        img_epsf_a = _img_tag((pa.get("figures") or {}).get("epsf"), f"ePSF {ra.label}")
        img_epsf_b = _img_tag((pb.get("figures") or {}).get("epsf"), f"ePSF {rb.label}")
        img_scatter = _img_tag(self._plot_fwhm_scatter(ra, rb), "FWHM scatter")

        return f"""
<h2>3. PSF / MTF &nbsp;<span class="metric-label-ok">✓ bandwidth-independent</span></h2>
{err}
<div class="info-box">The Point Spread Function (PSF) describes how a point source
(star) is rendered. FWHM measures the core width; smaller FWHM = sharper stars.
The Modulation Transfer Function (MTF) shows how well contrast is preserved at each
spatial frequency; MTF50 is the frequency at which contrast falls to 50%.
These metrics are normalised to unit amplitude and are valid regardless of filter bandwidth.</div>

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Stars in catalog</td><td>{_val(pa.get("n_stars_total"), "d")}</td><td>{_val(pb.get("n_stars_total"), "d")}</td></tr>
  <tr><td>Stars used for PSF</td><td>{_val(pa.get("n_stars_used"), "d")}</td><td>{_val(pb.get("n_stars_used"), "d")}</td></tr>
  <tr><td>FWHM (px)</td><td class="{ca}">{_val(pa.get("fwhm_px"))}</td><td class="{cb}">{_val(pb.get("fwhm_px"))}</td></tr>
  <tr><td>FWHM (arcsec)</td><td class="{ca}">{_val(pa.get("fwhm_arcsec"))}</td><td class="{cb}">{_val(pb.get("fwhm_arcsec"))}</td></tr>
  <tr><td>Moffat β</td><td>{_val(pa.get("beta"))}</td><td>{_val(pb.get("beta"))}</td></tr>
  <tr><td>Ellipticity</td><td>{_val(pa.get("ellipticity"))}</td><td>{_val(pb.get("ellipticity"))}</td></tr>
  <tr><td>MTF50 (cyc/px)</td><td class="{ma}">{_val(pa.get("mtf50_cycles_per_px"), ".4f")}</td><td class="{mb}">{_val(pb.get("mtf50_cycles_per_px"), ".4f")}</td></tr>
  <tr><td>MTF @ Nyquist</td><td>{_val(pa.get("mtf_nyquist"), ".4f")}</td><td>{_val(pb.get("mtf_nyquist"), ".4f")}</td></tr>
</table>

{img_mtf}
<p class="caption">MTF curves for both filters overlaid. Higher curve = better
contrast preservation at fine scales.</p>

{img_scatter}
<p class="caption">Per-star FWHM correlation. Points near the slope = 1 line indicate
consistent star size between filters. Systematic offset reveals which filter produces
tighter stars. Points far from the line indicate individual star measurement scatter.</p>

<div style="display:flex;gap:10px;">
  <div style="flex:1;">{img_epsf_a}</div>
  <div style="flex:1;">{img_epsf_b}</div>
</div>
<p class="caption">Empirical PSFs (log scale). Tighter, rounder cores indicate
better optical quality. Ellipticity &gt; 0.1 may indicate filter tilt or astigmatism.</p>

{self._psf_simulation_html(ra, rb)}

<div class="info-box"><strong>What to look for:</strong> A smaller FWHM and higher
MTF50 indicate sharper image resolution. A higher Moffat β (steeper wing falloff)
indicates less scattered light. Ellipticity should be similar between filters;
large differences may indicate filter flatness issues.</div>"""

    def _plot_fwhm_scatter(self, ra: AnalysisResult, rb: AnalysisResult) -> plt.Figure | None:
        """Scatter plot of per-star FWHM_A vs FWHM_B for matched stars."""
        data_a = (ra.psf_metrics or {}).get("star_data", [])
        data_b = (rb.psf_metrics or {}).get("star_data", [])
        if not data_a or not data_b:
            return None

        # Match by nearest neighbour in image coordinates (valid post-alignment)
        matched_a, matched_b = [], []
        pos_b = np.array([[s["x"], s["y"]] for s in data_b])
        for sa in data_a:
            dists = np.sqrt((pos_b[:, 0] - sa["x"])**2 + (pos_b[:, 1] - sa["y"])**2)
            idx = int(np.argmin(dists))
            if dists[idx] < 15.0:
                matched_a.append(sa["fwhm"])
                matched_b.append(data_b[idx]["fwhm"])

        if len(matched_a) < 3:
            return None

        fa = np.array(matched_a)
        fb = np.array(matched_b)
        lo = min(fa.min(), fb.min()) * 0.9
        hi = max(fa.max(), fb.max()) * 1.1

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(fa, fb, alpha=0.65, color="steelblue", s=25, zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Slope = 1 (equal FWHM)")
        ax.set_xlabel(f"FWHM {ra.label} (px)")
        ax.set_ylabel(f"FWHM {rb.label} (px)")
        ax.set_title(f"Per-star FWHM correlation  (n = {len(fa)} matched stars)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _overlay_mtf(self, fig_a: plt.Figure, fig_b: plt.Figure,
                      label_a: str, label_b: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4))
        for fig_src, label, color in [(fig_a, label_a, "steelblue"),
                                       (fig_b, label_b, "tomato")]:
            try:
                src_ax = fig_src.axes[0]
                line = src_ax.lines[0]
                ax.plot(line.get_xdata(), line.get_ydata(),
                        color=color, linewidth=2, label=label)
            except (IndexError, AttributeError):
                pass
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0.5, color="red", linestyle=":", linewidth=0.8, label="Nyquist")
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("MTF")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title("MTF comparison")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_psf_simulation(self, ra: AnalysisResult,
                              rb: AnalysisResult) -> dict | None:
        """Convolve the test chart with each filter's ePSF.

        Returns a dict with uint8 numpy arrays (one per panel) at native pixel
        resolution, or None if unavailable.  Keys: 'original', 'conv_a', 'conv_b',
        'diff', 'diff_max', 'label_a', 'label_b'.
        """
        epsf_a = (ra.psf_metrics or {}).get("epsf_data")
        epsf_b = (rb.psf_metrics or {}).get("epsf_data")
        if epsf_a is None or epsf_b is None:
            return None
        if not _TEST_IMAGE_PATH.exists():
            return None
        try:
            test_arr = np.array(
                _PILImage.open(_TEST_IMAGE_PATH).convert("L"), dtype=float
            ) / 255.0

            os_a = (ra.psf_metrics or {}).get("epsf_oversampling", 2)
            os_b = (rb.psf_metrics or {}).get("epsf_oversampling", 2)
            kern_a = _ndimage_zoom(epsf_a, 1.0 / os_a, order=1)
            kern_b = _ndimage_zoom(epsf_b, 1.0 / os_b, order=1)
            kern_a = kern_a / kern_a.sum() if kern_a.sum() > 0 else kern_a
            kern_b = kern_b / kern_b.sum() if kern_b.sum() > 0 else kern_b

            # Convolution at full resolution
            conv_a = np.clip(fftconvolve(test_arr, kern_a, mode="same"), 0.0, 1.0)
            conv_b = np.clip(fftconvolve(test_arr, kern_b, mode="same"), 0.0, 1.0)
            diff = conv_a - conv_b

            # Downsample for display if image is very large (cap at 1200 px on long edge)
            h, w = test_arr.shape
            if max(h, w) > 1200:
                zoom_f = 1200.0 / max(h, w)
                test_arr = _ndimage_zoom(test_arr, zoom_f, order=1)
                conv_a   = _ndimage_zoom(conv_a,   zoom_f, order=1)
                conv_b   = _ndimage_zoom(conv_b,   zoom_f, order=1)
                diff     = _ndimage_zoom(diff,     zoom_f, order=1)

            d_max = max(float(abs(diff).max()), 1e-9)
            # Map diff to RGB using RdBu_r colormap
            diff_norm = (diff / d_max + 1.0) / 2.0          # [0, 1]
            diff_rgb = (plt.get_cmap("RdBu_r")(diff_norm)[:, :, :3] * 255).astype(np.uint8)

            return {
                "original": (test_arr * 255).astype(np.uint8),
                "conv_a":   (conv_a   * 255).astype(np.uint8),
                "conv_b":   (conv_b   * 255).astype(np.uint8),
                "diff":     diff_rgb,
                "diff_max": d_max,
                "label_a":  ra.label,
                "label_b":  rb.label,
            }
        except Exception:
            return None

    def _psf_simulation_html(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        """Return HTML block with four PSF simulation panels at 1:1 pixel resolution."""
        sim = self._plot_psf_simulation(ra, rb)
        if sim is None:
            return ""

        def panel(arr, title, caption=""):
            tag = _arr_img_tag(arr, title)
            cap = f'<p class="caption">{caption}</p>' if caption else ""
            return f'<div style="margin-bottom:20px;"><p><strong>{title}</strong></p>{tag}{cap}</div>'

        diff_caption = (
            f"Pixel-level difference A − B (RdBu_r colormap, range ±{sim['diff_max']:.4f}). "
            "Red = A brighter after convolution; blue = B brighter. "
            "Larger values in fine-detail regions indicate a measurable sharpness difference."
        )
        return f"""
<h3>PSF Simulation — test chart convolved at native pixel resolution</h3>
<p>Each image is rendered at 1 image-pixel : 1 screen-pixel so fine detail differences
are fully visible. Brighter, higher-contrast features indicate a tighter PSF.</p>
{panel(sim['original'], 'Original test chart')}
{panel(sim['conv_a'],   f"Convolved — {sim['label_a']}")}
{panel(sim['conv_b'],   f"Convolved — {sim['label_b']}")}
{panel(sim['diff'],     'Difference (A − B)', diff_caption)}"""

    # ── Section 4: Halo ───────────────────────────────────────────────────────

    def _section_halo(self, ra: AnalysisResult, rb: AnalysisResult,
                       img_a: AstroImage, img_b: AstroImage) -> str:
        err = _error_box("halo", ra, rb)
        ha = ra.halo_metrics or {}
        hb = rb.halo_metrics or {}
        ca, cb = _better_worse_class(ha.get("halo_to_core_ratio"),
                                      hb.get("halo_to_core_ratio"),
                                      higher_is_better=False)
        prof_a = _img_tag((ha.get("figures") or {}).get("halo_profile"), f"Halo {ra.label}")
        prof_b = _img_tag((hb.get("figures") or {}).get("halo_profile"), f"Halo {rb.label}")
        grid_tag = _img_tag(self._plot_halo_star_grid(ra, rb, img_a, img_b),
                            "Halo star comparison grid")

        return f"""
<h2>4. Halo Analysis &nbsp;<span class="metric-label-ok">✓ bandwidth-independent</span></h2>
{err}
<div class="info-box">Halos around bright stars result from internal reflections
within the filter substrate and AR coatings. The halo-to-core ratio measures the
amplitude of the broad halo component relative to the star core. This ratio is
normalised and valid across different filter bandwidths.</div>

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Stars fitted</td><td>{_val(ha.get("n_stars_fitted"), "d")}</td><td>{_val(hb.get("n_stars_fitted"), "d")}</td></tr>
  <tr><td>Halo / core ratio</td><td class="{ca}">{_val(ha.get("halo_to_core_ratio"))}</td><td class="{cb}">{_val(hb.get("halo_to_core_ratio"))}</td></tr>
  <tr><td>Halo radius (px)</td><td>{_val(ha.get("halo_radius_px"))}</td><td>{_val(hb.get("halo_radius_px"))}</td></tr>
</table>

<div style="display:flex;gap:10px;">
  <div style="flex:1;">{prof_a}</div>
  <div style="flex:1;">{prof_b}</div>
</div>
<p class="caption">Radial profiles (semi-log). A steep drop-off indicates a clean
filter. A raised floor or shoulder beyond ~10 px indicates a halo component.</p>

{grid_tag}
<p class="caption">Top-ranked halo stars side-by-side (Image A left, Image B right per
pair). Both cutouts in each pair share the same brightness scale so halo brightness is
directly comparable. Stars sorted by halo/core ratio (highest first). √ stretch applied
to reveal faint halo structure. <em>Inferno</em> colormap: bright = high intensity.</p>

<div class="info-box"><strong>Ideal:</strong> Halo/core ratio &lt; 0.05 is excellent;
&gt; 0.15 indicates significant internal reflection that will reduce contrast on
bright stars.</div>"""

    def _extract_cutout(self, data: np.ndarray,
                         xc: float, yc: float, half: int) -> np.ndarray:
        h, w = data.shape
        x0 = max(0, int(xc) - half)
        x1 = min(w, int(xc) + half + 1)
        y0 = max(0, int(yc) - half)
        y1 = min(h, int(yc) + half + 1)
        return data[y0:y1, x0:x1].copy()

    def _plot_halo_star_grid(self, ra: AnalysisResult, rb: AnalysisResult,
                              img_a: AstroImage, img_b: AstroImage) -> plt.Figure | None:
        stars_a = (ra.halo_metrics or {}).get("star_data", [])
        stars_b = (rb.halo_metrics or {}).get("star_data", [])
        if not stars_a:
            return None

        top_a = stars_a[:20]

        # Nearest-neighbour match in B within 20 px
        matched = []
        if stars_b:
            xs_b = np.array([s["xc"] for s in stars_b])
            ys_b = np.array([s["yc"] for s in stars_b])
            for sa in top_a:
                dists = np.sqrt((xs_b - sa["xc"]) ** 2 + (ys_b - sa["yc"]) ** 2)
                idx = int(np.argmin(dists))
                matched.append((sa, stars_b[idx] if dists[idx] <= 20.0 else None))
        else:
            matched = [(sa, None) for sa in top_a]

        if not matched:
            return None

        bgsub_a = img_a.background_subtracted() if img_a.background is not None else img_a.data
        bgsub_b = img_b.background_subtracted() if img_b.background is not None else img_b.data

        pairs_per_row = 4
        n = len(matched)
        n_rows = (n + pairs_per_row - 1) // pairs_per_row
        n_cols = pairs_per_row * 2

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 2.2, n_rows * 2.8))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        for ax in axes.flat:
            ax.axis("off")

        for idx, (sa, sb) in enumerate(matched):
            row = idx // pairs_per_row
            col_base = (idx % pairs_per_row) * 2

            r_a = sa.get("halo_radius_px") or HALO_FIT_RADIUS_PX
            r_b = (sb.get("halo_radius_px") if sb else r_a) or HALO_FIT_RADIUS_PX
            half = max(int(max(r_a, r_b) * 2.5), HALO_FIT_RADIUS_PX)

            cut_a = self._extract_cutout(bgsub_a, sa["xc"], sa["yc"], half)
            cut_b = (self._extract_cutout(bgsub_b, sb["xc"], sb["yc"], half)
                     if sb is not None else np.zeros_like(cut_a))

            # Shared normalisation: scale to 99.9th-pct peak so core clips, halos visible
            peak_a = float(np.percentile(cut_a, 99.9)) if cut_a.size > 0 else 1.0
            peak_b = float(np.percentile(cut_b, 99.9)) if cut_b.size > 0 else 1.0
            shared_max = max(peak_a, peak_b, 1e-9)

            # Asinh stretch: softening=0.05 boosts faint halo emission (1-20% of peak)
            # to 30-70% of the display range, while the saturated core clips cleanly.
            _soft = 0.05
            _norm = np.arcsinh(1.0 / _soft)
            def _asinh(arr):
                return np.arcsinh(np.clip(arr / shared_max, 0.0, None) / _soft) / _norm

            disp_a = np.clip(_asinh(cut_a), 0.0, 1.0)
            disp_b = np.clip(_asinh(cut_b), 0.0, 1.0)

            ax_a = axes[row, col_base]
            ax_b = axes[row, col_base + 1]

            ax_a.imshow(disp_a, origin="lower", cmap="inferno",
                        vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            h2c_a = sa.get("halo_to_core_ratio")
            ax_a.set_title(f"#{idx+1} {ra.label}"
                           + (f"\nh/c={h2c_a:.3f}" if h2c_a is not None else ""),
                           fontsize=7)
            ax_a.axis("off")

            ax_b.imshow(disp_b, origin="lower", cmap="inferno",
                        vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            if sb is not None:
                h2c_b = sb.get("halo_to_core_ratio")
                ax_b.set_title(f"#{idx+1} {rb.label}"
                               + (f"\nh/c={h2c_b:.3f}" if h2c_b is not None else ""),
                               fontsize=7)
            else:
                ax_b.set_title(f"#{idx+1} {rb.label}\n(no match)", fontsize=7)
            ax_b.axis("off")

        fig.suptitle(
            f"Top {n} halo stars — {ra.label} (left) vs {rb.label} (right) "
            f"per pair  |  shared scale per pair  |  asinh stretch (softening=0.05)",
            fontsize=9,
        )
        fig.tight_layout()
        return fig

    # ── Section 5: Ghost ──────────────────────────────────────────────────────

    def _section_ghost(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        err = _error_box("ghost", ra, rb)
        ga = ra.ghost_metrics or {}
        gb = rb.ghost_metrics or {}
        cands_a = ga.get("ghost_candidates", [])
        cands_b = gb.get("ghost_candidates", [])
        img_a = _img_tag((ga.get("figures") or {}).get("ghost_map"), f"Ghost map {ra.label}")
        img_b = _img_tag((gb.get("figures") or {}).get("ghost_map"), f"Ghost map {rb.label}")

        def cand_rows(cands, label):
            if not cands:
                return f"<tr><td colspan='4'>No ghost candidates detected in {label}</td></tr>"
            top = sorted(cands, key=lambda c: c.get("intensity_ratio", 0), reverse=True)[:25]
            rows = ""
            for c in top:
                rows += (f"<tr><td>{c['separation_px']:.1f}</td>"
                         f"<td>{c['dx']:.1f}, {c['dy']:.1f}</td>"
                         f"<td>{c['intensity_ratio']:.4f}</td>"
                         f"<td>{c['classification']}</td></tr>")
            return rows

        return f"""
<h2>5. Ghost Image Detection &nbsp;<span class="metric-label-ok">✓ bandwidth-independent</span></h2>
{err}
<div class="info-box">Ghosts are discrete secondary images caused by reflections
between the filter surfaces and the sensor. Unlike halos (which are diffuse),
ghosts are localised and appear at specific offsets from bright stars.
The ghost/parent intensity ratio is valid across different bandwidths.</div>

<h3>{ra.label} — {len(cands_a)} candidate(s){" &nbsp;<em>(showing top 25 by intensity ratio)</em>" if len(cands_a) > 25 else ""}</h3>
<table>
  <tr><th>Separation (px)</th><th>Offset (dx, dy)</th><th>Intensity ratio</th><th>Classification</th></tr>
  {cand_rows(cands_a, ra.label)}
</table>
{img_a}

<h3>{rb.label} — {len(cands_b)} candidate(s){" &nbsp;<em>(showing top 25 by intensity ratio)</em>" if len(cands_b) > 25 else ""}</h3>
<table>
  <tr><th>Separation (px)</th><th>Offset (dx, dy)</th><th>Intensity ratio</th><th>Classification</th></tr>
  {cand_rows(cands_b, rb.label)}
</table>
{img_b}"""

    # ── Section 6: Edge ───────────────────────────────────────────────────────

    def _section_edge(self, ra: AnalysisResult, rb: AnalysisResult,
                       bw_differ: bool) -> str:
        err = _error_box("edge", ra, rb)
        ea = ra.edge_metrics or {}
        eb = rb.edge_metrics or {}
        ca, cb = _better_worse_class(ea.get("edge_width_10_90_px"),
                                      eb.get("edge_width_10_90_px"),
                                      higher_is_better=False)
        ecr_warn = (' &nbsp;<span class="metric-label-warn">⚠ bandwidth-sensitive</span>'
                    if bw_differ else "")
        img_a = _img_tag((ea.get("figures") or {}).get("edge"), f"Edge {ra.label}")
        img_b = _img_tag((eb.get("figures") or {}).get("edge"), f"Edge {rb.label}")

        used_sl_a = ea.get("used_starless", False)
        used_sl_b = eb.get("used_starless", False)
        sl_note = ""
        if used_sl_a or used_sl_b:
            who = ", ".join(filter(None, [ra.label if used_sl_a else "",
                                          rb.label if used_sl_b else ""]))
            sl_note = (f'<div class="info-box">★ Edge analysis for <strong>{who}</strong> '
                       f'used the starless image so the strongest gradient search locates '
                       f'a nebula emission boundary rather than a star profile.</div>')

        return f"""
<h2>6. Local Contrast / Edge Analysis</h2>
{err}
{sl_note}
<div class="info-box">The Edge Spread Function (ESF) is extracted across a nebula
emission boundary. Its derivative is the Line Spread Function (LSF). The 10–90%
edge width measures how sharply the transition is rendered — a smaller value indicates
better local contrast and resolution of fine structure.
The normalised ESF shape is <strong>bandwidth-independent ✓</strong>.
The edge contrast ratio (bright/dark side signal) is <strong>bandwidth-sensitive ⚠</strong>.</div>

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Edge width 10–90% (px) ✓</td><td class="{ca}">{_val(ea.get("edge_width_10_90_px"))}</td><td class="{cb}">{_val(eb.get("edge_width_10_90_px"))}</td></tr>
  <tr><td>Edge width 10–90% (arcsec) ✓</td><td>{_val(ea.get("edge_width_10_90_arcsec"))}</td><td>{_val(eb.get("edge_width_10_90_arcsec"))}</td></tr>
  <tr><td>Edge contrast ratio{ecr_warn}</td><td>{_val(ea.get("edge_contrast_ratio"))}</td><td>{_val(eb.get("edge_contrast_ratio"))}</td></tr>
  <tr><td>Gradient magnitude</td><td>{_val(ea.get("gradient_magnitude"), ".2f")}</td><td>{_val(eb.get("gradient_magnitude"), ".2f")}</td></tr>
</table>

{"".join([f'<div style="flex:1;">{img}</div>' for img in [img_a, img_b] if img])}
<div class="info-box"><strong>What to look for:</strong> Both filters should show
similar edge widths if the images are seeing-limited. A filter with poorer substrate
quality may show a broader LSF. The edge contrast ratio may differ legitimately
between bandwidths — a narrower filter rejects more continuum background, which can
increase this ratio even with identical optical quality.</div>"""

    def _plot_radial_overlay(self, ra: AnalysisResult, rb: AnalysisResult) -> plt.Figure | None:
        """Overlay both radial power curves on a single axes."""
        pa = ra.power_metrics or {}
        pb = rb.power_metrics or {}
        freq_a = pa.get("freq_axis")
        rp_a = pa.get("radial_power")
        freq_b = pb.get("freq_axis")
        rp_b = pb.get("radial_power")
        if freq_a is None or rp_a is None or freq_b is None or rp_b is None:
            return None
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(freq_a, rp_a, color="steelblue", linewidth=2, label=ra.label)
        ax.semilogy(freq_b, rp_b, color="tomato", linewidth=2, label=rb.label)
        ax.axvline(0.10, color="gray", linestyle="--", linewidth=0.8,
                   label="Low / mid boundary (0.10 cyc/px)")
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("Radial power (normalised, log scale)")
        ax.set_title("Radial power spectrum — overlay")
        ax.set_xlim(0, 0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        return fig

    # ── Section 7: Power spectrum ──────────────────────────────────────────────

    def _section_power(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        err = _error_box("power", ra, rb)
        pa = ra.power_metrics or {}
        pb = rb.power_metrics or {}
        ca, cb = _better_worse_class(pa.get("mid_high_ratio"), pb.get("mid_high_ratio"))
        img_a = _img_tag((pa.get("figures") or {}).get("power_spectrum"), f"PS {ra.label}")
        img_b = _img_tag((pb.get("figures") or {}).get("power_spectrum"), f"PS {rb.label}")
        img_overlay = _img_tag(self._plot_radial_overlay(ra, rb), "Radial power overlay")

        sl_note = ""
        used_a = pa.get("used_starless", False)
        used_b = pb.get("used_starless", False)
        if used_a or used_b:
            who = ", ".join(filter(None, [ra.label if used_a else "",
                                          rb.label if used_b else ""]))
            sl_note = (f'<div class="info-box">★ Power spectrum for <strong>{who}</strong> '
                       f'was computed on the starless image to reduce star contamination '
                       f'of the spatial frequency content.</div>')

        return f"""
<h2>7. Micro-contrast / Power Spectrum &nbsp;<span class="metric-label-ok">✓ bandwidth-normalised</span></h2>
{err}
{sl_note}
<div class="info-box">The 2D power spectrum of a star-free nebula region reveals the
spatial frequency content of the image. All data is divided by the mean signal before
the FFT, making the result dimensionless and comparable across filters with different
bandwidths. The mid/high-frequency ratio (0.1–0.5 cyc/px vs 0–0.1 cyc/px) measures
fine detail content relative to coarse structure.
<br><strong>Note:</strong> This comparison is only meaningful when both images cover
the same target region.</div>

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Mid/high ratio</td><td class="{ca}">{_val(pa.get("mid_high_ratio"), ".4f")}</td><td class="{cb}">{_val(pb.get("mid_high_ratio"), ".4f")}</td></tr>
</table>

{img_overlay}
<p class="caption">Radial power spectra overlaid (log scale). Curves that diverge at
high frequencies indicate one filter preserves more fine-scale spatial detail. The
dashed line marks the boundary between low (coarse structure) and mid/high frequencies.</p>

<div style="display:flex;gap:10px;">
  <div style="flex:1;">{img_a}</div>
  <div style="flex:1;">{img_b}</div>
</div>"""

    # ── Section 8: Spatial detail ──────────────────────────────────────────────

    def _section_spatial(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        err = _error_box("spatial", ra, rb)
        sm = ra.spatial_metrics or {}
        figs = sm.get("figures", {})

        cr_a = sm.get("contrast_ratios_a", {})
        cr_b = sm.get("contrast_ratios_b", {})

        # Contrast ratio table
        cr_rows = ""
        for ks in sorted(set(list(cr_a.keys()) + list(cr_b.keys()))):
            va = cr_a.get(ks)
            vb = cr_b.get(ks)
            ca, cb = _better_worse_class(va, vb)
            cr_rows += (f"<tr><td>{ks} px</td>"
                        f"<td class='{ca}'>{_val(va)}</td>"
                        f"<td class='{cb}'>{_val(vb)}</td></tr>")

        # Wavelet SNR table
        snr_a = sm.get("wavelet_snr_a", {})
        snr_b = sm.get("wavelet_snr_b", {})
        snr_rows = ""
        for lvl in sorted(set(list(snr_a.keys()) + list(snr_b.keys()))):
            va = snr_a.get(lvl)
            vb = snr_b.get(lvl)
            ca, cb = _better_worse_class(va, vb)
            scale_approx = 2 ** lvl
            snr_rows += (f"<tr><td>Level {lvl} (~{scale_approx}px scale)</td>"
                         f"<td class='{ca}'>{_val(va)}</td>"
                         f"<td class='{cb}'>{_val(vb)}</td></tr>")

        sigma_a = _val(sm.get("sigma_noise_a"), ".5f")
        sigma_b = _val(sm.get("sigma_noise_b"), ".5f")

        def figs_for(prefix):
            out = ""
            for key in sorted(figs):
                if key.startswith(prefix):
                    out += _img_tag(figs[key], key) + "\n"
            return out

        sl_note = ""
        used_a = sm.get("used_starless_a", False)
        used_b = sm.get("used_starless_b", False)
        if used_a or used_b:
            who = ", ".join(filter(None, [ra.label if used_a else "",
                                          rb.label if used_b else ""]))
            sl_note = (f'<div class="info-box">★ Spatial detail analysis for '
                       f'<strong>{who}</strong> used the starless image to reduce '
                       f'star contamination of the spatial frequency maps.</div>')

        return f"""
<h2>8. Spatial Detail Comparison &nbsp;<span class="metric-label-ok">✓ bandwidth-normalised</span></h2>
{err}
{sl_note}
<div class="info-box">All maps below are computed on mean-signal-normalised data
(each image divided by its own mean signal), making them dimensionless and comparable
across different filter bandwidths. Images are shown side-by-side with a shared
colour scale; the third panel shows the difference A−B.</div>

<h3>8a. Local Standard Deviation Maps</h3>
<div class="info-box">Measures how much pixel values vary within a neighbourhood.
Higher values in nebula regions indicate more preserved local detail and contrast.
<strong>Contrast ratio</strong> = median(nebula std) / median(background std);
a higher ratio indicates better differentiation of nebula structure from background.</div>
<table>
  <tr><th>Kernel size</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  {cr_rows}
</table>
{figs_for("std_")}
<p class="caption">Side-by-side local σ maps at each kernel size (shared colour scale).
The difference map (right) highlights where one filter preserves more local variation.</p>

<h3>8b. Laplacian of Gaussian (LoG) Maps</h3>
<div class="info-box">The Laplacian of Gaussian highlights regions of rapid intensity
change at a specific spatial scale (controlled by σ). Brighter regions in |LoG| maps
indicate stronger local curvature — sharper edges and finer nebula filaments.
Smaller σ highlights finer features; larger σ highlights broader structures.</div>
{figs_for("log_")}
<p class="caption">|LoG| maps at σ = 1.5, 3, and 6 px (shared colour scale per row).
A filter preserving more fine detail shows brighter, more defined boundaries at small σ.</p>

<h3>8c. Wavelet Decomposition</h3>
<div class="info-box">A 4-level Daubechies-4 wavelet decomposition separates the
image into spatial scale bands. Level 1 (~2 px) is noise-dominated and used only
for noise estimation. Levels 2–3 carry the most relevant signal for filter comparison.
<strong>SNR</strong> = signal energy / noise energy at each level; SNR &gt; 1
indicates signal-dominated.
Estimated noise (σ): <strong>{ra.label}</strong> = {sigma_a},
<strong>{rb.label}</strong> = {sigma_b} (normalised units)</div>

{_img_tag(figs.get("wavelet_snr"), "Wavelet SNR")}
<p class="caption">Per-level SNR for both filters. Level 1 SNR &lt; 1 is expected
(noise-dominated). A filter preserving more fine detail shows higher SNR at level 2.</p>

<table>
  <tr><th>Wavelet level</th><th>{ra.label} SNR</th><th>{rb.label} SNR</th></tr>
  {snr_rows}
</table>

{figs_for("wavelet_level")}
<p class="caption">Reconstructed detail images at levels 2 and 3 (shared colour scale,
diverging colourmap). The difference panel (right) shows where fine structure differs
between the two filters.</p>"""

    # ── Section 9: Summary ────────────────────────────────────────────────────

    def _section_summary(self, ra: AnalysisResult, rb: AnalysisResult,
                          bw_differ: bool) -> str:
        sm_a = ra.spatial_metrics or {}
        sm_b = rb.spatial_metrics or {}

        def row(metric, val_a, val_b, fmt=".3f",
                higher_is_better=True, bw_flag="✓"):
            ca, cb = _better_worse_class(val_a, val_b, higher_is_better)
            label = f'{metric} <span class="metric-label-ok">{bw_flag}</span>'
            return (f"<tr><td>{label}</td>"
                    f"<td class='{ca}'>{_val(val_a, fmt)}</td>"
                    f"<td class='{cb}'>{_val(val_b, fmt)}</td></tr>")

        psf_a = ra.psf_metrics or {}
        psf_b = rb.psf_metrics or {}
        halo_a = ra.halo_metrics or {}
        halo_b = rb.halo_metrics or {}
        edge_a = ra.edge_metrics or {}
        edge_b = rb.edge_metrics or {}
        pw_a = ra.power_metrics or {}
        pw_b = rb.power_metrics or {}
        cr_a = sm_a.get("contrast_ratios_a", {})
        cr_b = sm_b.get("contrast_ratios_b", {}) if sm_b else {}
        snr_a = sm_a.get("wavelet_snr_a", {})
        snr_b = sm_b.get("wavelet_snr_b", {}) if sm_b else {}

        ecr_flag = "⚠" if bw_differ else "✓"

        rows = "".join([
            row("FWHM (px)", psf_a.get("fwhm_px"), psf_b.get("fwhm_px"),
                higher_is_better=False),
            row("MTF50 (cyc/px)", psf_a.get("mtf50_cycles_per_px"),
                psf_b.get("mtf50_cycles_per_px"), fmt=".4f"),
            row("Halo/core ratio", halo_a.get("halo_to_core_ratio"),
                halo_b.get("halo_to_core_ratio"), higher_is_better=False),
            row("Edge width 10–90% (px)", edge_a.get("edge_width_10_90_px"),
                edge_b.get("edge_width_10_90_px"), higher_is_better=False),
            row(f"Edge contrast ratio", edge_a.get("edge_contrast_ratio"),
                edge_b.get("edge_contrast_ratio"), bw_flag=ecr_flag),
            row("Power mid/high ratio", pw_a.get("mid_high_ratio"),
                pw_b.get("mid_high_ratio"), fmt=".4f"),
            row("Std contrast ratio (15px)", cr_a.get(15), cr_b.get(15)),
            row("Wavelet SNR level 2", snr_a.get(2), snr_b.get(2)),
            row("Wavelet SNR level 3", snr_a.get(3), snr_b.get(3)),
        ])

        legend = ('<p><span class="metric-label-ok">✓</span> = bandwidth-independent '
                  'comparison &nbsp;&nbsp; '
                  '<span class="metric-label-warn">⚠</span> = interpret with bandwidth '
                  'context (filters had different bandwidths)</p>')

        return f"""
<h2>9. Summary &amp; Recommendations</h2>
{legend}
<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  {rows}
</table>
<div class="info-box"><strong>How to read this table:</strong>
Green cells indicate the better value for that metric.
Red cells indicate the worse value. Metrics marked ⚠ may be influenced by the
difference in filter bandwidth and should not be used as the sole basis for
comparison.</div>"""
