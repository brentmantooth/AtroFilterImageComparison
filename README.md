# Filter Image Comparator

A Python desktop application for characterizing narrowband astrophotography filters through quantitative image analysis. Load two calibrated images taken through different filters and generate a detailed comparison report covering PSF quality, halo artifacts, ghost images, edge sharpness, spatial frequency content, and multi-scale detail preservation.

---

## Features

| Metric | Description | Bandwidth-independent? |
|--------|-------------|----------------------|
| **PSF / MTF** | Moffat profile fitting, empirical PSF, MTF curve and MTF50 | ✓ Yes |
| **Halo analysis** | Two-component radial profile fit; halo-to-core ratio | ✓ Yes |
| **Ghost detection** | Secondary reflection search around bright stars | ✓ Yes |
| **Edge analysis (LSF)** | Edge Spread Function, 10–90% edge width, Line Spread Function | ✓ Yes (width) / ⚠ (contrast ratio) |
| **Power spectrum** | Signal-normalised 2D FFT, mid/high spatial frequency ratio | ✓ Normalised |
| **Local std maps** | Local standard deviation at 3 kernel scales; contrast ratio metric | ✓ Normalised |
| **Laplacian of Gaussian** | Edge/detail enhancement at 3 spatial scales | ✓ Normalised |
| **Wavelet decomposition** | 4-level Daubechies-4 decomposition; per-level SNR; detail images | ✓ Normalised |

All analysis runs on linear (unstretched) calibrated image data. Display uses an [STF-equivalent statistical stretch](https://www.setiastro.com/statistical-stretch) (SETIAstroSuite MTF method). Images with different filter bandwidths are handled correctly — metrics are clearly labelled as bandwidth-independent or bandwidth-sensitive, and a warning banner appears in the report when bandwidths differ.

---

## Screenshot

> *(Add a screenshot here after first launch)*

---

## Requirements

### Python
Python 3.10+ (tested with Anaconda 3.12.7)

### Conda packages
```bash
conda install -c conda-forge pyqt astropy photutils scipy numpy matplotlib astroalign pillow pywavelets
```

### Pip packages
```bash
pip install xisf
```

`xisf` provides support for PixInsight's native `.xisf` format. All other dependencies are available via conda-forge.

---

## Installation

```bash
git clone https://github.com/<your-username>/AtroFilterImageComparison.git
cd AtroFilterImageComparison

# Install dependencies (see Requirements above)
conda install -c conda-forge pyqt astropy photutils scipy numpy matplotlib astroalign pillow pywavelets
pip install xisf
```

---

## Usage

```bash
cd AtroFilterImageComparison
python FilterImageCompare.py
```

### Workflow

1. **Load images** — Click **Open FITS / XISF…** in each panel to load Image A and Image B. Supported formats: `.fits`, `.fit`, `.fts`, `.xisf`.
2. **Review metadata** — Telescope, camera, filter, exposure, date, and pixel scale are read from the file headers and displayed automatically. Enter the filter bandwidth (nm) manually if not present in the header.
3. **Select metrics** — Check or uncheck the metrics you want to run in the control panel.
4. **Select ROI** *(optional)* — Click **Select ROI…** and draw a rectangle on either image to target a specific nebula region for edge and power spectrum analysis. If no ROI is selected, the app auto-detects the strongest edge and a star-free region automatically.
5. **Set output directory** — Browse to where the HTML report should be saved.
6. **Run** — Click **Run Analysis**. Images are aligned automatically using `astroalign` before per-pixel comparisons. Progress is shown in the status bar.
7. **Review report** — The HTML report opens automatically in your default browser when analysis completes.

---

## Output Report

The report is a self-contained HTML file (all plots embedded as base64 PNG) saved to your chosen output directory. It contains nine sections:

1. **Image metadata** — Side-by-side header info for both filters; bandwidth warning banner if bandwidths differ
2. **Observation context** — Seeing warning if FWHM > 3″; notes on valid comparison conditions
3. **PSF / MTF** — FWHM, Moffat β, ellipticity, MTF50, MTF at Nyquist; overlaid MTF curves; ePSF images
4. **Halo analysis** — Halo-to-core ratio, halo radius; side-by-side semi-log radial profiles
5. **Ghost detection** — Candidate table (separation, intensity ratio, classification); annotated image
6. **Edge analysis** — 10–90% edge width in pixels and arcseconds; ESF and LSF plots; edge contrast ratio (flagged ⚠ if bandwidths differ)
7. **Power spectrum** — Signal-normalised 2D power spectrum; radial power comparison; mid/high ratio
8. **Spatial detail** — Local σ maps (3 scales), |LoG| maps (3 scales), wavelet detail images and SNR bar chart
9. **Summary table** — All scalar metrics side by side; better value highlighted green, worse value highlighted red

---

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| FITS | `.fits` `.fit` `.fts` | Standard calibrated output from all major acquisition software |
| XISF | `.xisf` | PixInsight native format; requires `pip install xisf` |

Images should be **calibrated and stacked** (bias/dark/flat corrected) but **not stretched**. Linear data is required for valid metric calculations.

---

## On-Sky vs Bench Testing

This tool is designed for **on-sky images**, not optical bench tests. Several important caveats apply:

- **Seeing is the dominant PSF contribution** on most nights. PSF/MTF comparisons between filters are most meaningful when both images were captured on the same night under similar atmospheric conditions.
- The app flags `seeing_dominated = True` and adds a warning in the report when FWHM exceeds 3″.
- **Halo, ghost, edge width, and spatial detail metrics** are less sensitive to seeing and are more reliably attributable to filter differences.
- **Astroalign** is used to register Image A onto the coordinate frame of Image B before any per-pixel comparison metrics are computed.

---

## Bandwidth Validity

Filters with different bandwidths (e.g., 3 nm vs 7 nm) produce different absolute ADU levels. The app handles this systematically:

**Bandwidth-independent metrics** (ratio or normalised — valid as-is):
- PSF FWHM and MTF (normalised PSF shape)
- Halo-to-core ratio and ghost-to-parent ratio
- Edge 10–90% width (normalised ESF)
- Local std contrast ratio, LoG maps, wavelet SNR (all mean-signal normalised)
- Power spectrum mid/high ratio (mean-signal normalised before FFT)

**Bandwidth-sensitive metrics** (flagged ⚠ in the report):
- Edge contrast ratio (bright/dark side signal; affected by background level)

When filter bandwidths differ, a banner appears at the top of the report, and each sensitive metric carries an explanatory note.

---

## Project Structure

```
FilterImageCompare.py       # Entry point
requirements.txt
core/
  models.py                 # Constants, AnalysisResult dataclass
  astro_image.py            # FITS/XISF loading, background estimation, statistical stretch
analysis/
  star_catalog.py           # DAOStarFinder + isolation filtering
  psf_analyzer.py           # Moffat fitting, ePSF builder, MTF via FFT
  halo_analyzer.py          # Radial profile extraction, two-component Moffat fit
  ghost_detector.py         # Secondary source search in annular regions
  edge_analyzer.py          # Sobel edge detection, ESF/LSF extraction
  power_spectrum.py         # Signal-normalised 2D FFT and radial average
  image_filters.py          # Local std maps, LoG maps, wavelet decomposition
report/
  report_builder.py         # Self-contained HTML report generator
gui/
  image_panel.py            # PyQt5 image display with ROI rubber-band selection
  control_panel.py          # Metric checkboxes, parameters, output directory
  analysis_thread.py        # QThread orchestrator; runs all engines off the main thread
  main_window.py            # QMainWindow; assembles panels, menu, signal wiring
```

---

## Key Dependencies and Acknowledgements

| Library | Purpose |
|---------|---------|
| [astropy](https://www.astropy.org/) | FITS I/O, Moffat2D model, Background2D |
| [photutils](https://photutils.readthedocs.io/) | DAOStarFinder, EPSFBuilder, morphology |
| [scipy](https://scipy.org/) | Optimisation, FFT, image filters |
| [PyWavelets](https://pywavelets.readthedocs.io/) | Daubechies-4 wavelet decomposition |
| [astroalign](https://astroalign.quatrope.org/) | Image registration |
| [xisf](https://pypi.org/project/xisf/) | PixInsight XISF format support |
| [PyQt5](https://riverbankcomputing.com/software/pyqt/) | GUI framework |
| [matplotlib](https://matplotlib.org/) | All plots and figures |

Statistical stretch algorithm adapted from [SETIAstroSuite](https://www.setiastro.com/statistical-stretch) (MTF-based autostretch, equivalent to PixInsight STF).

Wavelet noise estimation uses the robust MAD estimator from Donoho & Johnstone (1994).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
