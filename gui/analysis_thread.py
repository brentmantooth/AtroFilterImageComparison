from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from core.astro_image import AstroImage
from core.models import AnalysisResult
from analysis.psf_analyzer import PSFAnalyzer
from analysis.halo_analyzer import HaloAnalyzer
from analysis.ghost_detector import GhostDetector
from analysis.edge_analyzer import EdgeAnalyzer
from analysis.power_spectrum import PowerSpectrumAnalyzer
from analysis.image_filters import SpatialDetailAnalyzer
from report.report_builder import ReportBuilder


class AnalysisThread(QThread):
    """Runs all selected analysis engines off the main thread."""

    progress = pyqtSignal(int, str)               # (percent, status_text)
    finished = pyqtSignal(object, object, str)    # (result_a, result_b, report_path)
    error = pyqtSignal(str)

    def __init__(self, image_a: AstroImage, image_b: AstroImage,
                 settings: dict, parent=None):
        super().__init__(parent)
        self._image_a = image_a
        self._image_b = image_b
        self._settings = settings

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            self.error.emit(str(exc))

    def _execute(self) -> None:
        img_a = self._image_a
        img_b = self._image_b
        s = self._settings
        metrics = s.get("metrics", {})
        roi = s.get("roi")

        # Apply pixel scale override if provided
        pso = s.get("pixel_scale_override")
        if pso:
            img_a.pixel_scale = pso
            img_b.pixel_scale = pso

        result_a = AnalysisResult(label=img_a.label)
        result_b = AnalysisResult(label=img_b.label)

        # Alignment (always attempted; errors are non-fatal)
        self.progress.emit(2, "Aligning images…")
        aligned = self._align(img_a, img_b, result_a)

        step = 0
        total_steps = sum(1 for v in metrics.values() if v)
        total_steps += 1  # report generation

        def advance(msg: str) -> None:
            nonlocal step
            step += 1
            pct = int(step / (total_steps + 1) * 95)
            self.progress.emit(pct, msg)

        if metrics.get("psf"):
            advance("Computing PSF / MTF…")
            analyzer = PSFAnalyzer()
            result_a.psf_metrics = analyzer.analyze(img_a)
            result_b.psf_metrics = analyzer.analyze(img_b)

        if metrics.get("halo"):
            advance("Analysing halos…")
            analyzer = HaloAnalyzer()
            result_a.halo_metrics = analyzer.analyze(img_a)
            result_b.halo_metrics = analyzer.analyze(img_b)

        if metrics.get("ghost"):
            advance("Searching for ghosts…")
            fwhm_a = (result_a.psf_metrics or {}).get("fwhm_px") or 4.0
            fwhm_b = (result_b.psf_metrics or {}).get("fwhm_px") or 4.0
            det = GhostDetector()
            result_a.ghost_metrics = det.analyze(img_a, psf_fwhm_px=fwhm_a)
            result_b.ghost_metrics = det.analyze(img_b, psf_fwhm_px=fwhm_b)

        if metrics.get("edge"):
            advance("Extracting edge spread function…")
            ea = EdgeAnalyzer()
            result_a.edge_metrics = ea.analyze(img_a, roi=roi)
            result_b.edge_metrics = ea.analyze(img_b, roi=roi)

        if metrics.get("power"):
            advance("Computing power spectrum…")
            pa = PowerSpectrumAnalyzer()
            result_a.power_metrics = pa.analyze(img_a, roi=roi)
            result_b.power_metrics = pa.analyze(img_b, roi=roi)

        if metrics.get("spatial"):
            advance("Running spatial detail analysis…")
            wavelet_levels = s.get("wavelet_levels", 4)
            sda = SpatialDetailAnalyzer()
            spatial = sda.analyze(img_a, img_b, levels=wavelet_levels)
            result_a.spatial_metrics = spatial
            result_b.spatial_metrics = spatial  # shared reference

        # Generate report
        advance("Generating HTML report…")
        report_path = ""
        try:
            builder = ReportBuilder()
            out = builder.generate(
                img_a, img_b, result_a, result_b,
                output_dir=s.get("output_dir", "."),
                open_browser=True,
            )
            report_path = str(out)
        except Exception as e:
            result_a.warnings.append(f"Report generation failed: {e}")

        self.progress.emit(100, "Done")
        self.finished.emit(result_a, result_b, report_path)

    # ------------------------------------------------------------------
    # Image alignment
    # ------------------------------------------------------------------

    def _align(self, img_a: AstroImage, img_b: AstroImage,
                result: AnalysisResult) -> bool:
        try:
            import astroalign as aa
            aligned_data, _ = aa.register(img_a.data, img_b.data)
            img_a.data = aligned_data
            return True
        except Exception as e:
            result.warnings.append(f"Alignment failed ({e}); proceeding unaligned.")
            return False
