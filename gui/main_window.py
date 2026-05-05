from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMessageBox, QAction, QFileDialog,
)

from gui.image_panel import ImagePanel
from gui.control_panel import AnalysisControlPanel
from gui.analysis_thread import AnalysisThread


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Image Comparator")
        self.resize(1200, 800)

        self._thread: AnalysisThread | None = None
        self._roi: tuple | None = None

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Image panels side by side
        splitter = QSplitter(Qt.Horizontal)
        self._panel_a = ImagePanel("Image A")
        self._panel_b = ImagePanel("Image B")
        splitter.addWidget(self._panel_a)
        splitter.addWidget(self._panel_b)
        splitter.setSizes([600, 600])
        main_layout.addWidget(splitter, stretch=1)

        # Control panel below images
        self._control = AnalysisControlPanel()
        self._control.setMaximumHeight(200)
        main_layout.addWidget(self._control)

        # Wire signals
        self._panel_a.image_loaded.connect(self._on_image_loaded)
        self._panel_b.image_loaded.connect(self._on_image_loaded)
        self._panel_a.roi_selected.connect(self._on_roi_selected)
        self._panel_b.roi_selected.connect(self._on_roi_selected)

        self._control.run_requested.connect(self._on_run)
        self._control.roi_mode_toggled.connect(self._on_roi_mode_toggled)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        act_open_a = QAction("Open Image &A…", self)
        act_open_a.triggered.connect(self._panel_a._open_file)
        file_menu.addAction(act_open_a)

        act_open_b = QAction("Open Image &B…", self)
        act_open_b.triggered.connect(self._panel_b._open_file)
        file_menu.addAction(act_open_b)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        analysis_menu = mb.addMenu("&Analysis")
        act_run = QAction("&Run Analysis", self)
        act_run.triggered.connect(lambda: self._control._on_run())
        analysis_menu.addAction(act_run)

        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_image_loaded(self, img) -> None:
        both_loaded = (self._panel_a.image is not None and
                       self._panel_b.image is not None)
        self._control.set_run_enabled(both_loaded)
        if both_loaded:
            self._control.set_alignment_status("Waiting for analysis…", ok=True)

    def _on_roi_mode_toggled(self, enabled: bool) -> None:
        self._panel_a.set_roi_mode(enabled)
        self._panel_b.set_roi_mode(enabled)
        if not enabled:
            self._control.set_roi(self._roi)

    def _on_roi_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._roi = (x0, y0, x1, y1)
        self._control.set_roi(self._roi)
        # Turn off ROI mode automatically after selection
        self._control._roi_btn.setChecked(False)
        self._on_roi_mode_toggled(False)

    def _on_run(self, settings: dict) -> None:
        img_a = self._panel_a.image
        img_b = self._panel_b.image
        if img_a is None or img_b is None:
            QMessageBox.warning(self, "Missing images",
                                "Please load both Image A and Image B before running.")
            self._control.set_run_enabled(True)
            return

        # Push bandwidth values from the input fields into the image objects
        self._panel_a.apply_bandwidth_from_field()
        self._panel_b.apply_bandwidth_from_field()

        # Merge ROI from the window state into settings
        settings["roi"] = self._roi

        self._thread = AnalysisThread(img_a, img_b, settings, parent=self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._control.update_progress(pct, msg)
        if "align" in msg.lower():
            ok = "fail" not in msg.lower()
            self._control.set_alignment_status(
                "Aligned ✓" if ok else "Alignment failed", ok=ok)

    def _on_finished(self, result_a, result_b, report_path: str) -> None:
        self._control.reset_progress()
        self._control.set_run_enabled(True)
        msg = "Analysis complete."
        if report_path:
            msg += f"\nReport saved to:\n{report_path}"
        QMessageBox.information(self, "Done", msg)

    def _on_error(self, msg: str) -> None:
        self._control.reset_progress()
        self._control.set_run_enabled(True)
        QMessageBox.critical(self, "Analysis error", msg)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Filter Image Comparator",
            "<b>Filter Image Comparator</b><br>"
            "Astrophotography narrowband filter characterisation tool.<br><br>"
            "Metrics: PSF/MTF · Halo · Ghost · Edge · Power spectrum · "
            "Spatial detail (std / LoG / wavelet)<br><br>"
            "Supports FITS and XISF input formats.",
        )
