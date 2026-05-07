from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint, QSettings, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QGroupBox, QFormLayout, QLineEdit, QFileDialog, QScrollArea,
    QSizePolicy, QRubberBand,
)

from core.astro_image import AstroImage

MAX_DISPLAY_PX = 1024   # max dimension for on-screen display (downsampled for speed)


def _downsample_for_display(arr: np.ndarray) -> np.ndarray:
    """Stride-sample arr so its longest dimension is ≤ MAX_DISPLAY_PX."""
    max_dim = max(arr.shape[:2])
    if max_dim <= MAX_DISPLAY_PX:
        return arr
    step = max_dim // MAX_DISPLAY_PX + 1
    return arr[::step, ::step]


class ZoomableImageLabel(QLabel):
    """QLabel that supports rubber-band ROI selection."""

    roi_selected = pyqtSignal(int, int, int, int)  # x0, y0, x1, y1 in image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 300)
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin = QPoint()
        self._pixmap_orig: QPixmap | None = None
        self._display_arr: np.ndarray | None = None   # keeps buffer alive for QImage
        self._roi_mode = False

    def set_roi_mode(self, enabled: bool) -> None:
        self._roi_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_image_array(self, arr: np.ndarray) -> None:
        arr = _downsample_for_display(arr)
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            self._display_arr = np.ascontiguousarray(arr.astype(np.uint8))
            qimg = QImage(self._display_arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            self._display_arr = np.ascontiguousarray(arr.astype(np.uint8))
            qimg = QImage(self._display_arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self) -> None:
        if self._pixmap_orig is not None:
            scaled = self._pixmap_orig.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.pos()
            self._rubber_band.setGeometry(QRect(self._origin, self._origin))
            self._rubber_band.show()

    def mouseMoveEvent(self, event):
        if self._roi_mode and not self._origin.isNull():
            self._rubber_band.setGeometry(
                QRect(self._origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            self._rubber_band.hide()
            rect = QRect(self._origin, event.pos()).normalized()
            img_rect = self._image_coords(rect)
            if img_rect:
                self.roi_selected.emit(*img_rect)
            self._origin = QPoint()

    def _image_coords(self, widget_rect: QRect) -> tuple[int, int, int, int] | None:
        if self._pixmap_orig is None or self.pixmap() is None:
            return None
        px = self.pixmap()
        lw, lh = self.width(), self.height()
        pw, ph = px.width(), px.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        scale_x = self._pixmap_orig.width() / pw
        scale_y = self._pixmap_orig.height() / ph
        x0 = int((widget_rect.left() - ox) * scale_x)
        y0 = int((widget_rect.top() - oy) * scale_y)
        x1 = int((widget_rect.right() - ox) * scale_x)
        y1 = int((widget_rect.bottom() - oy) * scale_y)
        orig_w = self._pixmap_orig.width()
        orig_h = self._pixmap_orig.height()
        x0 = max(0, min(x0, orig_w))
        y0 = max(0, min(y0, orig_h))
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)


class ImagePanel(QWidget):
    """Left or right image panel: file loading, display, metadata, bandwidth."""

    image_loaded = pyqtSignal(object)   # emits AstroImage
    roi_selected = pyqtSignal(int, int, int, int)

    def __init__(self, title: str = "Image", parent=None):
        super().__init__(parent)
        self._image: AstroImage | None = None
        self._starless_image: AstroImage | None = None
        self._build_ui(title)

    def _build_ui(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title + stretch checkbox + open button
        top = QHBoxLayout()
        top.addWidget(QLabel(f"<b>{title}</b>"))
        top.addStretch()
        self._stretch_cb = QCheckBox("Stretch")
        self._stretch_cb.setChecked(True)
        self._stretch_cb.setToolTip("Auto-stretch display (statistical MTF stretch).\n"
                                    "Uncheck for linear 0.1–99.9% percentile view.")
        self._stretch_cb.toggled.connect(self._refresh_display)
        top.addWidget(self._stretch_cb)
        self._btn_open = QPushButton("Open FITS / XISF…")
        self._btn_open.clicked.connect(self._open_file)
        top.addWidget(self._btn_open)
        layout.addLayout(top)

        # Image display
        self._img_label = ZoomableImageLabel()
        self._img_label.roi_selected.connect(self.roi_selected)
        layout.addWidget(self._img_label, stretch=1)

        # Metadata group — two side-by-side columns to reduce height
        meta_box = QGroupBox("Image info")
        meta_outer = QVBoxLayout(meta_box)
        meta_outer.setContentsMargins(4, 4, 4, 4)

        cols = QHBoxLayout()
        left_form = QFormLayout()
        left_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        right_form = QFormLayout()
        right_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._meta_fields: dict[str, QLabel] = {}
        for key in ["File", "Telescope", "Camera", "Filter", "Exposure"]:
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            left_form.addRow(f"{key}:", lbl)
            self._meta_fields[key] = lbl
        for key in ["Bit depth", "Gain", "Date", "Pixel scale", "Binning"]:
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            right_form.addRow(f"{key}:", lbl)
            self._meta_fields[key] = lbl

        cols.addLayout(left_form)
        cols.addLayout(right_form)
        meta_outer.addLayout(cols)

        # Bandwidth field (editable) — full width below the two columns
        self._bw_edit = QLineEdit()
        self._bw_edit.setPlaceholderText("e.g. 3")
        self._bw_edit.setMaximumWidth(80)
        bw_row = QHBoxLayout()
        bw_row.addWidget(QLabel("Bandwidth:"))
        bw_row.addWidget(self._bw_edit)
        bw_row.addWidget(QLabel("nm"))
        bw_row.addStretch()
        meta_outer.addLayout(bw_row)

        # Starless filename label
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel("Starless:"))
        self._starless_lbl = QLabel("—")
        self._starless_lbl.setStyleSheet("color: #666;")
        sl_row.addWidget(self._starless_lbl)
        sl_row.addStretch()
        meta_outer.addLayout(sl_row)

        layout.addWidget(meta_box)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def image(self) -> AstroImage | None:
        return self._image

    @property
    def starless_image(self) -> AstroImage | None:
        return self._starless_image

    def set_roi_mode(self, enabled: bool) -> None:
        self._img_label.set_roi_mode(enabled)

    def _open_file(self) -> None:
        settings = QSettings("FilterImageComparator", "FilterImageComparator")
        last_dir = settings.value("last_data_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image",
            last_dir,
            "Astronomical images (*.fits *.fit *.fts *.xisf);;All files (*.*)",
        )
        if not path:
            return
        settings.setValue("last_data_dir", str(Path(path).parent))
        img = AstroImage(path)
        try:
            img.load()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load error", str(e))
            return

        self._image = img
        self._starless_image = None
        self._starless_lbl.setText("—")
        self._starless_lbl.setStyleSheet("color: #666;")
        self._populate_metadata(img)
        self._refresh_display()
        self.image_loaded.emit(img)
        self._ask_about_starless(img)

    def _ask_about_starless(self, img: AstroImage) -> None:
        from PyQt6.QtWidgets import QMessageBox
        prefix = ""
        if img.is_color:
            prefix = ("This is a color (RGB) image. It has been converted to "
                      "luminance for analysis.\n\n")
        answer = QMessageBox.question(
            self, "Starless image",
            prefix + "Do you have a starless version of this image?\n\n"
            "If yes, it will be used for power spectrum and spatial detail "
            "analysis to reduce star contamination.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        settings = QSettings("FilterImageComparator", "FilterImageComparator")
        last_dir = settings.value("last_data_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open starless image", last_dir,
            "Astronomical images (*.fits *.fit *.fts *.xisf);;All files (*.*)",
        )
        if not path:
            return
        sl = AstroImage(path)
        try:
            sl.load()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox as _MB
            _MB.critical(self, "Load error", f"Starless load failed:\n{e}")
            return
        self._starless_image = sl
        img.starless_image = sl
        self._starless_lbl.setText(sl.path.name)
        self._starless_lbl.setStyleSheet("color: #155724;")

    def apply_bandwidth_from_field(self) -> None:
        """Push the bandwidth QLineEdit value into the AstroImage."""
        if self._image is None:
            return
        txt = self._bw_edit.text().strip()
        if txt:
            try:
                self._image.bandwidth_nm = float(txt)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _populate_metadata(self, img: AstroImage) -> None:
        fname = Path(img.path).name
        if img.is_color:
            fname += " (color→lum)"
        self._meta_fields["File"].setText(fname)
        for key in ["Bit depth", "Telescope", "Camera", "Filter", "Exposure",
                    "Gain", "Date", "Binning"]:
            val = img.meta.get(key, "—")
            self._meta_fields[key].setText(val)

        scale_txt = f"{img.pixel_scale:.3f} \"/px"
        if img.pixel_scale_is_estimated:
            scale_txt += " (estimated)"
        self._meta_fields["Pixel scale"].setText(scale_txt)

        if img.bandwidth_nm is not None:
            self._bw_edit.setText(str(img.bandwidth_nm))

    def _refresh_display(self) -> None:
        if self._image is None:
            return
        try:
            stretch = self._stretch_cb.isChecked()
            display = self._image.display_image(stretch=stretch)
            self._img_label.set_image_array(display)
        except Exception as exc:
            self._img_label.setText(f"Display error:\n{exc}")
