from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, gaussian_laplace, zoom
import pywt

from core.astro_image import AstroImage
from core.models import STD_KERNEL_SIZES, LOG_SIGMAS, WAVELET_NAME, WAVELET_LEVELS

MAX_DIM_FOR_STD = 2048   # downsample to this before generic_filter (performance)


class SpatialDetailAnalyzer:
    """Multi-scale spatial detail comparison: local std, LoG, and wavelet.

    Takes both images simultaneously and produces side-by-side comparison
    figures. All processing uses mean-signal-normalised data.
    """

    def analyze(self, image_a: AstroImage, image_b: AstroImage,
                kernel_sizes: tuple = STD_KERNEL_SIZES,
                log_sigmas: tuple = LOG_SIGMAS,
                wavelet: str = WAVELET_NAME,
                levels: int = WAVELET_LEVELS) -> dict:

        image_a.estimate_background()
        image_b.estimate_background()

        norm_a = self._normalise(image_a)
        norm_b = self._normalise(image_b)

        result: dict = {
            "contrast_ratios_a": {},
            "contrast_ratios_b": {},
            "wavelet_snr_a": {},
            "wavelet_snr_b": {},
            "sigma_noise_a": None,
            "sigma_noise_b": None,
        }
        figures: dict = {}

        if norm_a is None or norm_b is None:
            result["warning"] = "Mean signal ≤ 0 in one or both images; spatial analysis skipped."
            return result

        # Nebula / background masks (used for contrast ratio)
        mask_neb_a, mask_bg_a = self._make_masks(image_a)
        mask_neb_b, mask_bg_b = self._make_masks(image_b)

        # 1. Local standard deviation maps
        std_figs = self._std_analysis(
            norm_a, norm_b,
            mask_neb_a, mask_bg_a,
            mask_neb_b, mask_bg_b,
            kernel_sizes,
            image_a.label, image_b.label,
            result,
        )
        figures.update(std_figs)

        # 2. Laplacian of Gaussian maps
        log_figs = self._log_analysis(
            norm_a, norm_b, log_sigmas,
            image_a.label, image_b.label,
        )
        figures.update(log_figs)

        # 3. Wavelet decomposition
        wav_figs = self._wavelet_analysis(
            norm_a, norm_b, wavelet, levels,
            image_a.label, image_b.label,
            result,
        )
        figures.update(wav_figs)

        result["figures"] = figures
        return result

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _normalise(self, image: AstroImage) -> np.ndarray | None:
        bgsub = image.background_subtracted()
        positive = bgsub[bgsub > 0]
        if positive.size == 0:
            return None
        mean_signal = float(np.mean(positive))
        if mean_signal <= 0:
            return None
        return bgsub / mean_signal

    def _make_masks(self, image: AstroImage) -> tuple[np.ndarray, np.ndarray]:
        rms = image.background_rms
        if rms is None:
            rms_val = float(np.std(image.background_subtracted()))
        else:
            rms_val = float(np.median(rms))

        bgsub = image.background_subtracted()
        nebula_mask = bgsub > 2.0 * rms_val
        bg_mask = bgsub < 0.5 * rms_val

        # Fallback: use top-5% as nebula if no pixels pass threshold
        if not np.any(nebula_mask):
            threshold = np.percentile(bgsub, 95)
            nebula_mask = bgsub >= threshold

        return nebula_mask, bg_mask

    # ------------------------------------------------------------------
    # Local standard deviation maps
    # ------------------------------------------------------------------

    def _std_analysis(self, norm_a, norm_b,
                       mask_neb_a, mask_bg_a,
                       mask_neb_b, mask_bg_b,
                       kernel_sizes, label_a, label_b,
                       result: dict) -> dict:
        figures = {}
        for ks in kernel_sizes:
            std_a = self._compute_std_map(norm_a, ks)
            std_b = self._compute_std_map(norm_b, ks)

            # Contrast ratios
            cr_a = self._contrast_ratio(std_a, mask_neb_a, mask_bg_a)
            cr_b = self._contrast_ratio(std_b, mask_neb_b, mask_bg_b)
            result["contrast_ratios_a"][ks] = cr_a
            result["contrast_ratios_b"][ks] = cr_b

            fig = self._plot_side_by_side(
                std_a, std_b,
                f"Local σ — kernel {ks}px — {label_a}",
                f"Local σ — kernel {ks}px — {label_b}",
                diff_title=f"Diff (A−B), kernel {ks}px",
                cmap="viridis",
                nonlinear_norm=True,
            )
            figures[f"std_{ks}px"] = fig

        return figures

    def _compute_std_map(self, norm: np.ndarray, kernel_size: int) -> np.ndarray:
        # Downsample for performance if image is large
        factor = 1.0
        data = norm
        if max(norm.shape) > MAX_DIM_FOR_STD:
            factor = MAX_DIM_FOR_STD / max(norm.shape)
            zy = factor * norm.shape[0] / norm.shape[0]
            new_h = int(norm.shape[0] * factor)
            new_w = int(norm.shape[1] * factor)
            data = zoom(norm, (new_h / norm.shape[0], new_w / norm.shape[1]), order=1)
            kernel_size = max(3, int(kernel_size * factor) | 1)

        std_map = generic_filter(data, np.std, size=kernel_size)

        if factor < 1.0:
            std_map = zoom(std_map,
                           (norm.shape[0] / std_map.shape[0],
                            norm.shape[1] / std_map.shape[1]),
                           order=1)
        return std_map

    def _contrast_ratio(self, std_map: np.ndarray,
                         nebula_mask: np.ndarray,
                         bg_mask: np.ndarray) -> float | None:
        neb_vals = std_map[nebula_mask[:std_map.shape[0], :std_map.shape[1]]]
        bg_vals = std_map[bg_mask[:std_map.shape[0], :std_map.shape[1]]]
        if neb_vals.size == 0 or bg_vals.size == 0:
            return None
        bg_med = float(np.median(bg_vals))
        if bg_med <= 0:
            return None
        return float(np.median(neb_vals)) / bg_med

    # ------------------------------------------------------------------
    # Laplacian of Gaussian maps
    # ------------------------------------------------------------------

    def _log_analysis(self, norm_a, norm_b, sigmas,
                       label_a, label_b) -> dict:
        figures = {}
        for sigma in sigmas:
            log_a = np.abs(gaussian_laplace(norm_a, sigma=sigma))
            log_b = np.abs(gaussian_laplace(norm_b, sigma=sigma))
            fig = self._plot_side_by_side(
                log_a, log_b,
                f"|LoG| σ={sigma}px — {label_a}",
                f"|LoG| σ={sigma}px — {label_b}",
                diff_title=f"LoG diff (A−B), σ={sigma}px",
                cmap="hot",
                nonlinear_norm=True,
            )
            figures[f"log_sigma{sigma}"] = fig
        return figures

    # ------------------------------------------------------------------
    # Wavelet decomposition
    # ------------------------------------------------------------------

    def _wavelet_analysis(self, norm_a, norm_b, wavelet, levels,
                           label_a, label_b, result: dict) -> dict:
        figures = {}

        coeffs_a = pywt.wavedec2(norm_a, wavelet, level=levels,
                                   mode="periodization")
        coeffs_b = pywt.wavedec2(norm_b, wavelet, level=levels,
                                   mode="periodization")

        sigma_a = self._estimate_noise(coeffs_a)
        sigma_b = self._estimate_noise(coeffs_b)
        result["sigma_noise_a"] = sigma_a
        result["sigma_noise_b"] = sigma_b

        # Per-level SNR (index 1 = coarsest detail, -1 = finest detail)
        for lvl_idx in range(1, levels + 1):
            # coeffs layout: [approx, detail_coarsest, ..., detail_finest]
            # level 1 = finest = coeffs[-1]; level N = coarsest = coeffs[1]
            coeff_idx = levels + 1 - lvl_idx  # map human level to list index
            snr_a = self._level_snr(coeffs_a, coeff_idx, sigma_a, lvl_idx)
            snr_b = self._level_snr(coeffs_b, coeff_idx, sigma_b, lvl_idx)
            result["wavelet_snr_a"][lvl_idx] = snr_a
            result["wavelet_snr_b"][lvl_idx] = snr_b

        # SNR bar chart
        figures["wavelet_snr"] = self._plot_snr_bars(
            result["wavelet_snr_a"], result["wavelet_snr_b"], label_a, label_b, levels)

        # Reconstruct and display levels 2 and 3 (best signal content)
        for display_level in [2, 3]:
            if display_level > levels:
                continue
            coeff_idx = levels + 1 - display_level
            rec_a = self._reconstruct_level(coeffs_a, coeff_idx, wavelet, levels)
            rec_b = self._reconstruct_level(coeffs_b, coeff_idx, wavelet, levels)
            fig = self._plot_side_by_side(
                rec_a, rec_b,
                f"Wavelet level {display_level} — {label_a}",
                f"Wavelet level {display_level} — {label_b}",
                diff_title=f"Level {display_level} diff (A−B)",
                cmap="RdBu_r",
                symmetric_diff=True,
            )
            figures[f"wavelet_level{display_level}"] = fig

        return figures

    def _estimate_noise(self, coeffs) -> float:
        # Finest-level horizontal detail (last element, first sub-band)
        lh1 = coeffs[-1][0]
        return float(np.median(np.abs(lh1))) / 0.6745

    def _level_snr(self, coeffs, coeff_idx: int,
                    sigma_noise: float, human_level: int) -> float | None:
        if coeff_idx < 1 or coeff_idx >= len(coeffs):
            return None
        lh, hl, hh = coeffs[coeff_idx]
        signal_energy = float(np.sum(lh**2 + hl**2 + hh**2))
        # Noise amplifies by sqrt(3) * 2^(level/2) at each wavelet level
        noise_amp = sigma_noise * np.sqrt(3.0) * (2.0 ** (human_level / 2.0))
        noise_energy = (noise_amp ** 2) * lh.size
        if noise_energy <= 0:
            return None
        return signal_energy / noise_energy

    def _reconstruct_level(self, coeffs, target_coeff_idx: int,
                             wavelet: str, levels: int) -> np.ndarray:
        zeroed = [np.zeros_like(coeffs[0])]
        for i, detail in enumerate(coeffs[1:], start=1):
            if i == target_coeff_idx:
                zeroed.append(detail)
            else:
                zeroed.append(tuple(np.zeros_like(d) for d in detail))
        return pywt.waverec2(zeroed, wavelet, mode='periodization')

    # ------------------------------------------------------------------
    # Shared plotting helper
    # ------------------------------------------------------------------

    def _plot_side_by_side(self, arr_a: np.ndarray, arr_b: np.ndarray,
                            title_a: str, title_b: str,
                            diff_title: str = "",
                            cmap: str = "viridis",
                            symmetric_diff: bool = False,
                            nonlinear_norm: bool = False) -> plt.Figure:
        # Shared color scale: use percentile clipping to prevent bright outliers
        # (stars, hot pixels) from compressing the interesting nebula detail range.
        vmin = max(0.0, float(min(np.percentile(arr_a, 0.5), np.percentile(arr_b, 0.5))))
        vmax = float(max(np.percentile(arr_a, 99.5), np.percentile(arr_b, 99.5)))
        if vmax <= vmin:
            vmax = vmin + 1e-9

        # Sqrt (PowerNorm gamma=0.5) compresses bright stars, reveals faint nebula
        norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax) if nonlinear_norm else None

        diff = arr_a[:min(arr_a.shape[0], arr_b.shape[0]),
                      :min(arr_a.shape[1], arr_b.shape[1])] - \
               arr_b[:min(arr_a.shape[0], arr_b.shape[0]),
                      :min(arr_a.shape[1], arr_b.shape[1])]

        if symmetric_diff:
            # Percentile-based symmetric scale so sparse large-diff pixels don't crush detail
            d_max = float(np.percentile(np.abs(diff), 99.5)) or 1.0
            dvmin, dvmax = -d_max, d_max
        else:
            dvmin = float(np.percentile(diff, 0.5))
            dvmax = float(np.percentile(diff, 99.5))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, arr, title in zip(axes[:2], [arr_a, arr_b], [title_a, title_b]):
            im = ax.imshow(arr, origin="lower", cmap=cmap,
                           norm=norm if norm is not None else None,
                           vmin=None if norm is not None else vmin,
                           vmax=None if norm is not None else vmax,
                           interpolation="nearest", aspect="auto")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        im_diff = axes[2].imshow(diff, origin="lower", cmap="RdBu_r",
                                   vmin=dvmin, vmax=dvmax,
                                   interpolation="nearest", aspect="auto")
        axes[2].set_title(diff_title, fontsize=9)
        axes[2].axis("off")
        plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        return fig

    def _plot_snr_bars(self, snr_a: dict, snr_b: dict,
                        label_a: str, label_b: str, levels: int) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, levels + 1)
        width = 0.35
        vals_a = [snr_a.get(lvl) or 0.0 for lvl in x]
        vals_b = [snr_b.get(lvl) or 0.0 for lvl in x]
        bars_a = ax.bar(x - width / 2, vals_a, width, label=label_a, color="steelblue")
        bars_b = ax.bar(x + width / 2, vals_b, width, label=label_b, color="tomato")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                   label="SNR = 1 (signal = noise)")
        ax.set_xlabel("Wavelet level (1 = finest ~2px, 4 = coarsest ~16px)")
        ax.set_ylabel("Signal energy / Noise energy")
        ax.set_title("Wavelet per-level SNR comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Level {i}" for i in x])
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig
