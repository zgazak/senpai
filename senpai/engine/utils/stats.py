"""Statistics helpers shared by detection stages."""

import os

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy import fft as _scipy_fft


def fft_workers():
    """Context manager letting scipy FFT-based ops use all cores.

    scipy.fft defaults to a single worker, so scipy.signal.convolve /
    fftconvolve on full frames spend their time in single-threaded FFTs;
    wrapping the call sites multiplies nothing but threads — values are
    identical.
    """
    return _scipy_fft.set_workers(os.cpu_count() or 1)


def robust_background_stats(
    image: np.ndarray,
    *,
    sigma: float = 3.0,
    maxiters: int = 5,
    target_npix: int = 1_000_000,
) -> tuple[float, float, float]:
    """Sigma-clipped (mean, median, std) estimated on a strided subsample.

    Full-frame ``sigma_clipped_stats`` on a 66-Mpix float64 frame costs
    several seconds per call and was the single largest cost in the
    detection stack. A uniform stride still covers the whole frame, so
    spatial structure (vignetting, gradients, dense regions) is sampled
    fairly; on real 8k frames the 3-sigma detection threshold moves by
    <0.01 sigma vs the full computation. ``target_npix`` keeps the
    subsample large enough (~1e6 px) that the clipped std is stable to
    ~0.1%. Images at or below ``target_npix`` are used in full, so small
    frames (and unit tests) are bit-identical to the unstrided result.
    """
    stride = max(1, int(np.sqrt(image.size / target_npix)))
    if stride > 1:
        image = image[::stride, ::stride]
    return sigma_clipped_stats(image, sigma=sigma, maxiters=maxiters)
