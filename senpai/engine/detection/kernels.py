import functools
import logging

import cv2
import numpy as np
from scipy.ndimage import shift

logger = logging.getLogger(__name__)


def rotate_pil(array, angle):
    """Bilinear rotation with an expanded bounding box (PIL semantics).

    Implemented with cv2.warpAffine: ~20x faster than PIL on the 100x
    supersampled kernel intermediates (which dominate kernel-build cost at
    ~0.3 s each), with final-kernel differences <0.5% of amplitude confined
    to edge pixels — sub-resolution placement noise on normalized
    matched-filter kernels.
    """
    h, w = array.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(np.ceil(h * sin_a + w * cos_a))
    new_h = int(np.ceil(h * cos_a + w * sin_a))
    matrix[0, 2] += (new_w - 1) / 2.0 - (w - 1) / 2.0
    matrix[1, 2] += (new_h - 1) / 2.0 - (h - 1) / 2.0
    return cv2.warpAffine(
        np.ascontiguousarray(array, dtype=np.float32),
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def shift_filter_subpx(filter, pix_shift):
    pad = (pix_shift + 0.5).round().astype(int)

    padded = np.pad(filter, ((pad[0], pad[0]), (pad[1], pad[1])))
    shifted = shift(padded, pix_shift)

    shifted[np.where(np.abs(shifted) < 1e-4)] = 0.000
    shifted[np.where(shifted < 0)] = 0.001
    shifted[np.where(shifted > 1)] = 1

    return shifted


# maxsize must hold a full seed-search sweep (~19 kernels) plus the
# per-frame detection/mask kernels across concurrent track rates; at 32 the
# seed sweep alone evicted itself every frame and rebuilt ~1.7s of kernels.
@functools.lru_cache(maxsize=256)
def rectangle_pyramoid(
    length: float,
    sinx: float,
    cosx: float,
    width: int = 4,
    upsample: int = 100,
    pix_shift: tuple[float, float] | None = None,
    halo_fwhm: float | None = None,
    halo_level: float = 1e-3,
    verbose: bool = False,
):
    if verbose:
        logger.info("rectangle_pyramoid")

    angle = np.rad2deg(np.arctan2(sinx, cosx))

    width = int(width)
    length = int(length)

    # Bound the supersampled intermediate. The kernel is built at `upsample`x
    # resolution then PIL-rotated with expand=1, so the rotated bounding box
    # grows as ~(max(width, length) * upsample)^2. At upsample=100 a long
    # streak is catastrophic: L=600 builds a ~60000^2 float array (~15 GB),
    # and a fast coverage target (L~800-1000) reached ~46 GB and drew the OOM
    # killer (burr _full7). The 100x supersampling exists for sub-pixel
    # accuracy of the streak *edges*; a long streak does not need 100 samples
    # across its length. Cap the largest upsampled dimension so the
    # intermediate stays bounded (~MAX_UPSAMPLED_DIM^2) regardless of length —
    # the final resized kernel keeps the same pixel dimensions either way.
    MAX_UPSAMPLED_DIM = 6000
    longest = max(width, length, 1)
    upsample = max(1, min(upsample, MAX_UPSAMPLED_DIM // longest))

    # float32 from the start: the rotation already worked in float32 (PIL
    # mode 'F' before, cv2 now), so this only avoids building and padding
    # the supersampled intermediate at double width.
    pyramid = np.ones((width * upsample, length * upsample), dtype=np.float32)
    if verbose:
        logger.info("built base streak")

    if halo_fwhm is not None:
        if verbose:
            logger.info("adding halo")

        halo_fwhm = int(halo_fwhm / 2)
        # logger.info("adding nonzero halo")
        pyramid = np.pad(
            pyramid,
            (
                (halo_fwhm * upsample, halo_fwhm * upsample),
                (halo_fwhm * upsample, halo_fwhm * upsample),
            ),
            mode="constant",
            constant_values=0.0,
        )

        if verbose:
            logger.info("padded pyramid")

        pyramid2 = np.full(pyramid.shape, halo_level, dtype=np.float32)

        if verbose:
            logger.info("created pyramid2")

        pyramid2 = rotate_pil(pyramid2, -angle)

        if verbose:
            logger.info("rotated pyramid2")

    pyramid = rotate_pil(pyramid, -angle)
    if verbose:
        logger.info("rotated pyramid")

    if halo_fwhm is not None:
        # add nonzero halo to original pyramid
        pyramid[np.where(pyramid == 0)] = pyramid2[np.where(pyramid == 0)]
        if verbose:
            logger.info("added halo")

    pyramid = cv2.resize(
        pyramid,
        dsize=(int(pyramid.shape[1] / upsample), int(pyramid.shape[0] / upsample)),
        interpolation=cv2.INTER_AREA,
    )
    if verbose:
        logger.info("resized pyramid")

    if pix_shift is not None:
        pyramid = shift_filter_subpx(pyramid, pix_shift)
        if verbose:
            logger.info("shifted pyramid")

    pyramid[pyramid > 1.0] = 1.0

    return pyramid


@functools.lru_cache(maxsize=256)
def streak_matched_kernel(
    fwhm: float, angle_deg: float, length_fwhm: float = 5.0
) -> np.ndarray:
    """Directional matched filter: Gaussian cross-section extruded along an angle.

    Used as part of a filter bank to detect streak-shaped signal in residual
    images (after PSF-model subtraction). The kernel is seeing-limited
    perpendicular to the streak and flat along it, with Gaussian taper at the
    ends to avoid ringing in FFT convolution.

    Args:
        fwhm: PSF full width at half maximum in pixels.
        angle_deg: Streak direction in degrees (0 = along x-axis).
        length_fwhm: Kernel length as a multiple of FWHM.

    Returns:
        2D kernel array normalized to sum to 1.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    length = fwhm * length_fwhm

    # Kernel must encompass the rotated streak + Gaussian wings on all sides
    size = int(np.ceil(length + 6 * sigma))
    if size % 2 == 0:
        size += 1

    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Project pixel coordinates onto streak direction and perpendicular
    along = x * cos_a + y * sin_a
    perp = -x * sin_a + y * cos_a

    # Gaussian profile perpendicular to streak (seeing-limited width)
    cross_section = np.exp(-(perp**2) / (2 * sigma**2))

    # Flat along streak body, Gaussian taper beyond the ends
    half_len = length / 2
    excess = np.maximum(np.abs(along) - half_len, 0)
    along_taper = np.exp(-(excess**2) / (2 * sigma**2))

    kernel = cross_section * along_taper

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


def build_directional_filter_bank(
    fwhm: float, n_angles: int = 36, length_fwhm: float = 5.0
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build a bank of directional matched filters at evenly spaced angles.

    Each filter is a :func:`streak_matched_kernel` at a different orientation.
    Together they form a filter bank that can detect streak-shaped signal at
    any angle by convolving the image with each filter and comparing responses.

    Args:
        fwhm: PSF FWHM in pixels.
        n_angles: Number of angles to sample in [0, 180) degrees.
        length_fwhm: Each filter's length as a multiple of FWHM.

    Returns:
        Tuple of (list of kernel arrays, array of angles in degrees).
    """
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    # Round for lru_cache friendliness
    fwhm_r = round(float(fwhm), 2)
    length_r = round(float(length_fwhm), 2)
    kernels = [streak_matched_kernel(fwhm_r, float(a), length_r) for a in angles]
    return kernels, angles


@functools.lru_cache(maxsize=32)
def sidereal_kernel(fwhm: float) -> np.ndarray:
    """Generate a 2D Gaussian kernel for sidereal star detection.

    Args:
        fwhm (float): Full width at half maximum of the Gaussian in pixels.

    Returns:
        np.ndarray: 2D Gaussian kernel normalized to sum to 1.
    """
    # Convert FWHM to sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Make kernel size odd and ~6 sigma
    size = int(np.ceil(6 * sigma))
    if size % 2 == 0:
        size += 1

    # Create coordinate grid
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # Generate 2D Gaussian
    kernel = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    # Normalize to sum to 1
    kernel = kernel / kernel.sum()

    return kernel
