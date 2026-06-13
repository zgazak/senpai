import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# Global singleton instance
_config_instance: Optional["AppConfig"] = None


def load_yaml(path: Path) -> dict:
    """Load YAML file into dictionary."""
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
            return data.get("app", {})
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return {}


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = "INFO"

    model_config = ConfigDict(frozen=True)


class PlottingConfig(BaseModel):
    """Plotting configuration"""

    debug: bool = Field(description="Debug Plots")
    review: bool = Field(description="Review Plots")
    photometry: bool = Field(default=False, description="Photometry Plots")
    psfs: bool = Field(
        default=False,
        description="Per-frame empirical PSF plots: a stacked-star PSF panel for "
        "sidereal frames and a stacked-streak panel for rate frames (small, "
        "~<1MB each; separate from the heavy `debug` kernel/CC plots). A little "
        ".npy stamp is saved alongside so the panels regenerate after the fact.",
        validation_alias=AliasChoices("psfs", "streak"),
    )


class AstrometryConfig(BaseModel):
    """Astrometry(.net) configuration"""

    indices_series: str = Field(
        description="Indices series (5200/5200_LITE/5200_SENPAI/4100/5200_LITE_4100/4200/CUSTOM)"
    )
    indices_path: str = Field(description="Local indices path")
    max_sources: int = Field(description="Maximum number of sources to solve for")
    min_sources_for_attempt: int = Field(description="Minimum number of sources to attempt astrometry")
    min_width_degrees: float = Field(description="Minimum width in degrees")
    max_width_degrees: float = Field(description="Maximum width in degrees")
    cpulimit_seconds: int = Field(description="CPU limit in seconds")
    docker_image: str | None = Field(description="Docker image name")
    reduce_field_by_radius: float | None = Field(
        default=None,
        description="Reduce field to sources within this radius as % of image circle (null=full field, 1.0=circle contained by width/height)",
    )
    tweak_order: int = Field(
        default=3,
        description="SIP polynomial order for astrometry.net solve-field (2-5, higher for extreme pincushion distortion)",
    )
    sip_refit_order: int = Field(
        default=7,
        description="SIP order for post-solve refit using catalog stars (3-9, higher for extreme/complex distortion patterns)",
    )
    sip_refit_enabled: bool = Field(
        default=True,
        description="Enable SIP refit after initial solve using catalog stars for better edge distortion fitting",
    )


class StarCatalogConfig(BaseModel):
    """Star catalog configuration"""

    type: str = Field(description="Star catalog type")
    path: Optional[str] = Field(
        default=None,
        description="Star catalog path (required for local catalogs like SSTRC7, not needed for online catalogs like SDSS)",
    )
    faint_limit: float | None = Field(
        default=18.0,
        description="Default faint magnitude limit for online catalogs (e.g., Gaia G); "
        "set to None to use the service default.",
    )
    max_stars_per_frame: int | None = Field(
        default=None,
        description="Cap on catalog stars returned per frame for callers that "
        "request the full catalog (max_stars=None). Applied as a magnitude-"
        "stratified subsample so completeness statistics survive; bounds the "
        "per-frame memory/CPU on dense galactic-plane fields (a 74k-star field "
        "needed ~30 GB/worker uncapped). None = unbounded.",
    )

    @model_validator(mode="after")
    def validate_catalog_config(self):
        """Validate that path is provided for local catalogs but not required for online catalogs."""
        # Online catalogs don't need a path
        if self.type in ["sdss", "gaia"]:
            return self  # Path can be None for online catalogs

        # Local catalogs require a path (gaia_local = trimmed Gaia mirror dir)
        if self.type in ["sstrc7", "gaia_local"] and self.path is None:
            raise ValueError(f"path is required for catalog type '{self.type}'")

        return self


class RuntimeConfig(BaseModel):
    """CLI runtime configuration"""

    run_id: str = Field(default="senpai", description="Run identifier")
    output_dir: str = Field(default=".", description="Output directory")
    save_processed_fits: bool = Field(
        default=True,
        description="Write per-frame *_processed.fits next to the results. "
        "Needed for decoupled replotting, but ~260 MB/frame on 8k sensors "
        "(~94% of a night's output) — full-night runs disable it via "
        "`senpai-burr night --no-processed-fits`.",
    )

    model_config = ConfigDict(frozen=False)  # Allow updates to Runtime config


class DetectionConfig(BaseModel):
    """Detection configuration"""

    detect: bool = Field(default=False, description="Detect point sources")
    detect_streaks: bool = Field(default=True, description="Run streak detection when detect=True")
    snr_threshold: float = Field(default=3.0, description="SNR threshold")
    verbose: bool = Field(default=False, description="Verbose mode")
    streak_correlation_radius_fwhm: float = Field(
        default=5.0, description="Match radius for cross-frame streak correlation, in FWHM units"
    )
    streak_angle_tolerance_deg: float = Field(
        default=15.0, description="Angle tolerance for cross-frame streak matching"
    )


class VariableKernelConfig(BaseModel):
    """Configuration for variable streak kernels driven by WCS distortion."""

    enable: bool = Field(
        default=False,
        description="Enable variable streak kernels for rate-track refinement when distortion is high",
    )
    angle_thresh_deg: float = Field(
        default=1.0,
        description="Minimum max_angle_variation_deg required to enable variable kernels",
    )
    length_thresh_fraction: float = Field(
        default=0.05,
        description="Minimum max_length_variation_fraction required to enable variable kernels",
    )
    diagnostics_max_stars: int = Field(
        default=16,
        description="Maximum number of stars to use for variable-kernel diagnostics plots",
    )
    diagnostics_grid_nx: int = Field(
        default=4,
        description="Number of grid points in x for kernel diagnostic mosaics",
    )
    diagnostics_grid_ny: int = Field(
        default=4,
        description="Number of grid points in y for kernel diagnostic mosaics",
    )


class StreakDetectionConfig(BaseModel):
    """Configuration for streak-specific detection options."""

    variable_kernel: VariableKernelConfig = Field(
        default_factory=VariableKernelConfig,
        description="Variable-kernel configuration for streak WCS refinement",
    )


class ValidationConfig(BaseModel):
    """Configuration for box-based shift validation in rate tracking"""

    box_size: int = Field(
        default=11,
        description="Box size (pixels) around each star for lightweight validation",
    )
    n_random_trials: int = Field(default=8, description="Number of random shifts to test against proposed shift")
    random_radius_pixels: int = Field(default=40, description="Radius (pixels) for random shift generation")

    # Validation thresholds
    min_correlation_ratio: float = Field(
        default=0.98,
        description="Proposed shift must be within this ratio of best correlation (0.98 = within 2%)",
    )
    min_absolute_correlation: float = Field(
        default=0.6, description="Minimum absolute correlation required for validation"
    )
    lenient_absolute_correlation: float = Field(
        default=0.55,
        description="Lenient absolute correlation threshold when correlation ratio >= 0.93 (for cases with few stars)",
    )
    fewer_stars_correlation_ratio: float = Field(
        default=0.985,
        description="Stricter correlation ratio required when the proposed shift has "
        "fewer matched stars than the best trial. Was a hardcoded 0.99, which "
        "razor-thin-rejected correct shifts (ratio ~0.987) and fell through to a "
        "flipped shift; 0.985 keeps mild extra strictness over the base ratio.",
    )
    noise_correlation_ratio: float = Field(
        default=0.99,
        description="Strict correlation ratio required when >=3 random trials beat "
        "the proposed shift's star count (strong noise-correlation signal).",
    )
    noise_min_absolute_correlation: float = Field(
        default=0.70,
        description="Strict absolute-correlation floor for the same noise-signal case.",
    )
    max_validation_stars: int = Field(default=50, description="Maximum number of stars to use for validation")


class ExposureTimeConfig(BaseModel):
    """Configuration for exposure time header keys"""

    exposure_time_keys: list[str] = Field(default_factory=list, description="FITS header keys for exposure time")


class ObservationTimeConfig(BaseModel):
    """Configuration for observation time header keys"""

    observation_time_keys: list[str] = Field(default_factory=list, description="FITS header keys for observation time")
    format: str = Field(
        default="iso",
        description="Time format (supported: 'iso', or use datetime format code '%Y-%m-%dT%H:%M:%S.%f' or similar)",
    )


class SiteConfig(BaseModel):
    """Configuration for observatory site header keys"""

    site_latitude_keys: list[str] = Field(default_factory=list, description="FITS header keys for site latitude")
    site_longitude_keys: list[str] = Field(default_factory=list, description="FITS header keys for site longitude")
    site_altitude_keys: list[str] = Field(default_factory=list, description="FITS header keys for site altitude")
    positional_format: str = Field(
        default="sexagesimal",
        description="Format for positional values (supported: 'sexagesimal', 'float')",
    )
    positional_unit: str = Field(default="degrees", description="Unit for positional values")
    altitude_unit: str = Field(default="kilometers", description="Unit for altitude")


class PointingConfig(BaseModel):
    """Configuration for telescope pointing header keys"""

    boresight_azimuth_keys: list[str] = Field(
        default_factory=list, description="FITS header keys for boresight azimuth"
    )
    boresight_altitude_keys: list[str] = Field(
        default_factory=list, description="FITS header keys for boresight altitude"
    )
    ra_dec_format: str = Field(
        default="sexagesimal",
        description="Format for RA and DEC (supported: 'sexagesimal', 'float')",
    )
    ra_units: str = Field(default="hours", description="Unit for RA (supported: 'hours', 'degrees')")
    dec_units: str = Field(default="degrees", description="Unit for DEC")
    target_ra_keys: list[str] = Field(default_factory=list, description="FITS header keys for target RA")
    target_dec_keys: list[str] = Field(default_factory=list, description="FITS header keys for target DEC")


class TrackingConfig(BaseModel):
    """Configuration for telescope tracking header keys"""

    track_ra_rate_keys: list[str] = Field(default_factory=list, description="FITS header keys for RA tracking rate")
    track_dec_rate_keys: list[str] = Field(default_factory=list, description="FITS header keys for DEC tracking rate")
    track_ra_rate_unit: str = Field(default="arcseconds/second", description="Unit for RA tracking rate")
    track_dec_rate_unit: str = Field(default="arcseconds/second", description="Unit for DEC tracking rate")
    track_mode_keys: list[str] = Field(default_factory=list, description="FITS header keys for tracking mode")


class HeadersConfig(BaseModel):
    """Configuration for FITS header mappings"""

    exposure_time: ExposureTimeConfig = Field(
        default_factory=ExposureTimeConfig,
        description="Exposure time header configuration",
    )
    observation_time: ObservationTimeConfig = Field(
        default_factory=ObservationTimeConfig,
        description="Observation time header configuration",
    )
    site: SiteConfig = Field(default_factory=SiteConfig, description="Site header configuration")
    pointing: PointingConfig = Field(default_factory=PointingConfig, description="Pointing header configuration")
    tracking: TrackingConfig = Field(default_factory=TrackingConfig, description="Tracking header configuration")
    filter_keys: list[str] = Field(
        default=["FILTER", "FILTER1", "INSFILTE"],
        description="FITS header keys for observation filter",
    )


class PhotometryConfig(BaseModel):
    """Configuration for photometry measurements.

    Single source of truth for all photometry knobs — the engine
    (senpai.engine.photometry.utils) uses this class directly.
    """

    # Aperture photometry: fixed aperture size as multiple of FWHM
    aperture_radius_factor: float = Field(default=2.0, description="Aperture radius as multiple of FWHM")

    # Background annulus
    bg_inner_factor: float = Field(default=3.0, description="Background inner radius as multiple of FWHM")
    bg_outer_factor: float = Field(default=5.0, description="Background outer radius as multiple of FWHM")

    # Quality thresholds
    min_snr: float = Field(default=3.0, description="Minimum signal-to-noise ratio for the quality flag")
    max_crowding: float = Field(default=0.3, description="Maximum crowding factor for the quality flag")

    # Crowding / blending control for calibration stars. Used when selecting
    # stars for zero point and limiting magnitude, to avoid blended sources.
    isolation_radius_factor: float = Field(
        default=2.0, description="Isolation radius in units of photometric aperture radius"
    )
    isolation_delta_mag: float = Field(
        default=2.0, description="Minimum magnitude difference for a 'much brighter' neighbor"
    )

    # Limiting magnitude estimation
    limiting_snr: float = Field(
        default=3.0,
        description="SNR threshold used when estimating limiting magnitude (e.g., 3 or 5).",
    )
    limiting_completeness_fraction: float = Field(
        default=0.5,
        description="Completeness fraction for limiting magnitude (e.g., 0.5 for 50% of catalog stars above limiting_snr).",
    )
    completeness_isolate: bool = Field(
        default=True,
        description="Drop catalog stars blended with brighter neighbors from the completeness curve",
    )

    # Zero-point star selection. The ZP must come from well-measured stars only:
    # a faint catalog tail (where forced photometry latches onto neighbour flux /
    # trails and reports a spurious SNR floor) biases the median ZP up by ~1 mag.
    zp_min_snr: float = Field(default=20.0, description="Only stars at/above this SNR contribute to the zero point")
    zp_max_crowding: float = Field(default=0.2, description="...and below this crowding factor")
    zp_sigma_clip: float = Field(default=3.0, description="Sigma-clip threshold on the per-star ZP values")
    zp_min_stars: int = Field(default=8, description="Need at least this many stars to trust the high-SNR cut")

    # Uncertainty estimation
    include_read_noise: bool = Field(default=True, description="Include read noise in uncertainty")
    read_noise: float = Field(default=5.0, description="Read noise in electrons")
    gain: float = Field(default=1.0, description="Gain in electrons per ADU")

    # Magnitude selection for open band observations
    preferred_filters: list[str] = Field(
        default=["Johnson_V", "Johnson_R", "Sloan_r", "Gaia_G", "Sloan_g", "Johnson_B"],
        description="Preferred filters in order of preference",
    )

    # Multi-band calibration
    target_bands: list[str] = Field(
        default=["Johnson_V", "Sloan_r", "Gaia_G"],
        description="Target photometric bands for multi-band zero point calibration",
    )
    color_index_bands: tuple[str, str] = Field(
        default=("Gaia_BP", "Gaia_RP"),
        description="Bands forming the color index for color-term corrections",
    )
    enable_color_terms: bool = Field(
        default=True,
        description="Enable color term corrections in multi-band calibration",
    )


class CalibrationsConfig(BaseModel):
    """Configuration for calibration frames (flats, darks, etc.)"""

    master_flats_dir: str | None = Field(default=None, description="Directory containing master flat files")
    master_darks_dir: str | None = Field(default=None, description="Directory containing master dark files")
    auto_apply_flats: bool = Field(
        default=False,
        description="Automatically apply master flats during preprocessing",
    )
    auto_apply_darks: bool = Field(
        default=False,
        description="Automatically apply master darks during preprocessing",
    )
    dark_matching_headers: list[str] = Field(
        default=["XBINNING", "EXPTIME"],
        description="FITS header keywords that must match for dark frames (exposure time is handled separately)",
    )
    flat_matching_headers: list[str] = Field(
        default=["XBINNING", "FILTER"],
        description="FITS header keywords that must match for flat frames",
    )
    max_dark_exposure_ratio: float = Field(
        default=2.0,
        description="Maximum ratio between image and dark exposure times for automatic scaling",
    )

    # Preprocessing steps configuration
    auto_remove_row_median: bool = Field(
        default=True,
        description="Automatically remove row medians during preprocessing",
    )
    auto_remove_column_median: bool = Field(
        default=True,
        description="Automatically remove column medians during preprocessing",
    )
    auto_subtract_background: bool = Field(
        default=True,
        description="Automatically subtract background during preprocessing",
    )

    # Background subtraction parameters
    background_box_size: int = Field(default=20, description="Box size for background estimation")
    background_filter_size: int = Field(default=3, description="Filter size for background estimation")
    background_exclude_percentile: float = Field(
        default=50.0, description="Percentile to exclude in background estimation"
    )
    background_sigma: float = Field(default=3.0, description="Sigma for background estimation")
    background_maxiters: int = Field(default=10, description="Maximum iterations for background estimation")

    # Image scaling configuration
    auto_scale_images: bool = Field(default=False, description="Automatically scale images to optimize FWHM")
    scaling_method: str = Field(
        default="block_median",
        description="Scaling method: 'block_median' (fast + hot pixel removal) or 'blur_decimate' (better photometry)",
    )
    target_fwhm: float = Field(default=3.0, description="Target FWHM in pixels after scaling")
    oversample_threshold: float = Field(default=4.0, description="Only scale images if FWHM > this threshold")


class AppConfig(BaseModel):
    """Application configuration"""

    version: str = Field(description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    astrometry: AstrometryConfig = Field(default_factory=AstrometryConfig, description="Astrometry configuration")
    star_catalog: StarCatalogConfig = Field(default_factory=StarCatalogConfig, description="Star catalog configuration")
    plotting: PlottingConfig = Field(default_factory=PlottingConfig, description="Plotting configuration")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig, description="Runtime configuration options")
    detection: DetectionConfig = Field(default_factory=DetectionConfig, description="Detection configuration")
    streak: StreakDetectionConfig = Field(
        default_factory=StreakDetectionConfig,
        description="Streak detection and tracking configuration",
    )
    validation: ValidationConfig = Field(default_factory=ValidationConfig, description="Validation configuration")
    headers: HeadersConfig = Field(default_factory=HeadersConfig, description="FITS header mapping configuration")
    photometry: PhotometryConfig = Field(default_factory=PhotometryConfig, description="Photometry configuration")
    calibrations: CalibrationsConfig = Field(
        default_factory=CalibrationsConfig,
        description="Calibration frames configuration",
    )

    model_config = ConfigDict(frozen=True)


def get_config() -> AppConfig:
    """Get the global config instance.

    Returns:
        AppConfig: The global configuration instance. If not initialized,
        loads the default development configuration.
    """
    global _config_instance

    if _config_instance is None:
        raise RuntimeError("Config not initialized")

    return _config_instance


def initialize_config(config_path: Path) -> AppConfig:
    """Initialize the global config instance.

    Args:
        config_path: Path to override config YAML.

    Returns:
        AppConfig: Configuration instance
    """
    global _config_instance

    # Load base config
    logger.info(f"Loading configuration from {config_path}")
    config_data = load_yaml(config_path)

    # Create and validate config
    try:
        config = AppConfig(**config_data)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Set the singleton instance
    _config_instance = config
    return config


def get_or_initialize_config(config_path: Path | None = None) -> AppConfig:
    """Get a loaded config, if none, load config_path or LOCAL_OVERRIDE

    Args:
        AppConfig: Configuration instance

    Returns:
        AppConfig: Configuration instance
    """
    try:
        config = get_config()
    except RuntimeError:
        if config_path:
            config = initialize_config(config_path)
        else:
            from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE

            logger.info(f"No config intialized, using {LOCAL_APP_CONFIG_OVERRIDE}")
            config = initialize_config(LOCAL_APP_CONFIG_OVERRIDE)

    return config
