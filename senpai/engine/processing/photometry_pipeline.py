"""Photometry processing pipeline."""

import logging
from pathlib import Path

import numpy as np

from senpai.astrometry import solve_field
from senpai.catalog.runner import enforce_catalog, query_catalog
from senpai.core.config import get_config
from senpai.engine.detection.point.fwhm import measure_fwhm_from_catalog_stars
from senpai.engine.detection.point.sidereal import extract_point_sources
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.starfield import StarField
from senpai.engine.photometry.utils import (
    PhotometryConfig,
    measure_starfield_photometry,
)
from senpai.engine.utils.file_io import load_fits_file
from senpai.engine.utils.preprocessing import preprocess_image

logger = logging.getLogger(__name__)


def process_image_photometry(
    fits_path: str,
    config: PhotometryConfig | None = None,
    output_dir: Path | None = None,
    save_plots: bool = False,
    save_apertures: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Process photometry on a single FITS image.

    Parameters
    ----------
    fits_path : str
        Path to the FITS image
    config : PhotometryConfig, optional
        Photometry configuration
    output_dir : Path, optional
        Output directory for results
    save_plots : bool
        Whether to save diagnostic plots
    save_apertures : bool
        Whether to save aperture visualization
    verbose : bool
        Whether to print detailed output

    Returns
    -------
    dict
        Photometry results and summary
    """
    enforce_catalog()

    # Load and preprocess image
    logger.info(f"Loading image: {fits_path}")
    image = load_fits_file(fits_path)

    # Apply preprocessing
    app_config = get_config()
    image = preprocess_image(image, app_config, store_intermediates=False)

    # Extract point sources and solve astrometry
    logger.info("Extracting point sources and solving astrometry...")
    sources, initial_fwhm = extract_point_sources(image, max_detections=app_config.astrometry.max_sources)

    # Solve field to get WCS solution
    starfield = solve_field(sources)

    if not starfield.fit:
        logger.error("Astrometry failed - cannot perform photometry without WCS solution")
        return {"error": "Astrometry failed"}

    logger.info(f"Astrometry successful - {len(starfield.catalog_stars or [])} catalog stars found")

    # Query catalog stars and measure FWHM
    if starfield.wcs:
        catalog = query_catalog(starfield.wcs, max_stars=1000, apply_sip=True)

        # Apply radius filtering if configured
        if app_config.astrometry.reduce_field_by_radius is not None:
            from senpai.engine.utils.propagate_wcs import filter_catalog_stars_by_radius

            catalog = filter_catalog_stars_by_radius(
                catalog, starfield.image_metadata, app_config.astrometry.reduce_field_by_radius
            )
            logger.info(
                "Filtered catalog stars to %i stars within %.2f%% of image circle",
                len(catalog.stars),
                app_config.astrometry.reduce_field_by_radius * 100,
            )

        # Merge catalog information
        image_metadata = starfield.image_metadata.model_dump()
        image_metadata.update(catalog.image_metadata.model_dump())
        starfield.catalog_stars = catalog.stars
        starfield.image_metadata = type(starfield.image_metadata)(**image_metadata)

        # Measure FWHM from catalog stars
        logger.info("Measuring FWHM from catalog stars...")
        fwhm_stats = measure_fwhm_from_catalog_stars(
            image, catalog.stars, initial_fwhm, app_config, sat_level=sources.sat_level
        )
        starfield.fwhm_stats = fwhm_stats
        logger.info(f"FWHM measured: {fwhm_stats.median_fwhm:.2f} pixels (from {fwhm_stats.n_measurements} stars)")

    # Perform photometry
    logger.info("Performing photometry measurements...")
    photometry_results, summary = measure_starfield_photometry(image, starfield, config)

    # Prepare results
    results = {
        "image_path": fits_path,
        "astrometry_success": starfield.fit,
        "n_stars_measured": summary.n_stars,
        "n_quality_measurements": summary.n_quality,
        "median_snr": summary.median_snr,
        "median_background": summary.median_background,
        "limiting_magnitude": summary.limiting_magnitude,
        "zero_point": summary.zero_point,
        "zero_point_err": summary.zero_point_err,
        "fwhm_median": starfield.fwhm_stats.median_fwhm if starfield.fwhm_stats else None,
        "fwhm_n_measurements": starfield.fwhm_stats.n_measurements if starfield.fwhm_stats else None,
        "photometry_results": [],
    }

    # Add individual star results
    for result in photometry_results:
        star_result = {
            "star_id": getattr(result.star, "catalog_id", None),
            "x": result.star.x,
            "y": result.star.y,
            "ra": result.star.ra,
            "dec": result.star.dec,
            "magnitude": getattr(result.star, "magnitude", None),
            "magnitudes": getattr(result.star, "magnitudes", None),
            "optimal_radius": result.optimal_radius,
            "optimal_flux": result.optimal_flux,
            "optimal_flux_err": result.optimal_flux_err,
            "snr": result.snr,
            "background_level": result.background_level,
            "background_std": result.background_std,
            "crowding_factor": result.crowding_factor,
            "saturation_flag": result.saturation_flag,
            "edge_flag": result.edge_flag,
            "quality_flag": result.quality_flag,
            "fwhm_measured": result.fwhm_measured,
            "ellipticity": result.ellipticity,
            "sky_coverage": result.sky_coverage,
        }
        results["photometry_results"].append(star_result)

    # Print summary
    if verbose:
        logger.info("Photometry Summary:")
        logger.info(f"  Total stars measured: {summary.n_stars}")
        logger.info(f"  Quality measurements: {summary.n_quality}")
        logger.info(f"  Median SNR: {summary.median_snr:.2f}")
        logger.info(f"  Median background: {summary.median_background:.2f}")
        logger.info(f"  Limiting magnitude: {summary.limiting_magnitude:.2f}")
        if starfield.fwhm_stats:
            logger.info(
                f"  FWHM: {starfield.fwhm_stats.median_fwhm:.2f} pixels (from {starfield.fwhm_stats.n_measurements} measurements)"
            )
        if summary.zero_point is not None:
            logger.info(f"  Zero point: {summary.zero_point:.3f} +/- {summary.zero_point_err:.3f}")

    # Save plots if requested
    if save_plots and output_dir:
        _save_photometry_plots(image, photometry_results, starfield, output_dir)

    # Save aperture visualization if requested
    if save_apertures and output_dir:
        _save_aperture_visualization(image, photometry_results, output_dir)

    return results


def _save_photometry_plots(
    image: ProcessedFitsImage,
    photometry_results: list,
    starfield: StarField,
    output_dir: Path,
):
    """Save diagnostic plots for photometry results."""
    try:
        from senpai.engine.plotting.images import plot_single_frame

        # Plot with photometry results
        plot_single_frame(
            image.data,
            starfield=starfield,
            markersize=starfield.fwhm_stats.median_fwhm if starfield.fwhm_stats else 3.0,
            output_file=output_dir / "photometry_overview.png",
        )

        logger.info(f"Saved photometry overview plot: {output_dir / 'photometry_overview.png'}")
    except Exception as e:
        logger.warning(f"Could not save photometry plots: {e}")


def _save_aperture_visualization(
    image: ProcessedFitsImage,
    photometry_results: list,
    output_dir: Path,
):
    """Save aperture visualization for photometry results."""
    try:
        import matplotlib.pyplot as plt
        from photutils.aperture import CircularAnnulus, CircularAperture

        # Create figure
        _fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Show image
        ax.imshow(
            image.data,
            origin="lower",
            cmap="viridis",
            vmin=np.percentile(image.data, 1),
            vmax=np.percentile(image.data, 99),
        )

        # Plot apertures for quality measurements
        quality_results = [r for r in photometry_results if r.quality_flag]

        for i, result in enumerate(quality_results[:20]):  # Limit to first 20 for clarity
            # Optimal aperture
            aperture = CircularAperture((result.star.x, result.star.y), r=result.optimal_radius)
            aperture.plot(ax, color="red", lw=1, alpha=0.7)

            # Background annulus
            bg_aperture = CircularAnnulus(
                (result.star.x, result.star.y), r_in=result.optimal_radius * 1.5, r_out=result.optimal_radius * 2.5
            )
            bg_aperture.plot(ax, color="blue", lw=1, alpha=0.5)

            # Add star label
            ax.text(result.star.x + 5, result.star.y + 5, f"{i + 1}", color="white", fontsize=8)

        ax.set_title(f"Photometry Apertures ({len(quality_results)} quality measurements)")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        plt.tight_layout()
        plt.savefig(output_dir / "photometry_apertures.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved aperture visualization: {output_dir / 'photometry_apertures.png'}")
    except Exception as e:
        logger.warning(f"Could not save aperture visualization: {e}")
