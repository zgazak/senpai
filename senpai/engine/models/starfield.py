import numpy as np
from pydantic import BaseModel, field_serializer, model_validator

from senpai.engine.models.astrometry import (
    ReturnAstrometryConfig,
    WCSMetadata,
    WCSModel,
    WCSStatus,
)
from senpai.engine.models.metadata import DetectionMetadata, FWHMMetadata, ImageMetadata


class StarInImage(BaseModel):
    x: float
    y: float
    counts: float | None = None
    snr: float | None = None

    @field_serializer("x", "y", "counts", "snr")
    def serialize_floats(self, v: float | None) -> float | None:
        if v is None:
            return None
        return round(v, 2)


class SatelliteInImage(BaseModel):
    x: float
    y: float
    snr: float | None = None
    ra: float | None = None
    dec: float | None = None
    pixel_fwhm: float | None = None
    flux: float | None = None
    flux_err: float | None = None
    instrumental_magnitude: float | None = None
    calibrated_magnitudes: dict[str, float] | None = None  # {band: mag}
    magnitude_errs: dict[str, float] | None = None  # {band: err}
    observation_filter: str | None = None  # e.g. "Clear", "V"
    # Streak-specific fields (null for point sources)
    detection_type: str | None = None  # "point" | "streak"
    angle_deg: float | None = None  # Streak angle [0, 180)
    length_pixels: float | None = None  # Streak length in pixels
    rate_pixels_per_sec: float | None = None
    rate_arcsec_per_sec: float | None = None

    @field_serializer("x", "y", "snr", "pixel_fwhm", "flux", "flux_err")
    def serialize_floats(self, v: float | None) -> float | None:
        if v is None:
            return None
        return round(v, 2)

    @field_serializer("ra", "dec")
    def serialize_radec(self, v: float | None) -> float | None:
        if v is None:
            return None
        return round(v, 4)

    @field_serializer("instrumental_magnitude")
    def serialize_instrumental_mag(self, v: float | None) -> float | None:
        if v is None:
            return None
        return round(v, 3)

    @field_serializer("calibrated_magnitudes", "magnitude_errs")
    def serialize_mag_dicts(self, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return None
        return {k: round(val, 3) for k, val in v.items()}


class StarInSpace(BaseModel):
    ra: float
    dec: float
    magnitude: float | None = None  # Primary magnitude (for backward compatibility)
    magnitudes: dict[str, float] | None = None  # All available magnitudes by filter
    x: float | None = None
    y: float | None = None
    counts: float | None = None
    snr: float | None = None
    catalog: str | None = None
    catalog_id: str | None = None

    @field_serializer("x", "y", "snr", "counts", "magnitude")
    def serialize_floats(self, v: float | None) -> float | None:
        if v is None:
            return None
        return round(v, 2)

    @field_serializer("ra", "dec")
    def serialize_radec(self, v: float | None) -> float | None:
        if v is None:
            return None
        return round(v, 4)

    @field_serializer("magnitudes")
    def serialize_magnitudes(
        self, v: dict[str, float] | None
    ) -> dict[str, float] | None:
        if v is None or len(v) == 0:
            # If magnitudes is empty but magnitude is set, create magnitudes with primary magnitude
            if self.magnitude is not None:
                return {"Primary": round(self.magnitude, 3)}
            return None
        return {k: round(v_val, 3) for k, v_val in v.items()}


class StarListSpace(BaseModel):
    stars: list[StarInSpace] = []
    image_metadata: ImageMetadata

    def centers_radec(self) -> np.ndarray:
        # Get all valid RA/Dec pairs
        return np.array(
            [
                [star.ra, star.dec]
                for star in self.stars
                if star.ra is not None and star.dec is not None
            ]
        )


class SatelliteListImage(BaseModel):
    detections: list[SatelliteInImage] = []
    image_metadata: ImageMetadata

    def centers_xy(self) -> np.ndarray:
        return np.array(
            [
                [satellite.x, satellite.y, satellite.pixel_fwhm]
                for satellite in self.detections
            ]
        )


class StarListImage(BaseModel):
    detections: list[StarInImage] = []
    image_metadata: ImageMetadata
    # Frame saturation level measured during detection (ADU). Downstream
    # FWHM measurement reuses it: estimating saturation from a
    # magnitude-sorted catalog sample is structurally unreliable (the
    # percentile lands in the faint bulk), whereas the detection-flux-sorted
    # sample measures it correctly.
    sat_level: float | None = None

    def centers_xy(self) -> np.ndarray:
        return np.array([[star.x, star.y, star.counts] for star in self.detections])

    @classmethod
    def from_starfield(cls, starfield: "StarField") -> "StarListImage":
        sources = [
            StarInImage(x=star.x, y=star.y, counts=star.counts)
            for star in starfield.detections
            if star.x is not None and star.y is not None
        ]

        return cls(detections=sources, image_metadata=starfield.image_metadata)


class StarField(BaseModel):
    astrometric_fit_stars: list[StarInSpace] | None = None
    catalog_stars: list[StarInSpace] | None = None
    detections: list[StarInImage]
    image_metadata: ImageMetadata
    fit: bool = False
    wcs: WCSModel | None
    wcs_metadata: WCSMetadata | None = None
    detection_metadata: DetectionMetadata | None = None
    astrometry: ReturnAstrometryConfig | None = None
    wcs_status: WCSStatus = WCSStatus.NO_WCS
    limiting_magnitude: float | None = None
    fwhm_stats: FWHMMetadata | None = None
    scale_factor: float | None = None  # Track if image has been scaled
    # Optional per-field distortion diagnostics derived from the WCS
    # Keys are scalar metrics such as:
    #   - "delta_J"
    #   - "max_angle_variation_deg"
    #   - "max_length_variation_fraction"
    distortion_metrics: dict[str, float] | None = None

    @model_validator(mode="after")
    def create_wcs_metadata(self) -> "StarField":
        if self.wcs is not None and self.wcs_metadata is None:
            self.wcs_metadata = WCSMetadata.from_wcs(self.wcs.to_astropy_wcs())
        return self

    def centers_radec(self, centers: list[StarInSpace]) -> np.ndarray:
        return np.array(
            [
                [star.ra, star.dec]
                for star in centers
                if star.ra is not None and star.dec is not None
            ]
        )

    def centers_xy(self, centers: list[StarInSpace]) -> np.ndarray:
        return np.array(
            [
                [star.x, star.y]
                for star in centers
                if star.x is not None and star.y is not None
            ]
        )

    def astrometric_centers_radec(self) -> np.ndarray:
        return self.centers_radec(self.astrometric_fit_stars)

    def astrometric_centers_xy(self) -> np.ndarray:
        return self.centers_xy(self.astrometric_fit_stars)

    def catalog_centers_radec(self) -> np.ndarray:
        return self.centers_radec(self.catalog_stars)

    def catalog_centers_xy(self, limiting_magnitude: float | None = None) -> np.ndarray:
        if self.catalog_stars is None:
            return None

        # Use StarField's limiting_magnitude if not explicitly provided
        if limiting_magnitude is None:
            limiting_magnitude = self.limiting_magnitude

        if limiting_magnitude is None:
            return self.centers_xy(self.catalog_stars)
        else:
            return self.centers_xy(
                [
                    star
                    for star in self.catalog_stars
                    if star.magnitude is not None
                    and star.magnitude <= limiting_magnitude
                ]
            )

    def detection_centers_xy(self) -> np.ndarray:
        return self.centers_xy(self.detections)
