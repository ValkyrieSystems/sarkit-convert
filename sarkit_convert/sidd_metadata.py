import enum
import logging

import numpy as np
import numpy.typing as npt
import pyproj

import sarkit_convert._utils

logger = logging.getLogger(__name__)


class _ModelTypeCodes(enum.IntEnum):
    """GeoTIFF GTModelTypeGeoKey values"""

    ModelTypeProjected = 1
    ModelTypeGeographic = 2
    ModelTypeGeocentric = 3


def _get_transformation_matrix(
    model_transformation_tag: npt.ArrayLike | None,
    model_tiepoint_tag: npt.ArrayLike | None,
    model_pixel_scale_tag: npt.ArrayLike | None,
):
    """Matrix to convert from image coordinates to model coordinates

    See: http://geotiff.maptools.org/spec/geotiff2.6.html
    """

    if model_transformation_tag is not None:
        matrix = np.asarray(model_transformation_tag).reshape(4, 4)
        if np.any(matrix[-1] != (0.0, 0.0, 0.0, 1.0)):
            logger.warning(
                "Last row of ModelTransformation matrix must be (0, 0, 0, 1)"
            )
        return matrix

    if model_pixel_scale_tag is not None and model_tiepoint_tag is not None:
        sx, sy, sz = np.asarray(model_pixel_scale_tag)
        mtp = np.atleast_2d(model_tiepoint_tag)

        if mtp.shape[0] != 1:
            # Probably want to interpolate tiepoints if there is more than one
            # Should only be one tiepoint when ModelPixelScale is present
            return None

        i, j, k, x, y, z = mtp[0]
        tx = x - i * sx
        ty = y + j * sy
        tz = z - k * sz
        matrix = np.asarray(
            [
                [sx, 0.0, 0.0, tx],
                [0.0, -sy, 0.0, ty],
                [0.0, 0.0, sz, tz],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return matrix

    return None


def sidd_projection_from_geotiff(
    shape: tuple[int, int],
    gt_model_type_geo_key: int,
    projected_cs_type_geo_key: int | None,
    geographic_type_geo_key: int | None,
    model_transformation_tag: npt.ArrayLike | None,
    model_tiepoint_tag: npt.ArrayLike | None,
    model_pixel_scale_tag: npt.ArrayLike | None,
    grid_size: int = 11,
    max_order: int = 5,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate SIDD PolynomialProjection polynomials for a GeoTIFF

    Parameters
    ----------
    shape : tuple of int
        Shape of the image
    gt_model_type_geo_key : int
        Value of the GTModelTypeGeoKey
    projected_cs_type_geo_key : int, optional
        Value of the ProjectedCSTypeGeoKey. Required if gt_model_type_geo_key == 1
    geographic_type_geo_key : int, optional
        Value of the GeographicTypeGeoKey. Required if gt_model_type_geo_key == 2
    model_transformation_tag : array-like, optional
        Value of the ModelTransformationTag
    model_tiepoint_tag : array-like, optional
        Value of the ModelTiepointTag
    model_pixel_scale_tag : array-like, optional
        Value of the ModelPixelScaleTag
    grid_size : int, optional
        Number of fit points in each dimension
    max_order : int, optional
        Maximum order of generated polynomials

    Returns
    -------
    rowcol_to_lat : ndarray
        2D polynomial coefficients.  (row, col) -> latitude degrees
    rowcol_to_lon : ndarray
        2D polynomial coefficients.  (row, col) -> longitude degrees
    latlon_to_row : ndarray
        2D polynomial coefficients.  (latitude degrees, longitude degrees) -> row
    latlon_to_col : ndarray
        2D polynomial coefficients.  (latitude degrees, longitude degrees) -> col

    Notes
    -----
    See:
      * http://geotiff.maptools.org/spec/geotiff2.5.html
      * http://geotiff.maptools.org/spec/geotiff2.6.html

    """
    # Image Coordinates are [I J K 1] -> (column, row, vertical,  1)
    image_coords = np.stack(
        [
            *np.meshgrid(
                np.linspace(0, shape[1], grid_size),
                np.linspace(0, shape[0], grid_size),
            ),
            np.zeros((grid_size, grid_size)),  # no vertical component
            np.ones((grid_size, grid_size)),
        ],
        axis=-1,
    )

    matrix = _get_transformation_matrix(
        model_transformation_tag, model_tiepoint_tag, model_pixel_scale_tag
    )

    if matrix is None:
        raise RuntimeError("Failed to get transformation matrix")

    swap_axis = False
    if gt_model_type_geo_key == _ModelTypeCodes.ModelTypeProjected:
        if projected_cs_type_geo_key is None:
            raise ValueError(
                "projected_cs_type_geo_key must be provided for Projected model type"
            )

        crs = pyproj.CRS.from_epsg(int(projected_cs_type_geo_key))
    elif gt_model_type_geo_key == _ModelTypeCodes.ModelTypeGeographic:
        if geographic_type_geo_key is None:
            raise ValueError(
                "geographic_type_geo_key must be provided for Geographic model type"
            )

        crs = pyproj.CRS.from_epsg(int(geographic_type_geo_key))
        # GeoTIFF axis order convention is reversed from CRS for Geographic coordinate systems
        swap_axis = True
    else:
        raise RuntimeError(
            f"GTModelTypeGeoKey == {gt_model_type_geo_key} not supported"
        )

    # model                              image
    # coords =          matrix     *     coords
    # |-   -|     |-                 -|  |-   -|
    # |  X  |     |   a   b   c   d   |  |  I  |
    # |     |     |                   |  |     |
    # |  Y  |     |   e   f   g   h   |  |  J  |
    # |     |  =  |                   |  |     |
    # |  Z  |     |   i   j   k   l   |  |  K  |
    # |     |     |                   |  |     |
    # |  1  |     |   m   n   o   p   |  |  1  |
    # |-   -|     |-                 -|  |-   -|
    model_coords = np.inner(matrix, image_coords)
    if swap_axis:
        model_x = model_coords[1]
        model_y = model_coords[0]
    else:
        model_x = model_coords[0]
        model_y = model_coords[1]

    transformer = pyproj.Transformer.from_crs(crs, 4326)  # 4326 = WGS84 Lat/Lon
    lats, lons = transformer.transform(model_x, model_y)
    rows = image_coords[..., 1]
    cols = image_coords[..., 0]

    rc_span = max(np.ptp(rows), np.ptp(cols))
    tol_px = 0.1
    tol_lat = np.ptp(lats) / rc_span * tol_px
    tol_lon = np.ptp(lons) / rc_span * tol_px

    rowcol_to_lat = sarkit_convert._utils.polyfit2d_tol(
        rows.flatten(), cols.flatten(), lats.flatten(), max_order, max_order, tol_lat
    )
    rowcol_to_lon = sarkit_convert._utils.polyfit2d_tol(
        rows.flatten(), cols.flatten(), lons.flatten(), max_order, max_order, tol_lon
    )
    latlon_to_row = sarkit_convert._utils.polyfit2d_tol(
        lats.flatten(), lons.flatten(), rows.flatten(), max_order, max_order, tol=tol_px
    )
    latlon_to_col = sarkit_convert._utils.polyfit2d_tol(
        lats.flatten(), lons.flatten(), cols.flatten(), max_order, max_order, tol=tol_px
    )
    return rowcol_to_lat, rowcol_to_lon, latlon_to_row, latlon_to_col
