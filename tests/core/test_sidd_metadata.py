import json
import pathlib

import numpy as np
import numpy.polynomial.polynomial as npp
import tifffile
from smart_open import open

import sarkit_convert.sidd_metadata as scsm


def test_geotiff():
    gdal_coordinates = json.loads(
        (pathlib.Path(__file__).parent / "data" / "gdal_results.json").read_text()
    )
    for record in gdal_coordinates["datasets"]:
        with open(record["url"], "rb") as file, tifffile.TiffFile(file) as tif:
            geotiff_metadata = tif.geotiff_metadata
            shape = tif.pages[0].shape

        rowcol_to_lat, rowcol_to_lon, latlon_to_row, latlon_to_col = (
            scsm.sidd_projection_from_geotiff(
                shape,
                geotiff_metadata["GTModelTypeGeoKey"],
                geotiff_metadata.get("ProjectedCSTypeGeoKey"),
                geotiff_metadata.get("GeographicTypeGeoKey"),
                geotiff_metadata.get("ModelTransformation"),
                geotiff_metadata.get("ModelTiepoint"),
                geotiff_metadata.get("ModelPixelScale"),
            )
        )

        # compare to GDAL
        rows = record["rows"]
        cols = record["cols"]
        poly_lats = npp.polyval2d(rows, cols, rowcol_to_lat)
        poly_lons = npp.polyval2d(rows, cols, rowcol_to_lon)
        np.testing.assert_allclose(record["lats"], poly_lats)
        np.testing.assert_allclose(record["lons"], poly_lons)

        # verify round trip
        rows_rt = npp.polyval2d(poly_lats, poly_lons, latlon_to_row)
        cols_rt = npp.polyval2d(poly_lats, poly_lons, latlon_to_col)
        np.testing.assert_allclose(rows, rows_rt, atol=0.1)
        np.testing.assert_allclose(cols, cols_rt, atol=0.1)
