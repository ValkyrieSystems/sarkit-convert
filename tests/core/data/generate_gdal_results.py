import datetime
import json
import pathlib

import numpy as np
from osgeo import gdal, osr

# ModelType=Projected, ModelTransformation
DATASET_1 = "http://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks/Lynchburg%2C%20ND/e31f785f-d8b5-4f3a-8bfa-3787309be004/2023-11-18-03-20-25_UMBRA-05/2023-11-18-03-20-25_UMBRA-05_GEC.tif"

# ModelType=Geographic, ModelTransformation
DATASET_2 = "http://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks/Centerfield%2C%20Utah/1f41fffc-b701-432e-942d-869681f02622/2025-10-07-19-08-08_UMBRA-09/2025-10-07-19-08-08_UMBRA-09_GEC.tif"

# ModelType=Projected, ModelPixelScale + ModelTiepoint
DATASET_3 = "https://capella-open-data.s3.amazonaws.com/data/2025/8/26/CAPELLA_C13_SP_GEO_HH_20250826023518_20250826023527/CAPELLA_C13_SP_GEO_HH_20250826023518_20250826023527.tif"

NUM_POINTS = 25

gdal_results = {
    "gdal_version": gdal.__version__,
    "date": datetime.datetime.now(datetime.UTC).isoformat(),
    "datasets": [],
}

for fileurl in [DATASET_1, DATASET_2, DATASET_3]:
    gdal.UseExceptions()

    with gdal.Open("/vsicurl/" + fileurl) as dataset:
        geotransform = dataset.GetGeoTransform()
        file_srs = dataset.GetSpatialRef()
        shape = dataset.RasterYSize, dataset.RasterXSize

        output_srs = osr.SpatialReference()
        output_srs.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(file_srs, output_srs)

        def image_to_geo(x_pixel, y_line):
            x_geo = (
                geotransform[0] + x_pixel * geotransform[1] + y_line * geotransform[2]
            )
            y_geo = (
                geotransform[3] + x_pixel * geotransform[4] + y_line * geotransform[5]
            )
            return x_geo, y_geo

        rows, cols = np.meshgrid(
            np.linspace(0, shape[0], NUM_POINTS), np.linspace(0, shape[1], NUM_POINTS)
        )
        pts = np.stack([*image_to_geo(cols, rows)], axis=-1)
        gdal_latlon = np.asarray(transform.TransformPoints(pts.reshape(-1, 2)))[
            ..., :2
        ].reshape(pts.shape)

    record = {
        "url": fileurl,
        "rows": rows.tolist(),
        "cols": cols.tolist(),
        "lats": gdal_latlon[..., 0].tolist(),
        "lons": gdal_latlon[..., 1].tolist(),
    }
    gdal_results["datasets"].append(record)

output_file = pathlib.Path(__file__).parent / "gdal_results.json"
output_file.write_text(json.dumps(gdal_results, indent=2))
