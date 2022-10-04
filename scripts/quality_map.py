from etc_workflow.config import settings
from etc_workflow.minio import MinioConnection
from etc_workflow.mongo import MongoConnection
from etc_workflow.utilities.raster import _sentinel_raster_to_polygon, _download_sample_band_by_tile
from typing import List
from pathlib import Path

import os 
import json
import rasterio
import numpy as np

def _quality_map(tiles: List[str]):
    minio_client = MinioConnection()
    mongo_client = MongoConnection()
    json_file = {"type": "FeatureCollection", "features": []}

    for tile in tiles: 
        product_path = str(Path(settings.TMP_DIR, "classification_"))
        product_path = product_path + tile + ".tif"
        objects = minio_client.list_objects(settings.MINIO_BUCKET_CLASSIFICATIONS, prefix=settings.MINIO_DATA_FOLDER_NAME + "/classification_" + tile + ".tif")
        objects_length = len(list(objects))

        if (objects_length == 1): 
            minio_client.fget_object(settings.MINIO_BUCKET_CLASSIFICATIONS, settings.MINIO_DATA_FOLDER_NAME + "/classification_" + tile + ".tif", product_path)
            polygon_dict = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon"}}
            with rasterio.open(product_path) as band_file:
                _, json_d = _sentinel_raster_to_polygon(product_path)
                band = band_file.read()
                nodata = (1*(np.count_nonzero(band == 0)))/band.size
                polygon_dict["properties"]["nodata"] = "%.2f" % nodata
                polygon_dict["geometry"] = json_d
                json_file["features"].append(polygon_dict)
            os.remove(product_path)

        if (objects_length == 0):
            _object = _download_sample_band_by_tile(tile, minio_client=minio_client, mongo_collection=mongo_client.get_collection_object())
            polygon_dict = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon"}}
            _, json_d = _sentinel_raster_to_polygon(_object)
            polygon_dict["properties"]["nodata"] = "1"
            polygon_dict["geometry"] = json_d
            json_file["features"].append(polygon_dict)
        
            os.remove(_object)

    with open(str(Path(settings.TMP_DIR,'quality_map.json')), 'w') as f:
        json.dump(json_file, f)
    
    minio_client.fput_object(settings.MINIO_BUCKET_TILE_METADATA, str(Path(settings.MINIO_DATA_FOLDER_NAME,'quality_map.json')), str(Path(settings.TMP_DIR,'quality_map.json')), content_type="application/json")