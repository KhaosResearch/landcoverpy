import json
import os 
from pathlib import Path
from os.path import join
from operator import mul
from typing import Collection, List

import numpy as np
import rasterio

from slccw.config import settings
from slccw.minio import MinioConnection
from slccw.mongo import MongoConnection
from slccw.utilities.utils import get_season_dict, get_products_by_tile_and_date
from slccw.utilities.raster import _read_raster, _download_sample_band_by_title, _sentinel_raster_to_polygon, _download_sample_band_by_tile



def _get_dict_of_products_by_tile(tiles: List[str], mongo_collection: Collection, seasons: dict):
    """
    Query to mongo for obtaining products of the given seasons and tiles.
    Then, creates a dictionary with all available products related with its no data percentage.

    Parameters:
        tiles (List[str]): List of tiles names.
        mongo_collection (Collection): Data for mongo access.
        seasons (dict): Dictionary which keys are the name of the seasons. The values are tuples containing the start date and the ending date of each season

    Output:
        tiles (dict): Dictionary order by tile with all available products in mongo and with its no data percentage.

    """

    products_by_tiles = {}
    # Obtain title and no data percentage of every product available for each tile and each season in the given seasons list
    for tile in tiles:
        products_by_tiles[tile] = {}
        for season in seasons:
            start_date, end_date = seasons[season]
            product_metadata_cursor = get_products_by_tile_and_date(
                tile, mongo_collection, start_date, end_date, cloud_percentage=settings.MAX_CLOUD
            )
            products = list(product_metadata_cursor)

            dict_products = {}
            # Generate a dictionary of the products set with title as key and no data percentage as value
            for product in products:
                title = product["title"]
                nodata_percentage = _get_nodata_percentage(title, mongo_collection)
                dict_products[title] = nodata_percentage

            # Add the dictionary of products to the output dictionary by tile and season
            products_by_tiles[tile][season] = dict_products

    return products_by_tiles

def _get_nodata_percentage(product_title: str, mongo_collection: Collection):
    """
    Read a 10m sample band of a product and get the no data porcentage in the given product.

    Parameters:
        product_title (str): Name of the product to download.
        mongo_collection (Collection): Mongo collection.

    Returns:
        percentage (float): Percentage of no data in the given product.

    """

    minio_client = MinioConnection()
    sample_band = _download_sample_band_by_title(
        product_title, minio_client, mongo_collection
    )

    band = _read_raster(sample_band, to_tif=False)

    # Get the number of pixel in the band
    if len(band.shape) == 2:
        n_pixeles = mul(*np.shape(band))
    if len(band.shape) == 3:
        n_pixeles = np.shape(band)[1] * np.shape(band)[2]

    # Calculates percentage of no data
    percentage = (n_pixeles - np.count_nonzero(band)) * 100 / n_pixeles

    return percentage

def get_quality_map_nodata(tiles: List[str]):
    """
    Get percentage of no data for each product in a list of tiles.

    Parameters:
        tiles (List[str]): List of tiles used to compute the quality map
    """

    mongo_client = MongoConnection()
    minio_client = MinioConnection()

    products_by_tiles = _get_dict_of_products_by_tile(
        tiles=tiles, mongo_collection=mongo_client.get_collection_object(), seasons=get_season_dict()
    )

    tile_metadata_name = "quality_map_nodata.json"
    tile_metadata_path = join(settings.TMP_DIR, tile_metadata_name)

    with open(tile_metadata_path, "w") as f:
        json.dump(products_by_tiles, f)

    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_TILE_METADATA,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{tile_metadata_name}",
        file_path=tile_metadata_path,
        content_type="text/json",
    )

def get_log_from_quality_map_nodata(quality_map_local_path: str = None):
    """
    Extract a log file from the nodata quality map. The log file make a summary about possible lack of data in certain tiles.

    Parameters:
        quality_map_local_path (str): Local path of the quality_map, if it is None, "no_data.json" is looked for in Minio.
    """

    minio_client = MinioConnection()

    if quality_map_local_path is None:
        tile_metadata_name = "no_data.json"
        quality_map_local_path = join(settings.TMP_DIR,tile_metadata_name)
        minio_client.fget_object(
            bucket_name=settings.MINIO_BUCKET_TILE_METADATA,
            object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{tile_metadata_name}",
            file_path=quality_map_local_path
        )

    with open(quality_map_local_path) as f:
        quality_map = json.load(f)

    log_file = f"{settings.TMP_DIR}/log_nodata.txt"

    log_tiers = [[] for _ in range(4)]

    for tile, seasons in quality_map.items():
        for season in seasons:

            n_products_available = len(quality_map[tile][season])
            no_data_percentages = list(quality_map[tile][season].values())

            # No data available for a tile in a season
            if n_products_available == 0:
                log_tiers[0].append((tile, season))

            # Only one partial product available
            elif n_products_available == 1 and no_data_percentages[0] > 10:
                log_tiers[1].append((tile, season))
            
            # Only one complete product available
            elif n_products_available == 1:
                log_tiers[2].append((tile, season))

            # Several products available, but all are partial
            elif all(percentage  > 10 for percentage in no_data_percentages):
                log_tiers[3].append((tile, season))

    with open(log_file,"w") as f:
        f.write("Any product available for next tiles in seasons, data is needed\n\n")
        for (tile, season) in log_tiers[0]:
            f.write(f"Tile {tile} in {season}\n")

        f.write("\nOnly one partial product available for next tiles in seasons, data is needed\n\n")
        for (tile, season) in log_tiers[1]:
            f.write(f"Tile {tile} in {season}\n")

        f.write("\nOnly one complete product available for next tiles in seasons, it is not possible to create a composite\n\n")
        for (tile, season) in log_tiers[2]:
            f.write(f"Tile {tile} in {season}\n")

        f.write("\nSeveral products available, but all are partial for next tiles in seasons\n\n")
        for (tile, season) in log_tiers[3]:
            f.write(f"Tile {tile} in {season}\n")

                    
def quality_map_classified_tiles(tiles: List[str]):
    """
    Compute a quality map (geojson) including percentage of nodata in classified tiles.

    Parameters:
        tiles (List[str]): List of tiles used to compute the quality map
    """
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
                polygon_dict["properties"]["tile"] = tile
                polygon_dict["geometry"] = json_d
                json_file["features"].append(polygon_dict)
            os.remove(product_path)

        if (objects_length == 0):
            _object = _download_sample_band_by_tile(tile, minio_client=minio_client, mongo_collection=mongo_client.get_collection_object())
            polygon_dict = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon"}}
            _, json_d = _sentinel_raster_to_polygon(_object)
            polygon_dict["properties"]["nodata"] = "1"
            polygon_dict["properties"]["tile"] = tile
            polygon_dict["geometry"] = json_d
            json_file["features"].append(polygon_dict)
        
            os.remove(_object)

    geojson_filename = 'quality_map_classified_tiles.geojson'

    with open(join(settings.TMP_DIR,geojson_filename), 'w') as f:
        json.dump(json_file, f)
    
    minio_client.fput_object(
        settings.MINIO_BUCKET_TILE_METADATA,
        join(settings.MINIO_DATA_FOLDER_NAME, geojson_filename),
        join(settings.TMP_DIR, geojson_filename),
        content_type="application/json"
    )
    