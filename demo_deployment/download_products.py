from datetime import datetime, timedelta
from ds_download.download_using_sentinel_api import download_product_using_sentinel_api
import json
import os
from landcoverpy.utilities.geometries import _group_validated_data_points_by_tile, _kmz_to_geojson, _csv_to_geojson

def download_products():

    with open(os.getenv("SEASONS_FILE"), "r") as f:
        seasons = json.load(f)

    data_file = os.getenv("DB_FILE")
    if data_file.endswith(".kmz"):
        data_file = _kmz_to_geojson(data_file)
    if data_file.endswith(".csv"):
        data_file = _csv_to_geojson(data_file, sep=';')

    polygons_per_tile = _group_validated_data_points_by_tile(data_file)
    tiles_to_train = set(polygons_per_tile.keys())

    tiles_to_predict_env = os.getenv("TILES_TO_PREDICT", "[]")
    if tiles_to_predict_env.lower() != "prediction":
        try:
            tiles_to_predict = set(json.loads(tiles_to_predict_env))
        except json.JSONDecodeError:
            raise ValueError(f"Error parsing TILES_TO_PREDICT: {tiles_to_predict_env}. Expected format: '['NNLLL', 'NNLLL', ...]'")
    else:
        tiles_to_predict = set()

    tiles_to_download = tiles_to_train.union(tiles_to_predict)

    for season in seasons:
        start_date = datetime.strptime(seasons[season]["start"], "%Y-%m-%d")
        end_date = datetime.strptime(seasons[season]["end"], "%Y-%m-%d")

        for tile in tiles_to_download:
            print(f"Processing {tile} for season {season}")
            download_product_using_sentinel_api(False, True, start_date, end_date, tile_id=tile)



            
