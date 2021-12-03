import io
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from minio import Minio
from mgrs import MGRS
from pymongo import MongoClient
from rasterio import mask
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from shapely.ops import transform, shape
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import shutil

from config import settings

from aster import get_slope_aspect_from_tile
from utils import(
    get_products_by_tile_and_date,
    connect_mongo_products_collection,
    connect_mongo_composites_collection,
    group_polygons_by_tile,
    get_product_rasters_paths,
    get_minio,
    read_raster,
    normalize,
    get_composite,
    create_composite,
    get_raster_filename_from_path,
    get_raster_name_from_path
)



def workflow(training: bool = True):
    '''
        Step 1: Load Sentinel-2 imagery
        (Skip (?))Step 2: Load pre-processed ASTER DEM
        TODO Step 3: Load River shapefiles (maybe not neccesary)
        TODO Step 4: Calculate all intermediate bands ???
        (Skip) Step 5: Terrain correction (Will not be calculated, according to experts it doesn't provide better results)
        (Skip) Step 6: Index calculation  (If not tc, already calculated)
        Step 7: Normalization (and convert to tabular structure) (outlier removal(?))
        Step 8: layer selection
        Step 9: Train models 
    '''

    # TODO Iterate over a set of geojson/databases (the databases may not be equal)
    polygons_per_tile = group_polygons_by_tile(Path("forest-db/test.geojson"))

    # Tiles related to the traininig zone
    tiles = polygons_per_tile.keys() 
    # This parameters should be coming from somewhere else
    #bands = ["AOT_10m", "B01_60m", "B02_10m", "B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B08_10m", "B09_60m", "B11_20m", "B12_20m", "B8A_20m", "WVP_10m"] # Removed , "SCL_20m"
    spring_start = datetime(2020, 3, 1)
    spring_end = datetime(2020, 3, 31)
    summer_start = datetime(2021, 7, 1)
    summer_end = datetime(2021, 7, 31)
    autumn_start = datetime(2020, 10, 1)
    autumn_end = datetime(2020, 10, 31)
    
    # Step 1 
    minio_client = get_minio()
    mongo_products_collection = connect_mongo_products_collection()
    
    # Model input
    train_df = None

    skip_bands = ['TCI','cover-percentage','ndsi']
    not_normalizable_bands = ['cover-percentage', 'SCL','ndsi']
    no_data_value = {'cover-percentage':-1, 'ndsi':-1, 'slope':-99999, 'aspect':-99999}

    # Search product metadata in Mongo
    for tile in tiles:
        print(f"Working through tiles {tile}")
        for geometry_id in [1]: # Take some ids to speed up the demo  
            # TODO Filter por fecha, take 3 of every year y añadir todas las bandas
            # Query sample {"title": {"$regex": "_T30SUF_"}, "date": {"$gte": ISODate("2020-07-01T00:00:00.000+00:00"),"$lte": ISODate("2020-07-31T23:59:59.999+00:00")}}
            
            product_metadata_cursor_spring = get_products_by_tile_and_date(tile, mongo_products_collection, spring_start, spring_end)
            product_metadata_cursor_summer = get_products_by_tile_and_date(tile, mongo_products_collection, summer_start, summer_end)
            product_metadata_cursor_autumn = get_products_by_tile_and_date(tile, mongo_products_collection, autumn_start, autumn_end)

            product_per_season = {
                "spring": list(product_metadata_cursor_spring),
                "autumn": list(product_metadata_cursor_autumn),
                "summer": list(product_metadata_cursor_summer),
            } 

            season_df = None
            for season, products_metadata in product_per_season.items():
                bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
                bucket_composites = settings.MINIO_BUCKET_NAME_COMPOSITES
                current_bucket = None
                if len(products_metadata) == 1:
                    product_metadata = products_metadata[0]
                    current_bucket = bucket_products
                elif len(products_metadata) > 1:
                    mongo_composites_collection = connect_mongo_composites_collection()
                    products_metadata_list = list(products_metadata)
                    product_metadata = get_composite(products_metadata_list, mongo_composites_collection)
                    if product_metadata is None:
                        create_composite(products_metadata_list, minio_client, bucket_products, bucket_composites, mongo_composites_collection)
                        product_metadata = get_composite(products_metadata_list, mongo_composites_collection)
                    current_bucket = bucket_composites
                else:
                    print("No product found in selected date")
                    continue
                
                product_name = product_metadata["title"]
                (rasters_paths, is_band) = get_product_rasters_paths(product_metadata, minio_client, current_bucket)

                temp_product_folder = Path(settings.TMP_DIR, product_name + '.SAFE')
                if not temp_product_folder.exists():
                    Path.mkdir(temp_product_folder)

                # Create product dataframe
                single_product_df = None

                # Get all sentinel bands
                already_read = []
                for raster_path in rasters_paths:
                    raster_filename = get_raster_filename_from_path(raster_path)
                    raster_name = get_raster_name_from_path(raster_path)
                    temp_path = Path(temp_product_folder,raster_filename)

                    if any(x in raster_name for x in skip_bands):
                        continue
                    # Read only the first band to avoid duplication of different spatial resolution
                    if raster_name in str(already_read):
                        continue
                    already_read.append(raster_name)

                    print(f"Reading raster: -> {current_bucket}:{raster_path} into {temp_path}")
                    minio_client.fget_object(
                        bucket_name=current_bucket,
                        object_name=raster_path,
                        file_path=str(temp_path),
                    )

                    # Once all rasters are loaded, they need to be passed to feature selection (PCA, regression, matusita)
                    band_no_data_value = no_data_value.get(raster_name,0)
                    raster = read_raster(temp_path, mask_geometry = polygons_per_tile[tile][geometry_id]["geometry"], rescale=True, no_data_value=band_no_data_value)

                    if not any(x in raster_name for x in not_normalizable_bands):
                        raster = normalize(raster)
                    
                    raster_df = pd.DataFrame({f"{season}_{raster_name}": raster.flatten()})
                    raster_df= raster_df.dropna()

                    if single_product_df is None:
                        single_product_df = raster_df
                    else: 
                        single_product_df = pd.concat([single_product_df, raster_df], axis=1)

                    # Remove raster from disk
                    print(f"Removing file {str(temp_path)}")
                    Path.unlink(temp_path)
                
                print(f"Removing folder {str(temp_product_folder)}")
                Path.rmdir(temp_product_folder)

                if season_df is None:
                    season_df = single_product_df
                else: 
                    season_df = pd.concat([season_df, single_product_df], axis=1)
 
            slope_path, aspect_path = get_slope_aspect_from_tile(tile,mongo_products_collection,minio_client,settings.MINIO_BUCKET_NAME_PRODUCTS,settings.MINIO_BUCKET_NAME_ASTER)


            raster_name = 'slope'
            band_no_data_value = no_data_value.get(raster_name,0)
            slope = read_raster(slope_path,polygons_per_tile[tile][geometry_id]["geometry"],True,band_no_data_value)
            raster_df = pd.DataFrame({raster_name: slope.flatten()})
            raster_df = raster_df.dropna()
            season_df = pd.concat([season_df, raster_df], axis=1)

            raster_name = 'aspect'
            band_no_data_value = no_data_value.get(raster_name,0)
            aspect = read_raster(aspect_path,polygons_per_tile[tile][geometry_id]["geometry"],True,band_no_data_value)
            raster_df = pd.DataFrame({raster_name: aspect.flatten()})
            raster_df = raster_df.dropna()
            season_df = pd.concat([season_df, raster_df], axis=1)

            if training:
                # Add classification label
                season_df["class"] = polygons_per_tile[tile][geometry_id]["label"]

            if train_df is None:
                train_df = season_df
            else: 
                train_df = pd.concat([train_df, season_df], axis=0)

    print(train_df.head())  
    train_df.to_csv("dataset.csv", index=False)


if __name__ == '__main__':
    import time
    start = time.time()
    print("Training")
    workflow(training=True)
    end1 = time.time()
    print('Training function took {:.3f} ms'.format((end1-start)*1000.0))
    # print("Testing")
    # workflow(training=False)
    # end2 = time.time()
    # print('Predict function took {:.3f} ms'.format((end2-end1)*1000.0))
    # print('Workflow in total took {:.3f} ms'.format((end2-start)*1000.0))
