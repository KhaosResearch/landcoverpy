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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from config import settings

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

def pca(data: pd.DataFrame, variance_explained: int = 75):
    '''
    Return the main columns after a Principal Component Analysis.

    Source: https://bitbucket.org/khaosresearchgroup/enbic2lab-images/src/master/soil/PCA_variance/pca-variance.py
    '''
    def _dimension_reduction(percentage_variance: list, variance_explained: int):
        cumulative_variance = 0
        n_components = 0
        while cumulative_variance < variance_explained:
            cumulative_variance += percentage_variance[n_components]
            n_components += 1
        return n_components

    pca = PCA()
    pca_data = pca.fit_transform(data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    number_components = _dimension_reduction(per_var, variance_explained)

    pca = PCA(number_components)
    pca_data = pca.fit_transform(data)
    pc_labels = ["PC" + str(x) for x in range(1, number_components + 1)]
    pca_df = pd.DataFrame(data=pca_data, columns=pc_labels)

    pca_df.to_csv("PCA_plot.csv", sep=',', index=False)

    col = []
    for columns in np.arange(number_components):
        col.append("PC" + str(columns + 1))

    loadings = pd.DataFrame(pca.components_.T, columns=col, index=data.columns)
    loadings.to_csv("covariance_matrix.csv", sep=',', index=True)

    # Extract the most important column from each PC https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis 
    col_ids = [np.abs(pc).argmax() for pc in pca.components_]
    column_names = data.columns 
    return [column_names[id_] for id_ in col_ids]

    

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

    skip_bands = ['TCI']
    not_normalizable_bands = ['cover-percentage', 'SCL','ndsi']
    no_data_value = {'cover-percentage':-1, 'ndsi':-1}

    # Search product metadata in Mongo
    for tile in tiles:
        print(f"Working through tiles {tile}")
        for geometry_id in [4]: # Take some ids to speed up the demo  
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

            if training:
                # Add classification label
                season_df["class"] = polygons_per_tile[tile][geometry_id]["label"]

            if train_df is None:
                train_df = season_df
            else: 
                train_df = pd.concat([train_df, season_df], axis=0)
    print(train_df.head())  
    train_df.to_csv("dataset.csv", index=False)
    # Temporal code to load premade dataset for training 
    # train_df = pd.read_csv("dataset2.csv")
    # train_df = train_df.drop("Unnamed: 0", axis=1)
    # print(train_df.head())
    #if training:
        # Prepare data for training
        #x_train_data = train_df.drop("class", axis=1)
        #y_train_data = train_df["class"] 

        # Filter bands according to PCA, matusita etc
        #pc_columns = pca(x_train_data,95)
        #print(pc_columns)
        #reduced_x_train_data = x_train_data[pc_columns]
        
        # Train model 
        #clf = RandomForestClassifier()
        #clf.fit(reduced_x_train_data, y_train_data)
        #joblib.dump(clf, 'model.pkl', compress=1)
    #else: # Prediction
        
        #reduced_predict_data = train_df[['B02_10m', 'AOT_10m']] # This should be coming from the PCA used during training
        #clf = joblib.load('model.pkl')
        #results = clf.predict(reduced_predict_data)
        #print(results)

if __name__ == '__main__':
    import time
    start = time.time()
    print("Training")
    workflow(training=True)
    end1 = time.time()
    # print('Training function took {:.3f} ms'.format((end1-start)*1000.0))
    # print("Testing")
    # workflow(training=False)
    # end2 = time.time()
    # print('Predict function took {:.3f} ms'.format((end2-end1)*1000.0))
    # print('Workflow in total took {:.3f} ms'.format((end2-start)*1000.0))