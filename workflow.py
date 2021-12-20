from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import rasterio
from itertools import compress
from config import settings
from aster import get_slope_aspect_from_tile
from utils import(
    _get_kwargs_raster,
    get_products_by_tile_and_date,
    connect_mongo_products_collection,
    connect_mongo_composites_collection,
    group_polygons_by_tile,
    get_product_rasters_paths,
    get_minio,
    normalize,
    read_raster,
    get_composite,
    create_composite,
    get_raster_filename_from_path,
    get_raster_name_from_path,
    filter_rasters_paths_by_pca,
    download_sample_band,
)



def workflow(training: bool, visualization: bool, predict: bool):
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
    spring_start = datetime(2021, 3, 1)
    spring_end = datetime(2021, 4, 30)
    summer_start = datetime(2021, 7, 1)
    summer_end = datetime(2021, 8, 30)
    autumn_start = datetime(2021, 10, 1)
    autumn_end = datetime(2021, 11, 30)
    
    # Step 1 
    minio_client = get_minio()
    mongo_products_collection = connect_mongo_products_collection()
    
    # Model input
    final_df = None

    skip_bands = ['TCI','cover-percentage','ndsi','SCL','classifier',]
    no_data_value = {'cover-percentage':-1, 'ndsi':-1, 'slope':-1, 'aspect':-1}
    pc_columns = ['aspect', 'autumn_B01', 'autumn_evi', 'spring_AOT', 'spring_B01', 'spring_WVP', 'spring_evi', 'summer_B01', 'summer_B02', 'summer_evi', 'summer_moisture', "landcover"]

    # Search product metadata in Mongo
    for tile in tiles:
        tile_df = None
        print(f"Working through tiles {tile}")
        for geometry_id in [1,2]: # Take some ids to speed up the demo  
            if predict:
                geometry = None
            # TODO Filter por fecha, take 3 of every year y añadir todas las bandas
            # Query sample {"title": {"$regex": "_T30SUF_"}, "date": {"$gte": ISODate("2020-07-01T00:00:00.000+00:00"),"$lte": ISODate("2020-07-31T23:59:59.999+00:00")}}
            else:
                geometry = polygons_per_tile[tile][geometry_id]["geometry"]
            geometry_df = None
            
            max_cloud_percentage=0.35
            product_metadata_cursor_spring = get_products_by_tile_and_date(tile, mongo_products_collection, spring_start, spring_end, max_cloud_percentage)
            product_metadata_cursor_summer = get_products_by_tile_and_date(tile, mongo_products_collection, summer_start, summer_end, max_cloud_percentage)
            product_metadata_cursor_autumn = get_products_by_tile_and_date(tile, mongo_products_collection, autumn_start, autumn_end, max_cloud_percentage)

            product_per_season = {
                "spring": list(product_metadata_cursor_spring),
                "autumn": list(product_metadata_cursor_autumn),
                "summer": list(product_metadata_cursor_summer),
            }


            for season, products_metadata in product_per_season.items():
                print(season)
                if len(products_metadata) == 0:
                    print("No product found in selected date")
                    continue
                bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
                bucket_composites = settings.MINIO_BUCKET_NAME_COMPOSITES
                current_bucket = None

                is_valid = []
                for product_metadata in products_metadata:
                    (rasters_paths,is_band) = get_product_rasters_paths(product_metadata, minio_client, bucket_products)
                    rasters_paths = list(compress(rasters_paths,is_band))
                    sample_band_path = rasters_paths[0]
                    sample_band_filename = get_raster_filename_from_path(sample_band_path)
                    file_path = str(Path(settings.TMP_DIR,product_metadata['title'],sample_band_filename))
                    minio_client.fget_object(
                                bucket_name=bucket_products,
                                object_name=sample_band_path,
                                file_path=file_path,
                            )
                    sample_band = read_raster(file_path)
                    num_pixels = np.product(sample_band.shape)
                    num_nans = np.isnan(sample_band).sum()
                    nan_percentage = num_nans/num_pixels
                    is_valid.append(nan_percentage < 0.2)
                products_metadata = list(compress(products_metadata,is_valid))
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

                
                product_name = product_metadata["title"]
                (rasters_paths, is_band) = get_product_rasters_paths(product_metadata, minio_client, current_bucket)
                if predict:
                    rasters_paths = filter_rasters_paths_by_pca(rasters_paths, pc_columns, season)
                temp_product_folder = Path(settings.TMP_DIR, product_name + '.SAFE')
                if not temp_product_folder.exists():
                    Path.mkdir(temp_product_folder)
                print(f"Processing product {product_name}")

                # Create product dataframe
                one_season_df = None

                # Get all sentinel bands
                already_read = []
                for i, raster_path in enumerate(rasters_paths):
                    raster_filename = get_raster_filename_from_path(raster_path)
                    raster_name = get_raster_name_from_path(raster_path)
                    temp_path = Path(temp_product_folder,raster_filename)

                    if any(x in raster_name for x in skip_bands):
                        continue
                    # Read only the first band to avoid duplication of different spatial resolution
                    if raster_name in str(already_read):
                        continue
                    already_read.append(raster_name)

                    print(f"Downloading raster {raster_name} from minio into {temp_path}")
                    minio_client.fget_object(
                        bucket_name=current_bucket,
                        object_name=raster_path,
                        file_path=str(temp_path),
                    )
                    kwargs = _get_kwargs_raster(str(temp_path))
                    spatial_resolution = kwargs["transform"][0]
                    if spatial_resolution == 10:
                        kwargs_10m = kwargs
                    # Once all rasters are loaded, they need to be passed to feature selection (PCA, regression, matusita)
                    band_no_data_value = no_data_value.get(raster_name,0)

                    normalize = is_band[i]
                    path_to_disk = None
                    if visualization:
                        path_to_disk = str(Path(settings.TMP_DIR,'visualization',raster_filename))
                    raster = read_raster(
                                band_path=temp_path, 
                                mask_geometry = geometry, 
                                rescale=True, 
                                no_data_value=band_no_data_value,
                                path_to_disk=path_to_disk,
                                normalize_raster=normalize,
                            )

                    raster_df = pd.DataFrame({f"{season}_{raster_name}": raster.flatten()})

                    if one_season_df is None:
                        one_season_df = raster_df
                    else: 
                        one_season_df = pd.concat([one_season_df, raster_df], axis=1)

                    # Remove raster from disk
                    if temp_path.exists() and temp_path.is_file():
                        Path.unlink(temp_path)
                
                if temp_product_folder.is_dir() and (not any(Path(temp_product_folder).iterdir())):
                    Path.rmdir(temp_product_folder)

                if geometry_df is None:
                    geometry_df = one_season_df
                else: 
                    geometry_df = pd.concat([geometry_df, one_season_df], axis=1)
 
            slope_path, aspect_path = get_slope_aspect_from_tile(tile,mongo_products_collection,minio_client,settings.MINIO_BUCKET_NAME_ASTER)

            raster_name = 'slope'
            if (not predict) or (predict and raster_name in pc_columns):
                band_no_data_value = no_data_value.get(raster_name,0)

                raster = read_raster(
                                band_path=slope_path,
                                mask_geometry=geometry,
                                rescale=True,
                                no_data_value=band_no_data_value,
                                normalize_raster=True, 
                                path_to_disk=str(Path(settings.TMP_DIR,'visualization','slope.tif'))
                        )
                raster_df = pd.DataFrame({raster_name: raster.flatten()})
                raster_df = raster_df.dropna()
                geometry_df = pd.concat([geometry_df, raster_df], axis=1)

            raster_name = 'aspect'
            if (not predict) or (predict and raster_name in pc_columns):
                band_no_data_value = no_data_value.get(raster_name,0)
                raster = read_raster(
                                band_path=aspect_path,
                                mask_geometry=geometry,
                                rescale=True,
                                no_data_value=band_no_data_value,
                                normalize_raster=True, 
                                path_to_disk=str(Path(settings.TMP_DIR,'visualization','aspect.tif'))
                        )
                raster_df = pd.DataFrame({raster_name: raster.flatten()})
                raster_df = raster_df.dropna()
                geometry_df = pd.concat([geometry_df, raster_df], axis=1)

            if not predict:
                # Add classification label
                geometry_df["class"] = polygons_per_tile[tile][geometry_id]["label"]

            if tile_df is None:
                tile_df = geometry_df
            else: 
                tile_df = pd.concat([tile_df, geometry_df], axis=0)
            if predict:
                break

        if final_df is None:
            final_df = tile_df
        else:
            final_df = pd.concat([final_df, tile_df], axis=0)

        if predict:
            print(kwargs_10m)
            kwargs_10m['dtype'] = 'float32'
            kwargs_10m['driver'] = 'GTiff'
            kwargs_10m['nodata'] = 0
            clf = joblib.load('model.pkl')
            predict_df = tile_df
            predict_df.sort_index(inplace=True, axis=1)
            predict_df.fillna(0, inplace=True)
            print(predict_df.head())  
            predictions = clf.predict(predict_df)
            print(predictions)
            predictions = np.where(predictions == 'unclassified', 0, 1)
            print(predictions)
            predictions = np.reshape(predictions, (1, kwargs_10m['height'],kwargs_10m['width']))

            with rasterio.open(str(Path(settings.TMP_DIR,'classification.tif')), "w", **kwargs_10m) as classification_file:
                classification_file.write(predictions)
        
    if not predict:
        final_df = final_df.fillna(np.nan)
        final_df = final_df.dropna()
        print(final_df.head())  
        final_df.to_csv("dataset.csv", index=False)


if __name__ == '__main__':
    import time
    start = time.time()
    print("Training")
    workflow(training=True, visualization=True, predict=False)
    end1 = time.time()
    print('Training function took {:.3f} ms'.format((end1-start)*1000.0))
    # print("Testing")
    # workflow(training=False)
    # end2 = time.time()
    # print('Predict function took {:.3f} ms'.format((end2-end1)*1000.0))
    # print('Workflow in total took {:.3f} ms'.format((end2-start)*1000.0))
