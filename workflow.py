from datetime import datetime
from pathlib import Path
import joblib
from os.path import join
import numpy as np
from glob import glob
import pandas as pd
import rasterio
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
    mask_polygons_by_tile,
    read_raster,
    get_composite,
    create_composite,
    get_raster_filename_from_path,
    get_raster_name_from_path,
    filter_rasters_paths_by_pca,
    kmz_to_geojson,
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

    # Iterate over a set of geojson/databases (the databases may not be equal)
    # Model input
    final_df = None

    geojson_files = []
    for data_class in glob(join(settings.DB_DIR,"*.kmz")):
        print(f"Working with database {data_class}")
        geojson_files.append(kmz_to_geojson(data_class))
    polygons_per_tile = group_polygons_by_tile(*geojson_files)

    # Tiles related to the traininig zone
    tiles = polygons_per_tile.keys() 
    # This parameters should be coming from somewhere else
    spring_start = datetime(2021, 3, 1)
    spring_end = datetime(2021, 5, 30)
    summer_start = datetime(2021, 7, 1)
    summer_end = datetime(2021, 9, 30)
    autumn_start = datetime(2021, 10, 1)
    autumn_end = datetime(2021, 12, 30)
    
    # Step 1 
    minio_client = get_minio()
    mongo_products_collection = connect_mongo_products_collection()
    


    # Names of the bands that are not taken into account
    skip_bands = ['TCI','cover-percentage','ndsi','SCL','classifier',"bri",]
    # Indexes that have to be normalized in training data
    normalizable_indexes = ['bri']
    no_data_value = {'cover-percentage':-1, 'ndsi':-1, 'slope':-1, 'aspect':-1}
    # PCA resulting columns, this should come from somewhere else
    pc_columns = ['aspect', 'autumn_B01', 'autumn_evi', 'spring_AOT', 'spring_B01', 'spring_WVP', 'spring_evi', 'summer_B01', 'summer_B02', 'summer_evi', 'summer_moisture', "landcover"]

    # Search product metadata in Mongo
    for i, tile in enumerate(tiles): # Sample data
        print(f"Working in tile {tile}, {i}/{len(tiles)}")
        tile_df = None

        # Mongo query for obtaining valid products
        max_cloud_percentage=20
        product_metadata_cursor_spring = get_products_by_tile_and_date(tile, mongo_products_collection, spring_start, spring_end, max_cloud_percentage)
        product_metadata_cursor_summer = get_products_by_tile_and_date(tile, mongo_products_collection, summer_start, summer_end, max_cloud_percentage)
        product_metadata_cursor_autumn = get_products_by_tile_and_date(tile, mongo_products_collection, autumn_start, autumn_end, max_cloud_percentage)

        product_per_season = {
                "spring": list(product_metadata_cursor_spring),
                "autumn": list(product_metadata_cursor_autumn),
                "summer": list(product_metadata_cursor_summer),
            }
    
        if len(product_per_season["spring"]) == 0 or len(product_per_season["autumn"]) == 0 or len(product_per_season["spring"]) == 0:
            print(f"There is no valid data for tile {tile}. Skipping it...")
            continue

        print(f"Working through tile {tile}")
    

        # Dataframe for storing data of a tile
        geometry_df = None

        slope_path, aspect_path = get_slope_aspect_from_tile(tile,mongo_products_collection,minio_client,settings.MINIO_BUCKET_NAME_ASTER)

        #Get crop mask and dataset labeled with database points in tile
        crop_mask, dt_labeled = mask_polygons_by_tile(slope_path, polygons_per_tile, tile)

        #Save crop mask to tif file
        if visualization:
            crop_kwargs =  _get_kwargs_raster(str(slope_path))   
            crop_mask_save = np.reshape(crop_mask, (crop_kwargs['count'],*np.shape(crop_mask)))            
            crop_kwargs['dtype'] = 'int16'
            crop_kwargs['driver'] = 'GTiff'
            crop_kwargs['nodata'] =  0
            with rasterio.open(str(Path(settings.TMP_DIR,'mask.tif')), "w", **crop_kwargs) as mask_file:
                mask_file.write(crop_mask_save.astype(int))
        


        # Add slope and aspect data
        raster_name = 'slope'
        if (not predict) or (predict and raster_name in pc_columns):
            band_no_data_value = no_data_value.get(raster_name,0)

            raster = read_raster(
                            band_path=slope_path,
                            rescale=True,
                            no_data_value=band_no_data_value,
                            normalize_raster=True, 
                            path_to_disk=str(Path(settings.TMP_DIR,'visualization','slope.tif'))
                    )
            raster_masked = np.ma.masked_array(raster, mask=crop_mask)                    
            raster_masked = np.ma.compressed(raster_masked).flatten()
            raster_df = pd.DataFrame({raster_name: raster_masked})
            raster_df = raster_df.dropna()
            geometry_df = pd.concat([geometry_df, raster_df], axis=1)

        raster_name = 'aspect'
        if (not predict) or (predict and raster_name in pc_columns):
            band_no_data_value = no_data_value.get(raster_name,0)
            raster = read_raster(
                            band_path=aspect_path,
                            rescale=True,
                            no_data_value=band_no_data_value,
                            normalize_raster=True, 
                            path_to_disk=str(Path(settings.TMP_DIR,'visualization','aspect.tif'))
                    )
                    
            raster_masked = np.ma.masked_array(raster, mask=crop_mask)                    
            raster_masked = np.ma.compressed(raster_masked).flatten()
            raster_df = pd.DataFrame({raster_name: raster_masked})
            raster_df = raster_df.dropna()
            geometry_df = pd.concat([geometry_df, raster_df], axis=1)

        for season, products_metadata in product_per_season.items():
            print(season)
            bucket_products = settings.MINIO_BUCKET_NAME_PRODUCTS
            bucket_composites = settings.MINIO_BUCKET_NAME_COMPOSITES
            current_bucket = None
            #products_metadata = filter_valid_products(products_metadata, minio_client)

            if len(products_metadata) == 0:
                print(f"Products found in tile {tile} are not valid")
                break # Next geometry
            
            elif len(products_metadata) == 1:
                product_metadata = products_metadata[0]
                current_bucket = bucket_products
            else :
                # If there are multiple products for one season, use a composite.
                mongo_composites_collection = connect_mongo_composites_collection()
                products_metadata_list = list(products_metadata)
                product_metadata = get_composite(products_metadata_list, mongo_composites_collection)
                if product_metadata is None:
                    create_composite(products_metadata_list, minio_client, bucket_products, bucket_composites, mongo_composites_collection)
                    product_metadata = get_composite(products_metadata_list, mongo_composites_collection)
                current_bucket = bucket_composites

            product_name = product_metadata["title"]
            (rasters_paths, is_band) = get_product_rasters_paths(product_metadata, minio_client, current_bucket)
            # In predict phase, use only pca-selected rasters
            if predict:
                rasters_paths = filter_rasters_paths_by_pca(rasters_paths, pc_columns, season)
            temp_product_folder = Path(settings.TMP_DIR, product_name + '.SAFE')
            if not temp_product_folder.exists():
                Path.mkdir(temp_product_folder)
            print(f"Processing product {product_name}")

            # Dataframe for storing one season data of a tile
            one_season_df = None

            # Read bands and indexes.
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

                band_no_data_value = no_data_value.get(raster_name,0)

                normalize = is_band[i]
                # All bands are normalized, along with indexes that are not implicitly normalized
                if (not normalize) and any(x in raster_name for x in normalizable_indexes):
                    normalize = True
                path_to_disk = None
                if visualization:
                    path_to_disk = str(Path(settings.TMP_DIR,'visualization',raster_filename))
                raster = read_raster(
                            band_path=temp_path, 
                            rescale=True, 
                            no_data_value=band_no_data_value,
                            path_to_disk=path_to_disk,
                            normalize_raster=normalize,
                        )
                raster_masked = np.ma.masked_array(raster[0], mask=crop_mask)                    
                raster_masked = np.ma.compressed(raster_masked)


                #raster_df = pd.DataFrame({raster_name: raster_masked})
                raster_df = pd.DataFrame({f"{season}_{raster_name}": raster_masked})
                raster_df = raster_df.dropna()

                if one_season_df is None:
                    one_season_df = raster_df
                else: 
                    one_season_df = pd.concat([one_season_df, raster_df], axis=1)
                if temp_path.exists() and temp_path.is_file():
                    Path.unlink(temp_path)
            
            if temp_product_folder.is_dir() and (not any(Path(temp_product_folder).iterdir())):
                Path.rmdir(temp_product_folder)

            if geometry_df is None:
                geometry_df = one_season_df
            else: 
                geometry_df = pd.concat([geometry_df, one_season_df], axis=1)

        if tile_df is None:
            tile_df = geometry_df
        else: 
            tile_df = pd.concat([tile_df, geometry_df], axis=0)
        if predict:
            break
        break

    raster_masked = np.ma.masked_array(dt_labeled, mask=crop_mask)
    raster_masked = np.ma.compressed(raster_masked).flatten()
    raster_df = pd.DataFrame({"label": raster_masked})
    raster_df = raster_df.dropna()
    tile_df = pd.concat([tile_df, raster_df], axis=1)

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
    workflow(training=True, visualization=False, predict=False)
    end1 = time.time()
    print('Training function took {:.3f} ms'.format((end1-start)*1000.0))
    # print("Testing")
    # workflow(training=False)
    # end2 = time.time()
    # print('Predict function took {:.3f} ms'.format((end2-end1)*1000.0))
    # print('Workflow in total took {:.3f} ms'.format((end2-start)*1000.0))
