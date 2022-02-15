from datetime import datetime
from pathlib import Path
import joblib
from os.path import join
import numpy as np
from glob import glob
import pandas as pd
import rasterio
from config import settings
from aster import get_dem_from_tile
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
    skip_bands = ['TCI','cover-percentage','ndsi','SCL','classifier',"bri","WVP"]
    no_data_value = {'slope':-99999, 'aspect':-99999, "ndvi":-99999, "osavi":-99999, "osavi":-99999, "ndre":-99999, "ndbg":-99999, "moisture":-99999, "mndwi":-99999, "evi2":-99999, "evi":-99999}
    # PCA resulting columns, this should come from somewhere else
    pc_columns = ['aspect', 'autumn_evi', 'slope', 'spring_AOT', 'spring_B02', 'spring_B04', 'spring_B07', 'spring_evi', 'summer_WVP', 'summer_evi']
    # Ranges for normalization of each raster
    normalize_range = {"slope":(0,70), "aspect":(0,360), "dem":(0,2000)}

    if predict:
        print("Predicting tiles")
    else:
        print("Creating dataset from tiles")

    for i, tile in enumerate(tiles):
        print(f"Working in tile {tile}, {i}/{len(tiles)}")

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
    
        if len(product_per_season["spring"]) == 0 or len(product_per_season["autumn"]) == 0 or len(product_per_season["summer"]) == 0:
            print(f"There is no valid data for tile {tile}. Skipping it...")
            continue

        print(f"Working through tile {tile}")
    

        # Dataframe for storing data of a tile
        tile_df = None

        #Get crop mask and dataset labeled with database points in tile
        crop_mask, label_lon_lat = mask_polygons_by_tile(polygons_per_tile, tile)

        if predict:
            crop_mask = np.zeros_like(crop_mask)
        

        dems_raster_names = ["slope", "aspect", "dem",]
        for dem_name in dems_raster_names:
            # Add dem and aspect data
            if (not predict) or (predict and dem_name in pc_columns):
                dem_path = get_dem_from_tile(tile,mongo_products_collection,minio_client, dem_name)
                band_no_data_value = no_data_value.get(dem_name,-99999)
                band_normalize_range = normalize_range.get(dem_name,None)
                raster = read_raster(
                                band_path=dem_path,
                                rescale=True,
                                no_data_value=band_no_data_value,
                                normalize_range=band_normalize_range, 
                                path_to_disk=str(Path(settings.TMP_DIR,'visualization',f'{dem_name}.tif')),
                        )
                raster_masked = np.ma.masked_array(raster, mask=crop_mask)                    
                raster_masked = np.ma.compressed(raster_masked).flatten()
                raster_df = pd.DataFrame({dem_name: raster_masked})
                tile_df = pd.concat([tile_df, raster_df], axis=1)

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

            # (Optional) For validate dataset geometries, the product name is added.
            raster_product_name = np.full_like(raster_masked, product_name, dtype=object)
            raster_df = pd.DataFrame({f"{season}_product_name": raster_product_name})
            tile_df = pd.concat([tile_df, raster_df], axis=1)

            (rasters_paths, is_band) = get_product_rasters_paths(product_metadata, minio_client, current_bucket)
            # In predict phase, use only pca-selected rasters
            if predict:
                (rasters_paths, is_band) = filter_rasters_paths_by_pca(rasters_paths, is_band, pc_columns, season)
            temp_product_folder = Path(settings.TMP_DIR, product_name + '.SAFE')
            if not temp_product_folder.exists():
                Path.mkdir(temp_product_folder)
            print(f"Processing product {product_name}")

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
                band_normalize_range = normalize_range.get(raster_name,None)
                if is_band[i] and (band_normalize_range is None):
                    band_normalize_range = (0,7000)

                path_to_disk = None
                if visualization:
                    path_to_disk = str(Path(settings.TMP_DIR,'visualization',raster_filename))
                raster = read_raster(
                            band_path=temp_path, 
                            rescale=True, 
                            no_data_value=band_no_data_value,
                            path_to_disk=path_to_disk,
                            normalize_range=band_normalize_range,
                        )
                raster_masked = np.ma.masked_array(raster[0], mask=crop_mask)                    
                raster_masked = np.ma.compressed(raster_masked)


                #raster_df = pd.DataFrame({raster_name: raster_masked})
                raster_df = pd.DataFrame({f"{season}_{raster_name}": raster_masked})

                tile_df = pd.concat([tile_df, raster_df], axis=1)
                if temp_path.exists() and temp_path.is_file():
                    Path.unlink(temp_path)
            
            if temp_product_folder.is_dir() and (not any(Path(temp_product_folder).iterdir())):
                Path.rmdir(temp_product_folder)

        
                
        if not predict:

            raster_masked = np.ma.masked_array(label_lon_lat[:,:,0], mask=crop_mask)
            raster_masked = np.ma.compressed(raster_masked).flatten()
            raster_df = pd.DataFrame({"class": raster_masked})
            tile_df = pd.concat([tile_df, raster_df], axis=1)

            raster_masked = np.ma.masked_array(label_lon_lat[:,:,1], mask=crop_mask)
            raster_masked = np.ma.compressed(raster_masked).flatten()
            raster_df = pd.DataFrame({"longitude": raster_masked})
            tile_df = pd.concat([tile_df, raster_df], axis=1)

            raster_masked = np.ma.masked_array(label_lon_lat[:,:,2], mask=crop_mask)
            raster_masked = np.ma.compressed(raster_masked).flatten()
            raster_df = pd.DataFrame({"latitude": raster_masked})
            tile_df = pd.concat([tile_df, raster_df], axis=1)
            
 
            tile_df.to_csv(f"./tiles_datasets/dataset_{tile}.csv", index=False)

            if final_df is None:
                final_df = tile_df
            else:
                final_df = pd.concat([final_df, tile_df], axis=0)

        if predict:

            kwargs_10m['nodata'] = 0
            clf = joblib.load('model.joblib')
            predict_df = tile_df
            predict_df.sort_index(inplace=True, axis=1)
            predict_df = predict_df.replace([np.inf, -np.inf], np.nan)
            nodata_rows = np.isnan(predict_df).any(axis=1)
            predict_df.fillna(0, inplace=True)
            predictions = clf.predict(predict_df)
            predictions[nodata_rows] = "nodata"
            predictions = np.reshape(predictions, (1, kwargs_10m['height'],kwargs_10m['width']))
            encoded_predictions = predictions.copy()
            mapping = {"nodata":0,"beaches":1,"bosqueRibera":2,"cities":3,"dehesas":4,"matorral":5,"pastos":6,"plantacion":7,"rocks":8,"water":9,"wetland":10,"agricola":11}
            for class_, value in mapping.items():
                encoded_predictions = np.where(encoded_predictions == class_, value, encoded_predictions)

            kwargs_10m["driver"] = "GTiff"
            with rasterio.open(str(Path(settings.TMP_DIR,f'classification_{tile}.tif')), "w", **kwargs_10m) as classification_file:
                classification_file.write(encoded_predictions)
            print(f'classification_{tile}.tif saved')
            
        
    if not predict:
        print(final_df.head())  
        final_df.to_csv("dataset.csv", index=False)


if __name__ == '__main__':
    workflow(training=True, visualization=False, predict=False)
