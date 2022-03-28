from datetime import datetime
from pathlib import Path
import joblib
from os.path import join
import numpy as np
from glob import glob
import pandas as pd
import rasterio
from typing import List
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
    download_sample_band,
    safe_minio_execute,
)



def workflow(visualization: bool, predict: bool, tiles_to_predict: List[str] = None):
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

    # This parameters should be coming from somewhere else
    spring_start = datetime(2021, 3, 1)
    spring_end = datetime(2021, 3, 31)
    summer_start = datetime(2021, 6, 1)
    summer_end = datetime(2021, 6, 30)
    autumn_start = datetime(2021, 11, 1)
    autumn_end = datetime(2021, 11, 30)
    
    # Step 1 
    minio_client = get_minio()
    mongo_products_collection = connect_mongo_products_collection()
    


    # Names of the indexes that are taken into account
    indexes_used = ["cri1","ri","evi2","mndwi","moisture","ndyi","ndre","ndvi","osavi"]
    # Name of the sentinel bands that are ignored
    skip_bands = ["tci"]
    # PCA resulting columns, this should come from somewhere else
    pc_columns = sorted(["slope","aspect","dem","spring_cri1","spring_ri","spring_evi2","spring_mndwi","spring_moisture","spring_ndyi","spring_ndre","spring_ndvi","spring_osavi","spring_AOT","spring_B01","spring_B02","spring_B03","spring_B04","spring_B05","spring_B06","spring_B07","spring_B08","spring_B09","spring_B11","spring_B12","spring_B8A","summer_cri1","summer_ri","summer_evi2","summer_mndwi","summer_moisture","summer_ndyi","summer_ndre","summer_ndvi","summer_osavi","summer_AOT","summer_B01","summer_B02","summer_B03","summer_B04","summer_B05","summer_B06","summer_B07","summer_B08","summer_B09","summer_B11","summer_B12","summer_B8A","autumn_cri1","autumn_ri","autumn_evi2","autumn_mndwi","autumn_moisture","autumn_ndyi","autumn_ndre","autumn_ndvi","autumn_osavi","autumn_AOT","autumn_B01","autumn_B02","autumn_B03","autumn_B04","autumn_B05","autumn_B06","autumn_B07","autumn_B08","autumn_B09","autumn_B11","autumn_B12","autumn_B8A"])
    # Ranges for normalization of each raster
    normalize_range = {"slope":(0,70), "aspect":(0,360), "dem":(0,2000)}

    if predict:
        print("Predicting tiles")
        if tiles_to_predict is not None:
            polygons_per_tile = {}
            for tile_to_predict in tiles_to_predict:
                polygons_per_tile[tile_to_predict] = []
    else:
        print("Creating dataset from tiles")

    # Tiles related to the traininig zone
    tiles = polygons_per_tile.keys() 

    for i, tile in enumerate(tiles):
        print(f"Working in tile {tile}, {i}/{len(tiles)}")
        # Mongo query for obtaining valid products
        max_cloud_percentage=20
        product_metadata_cursor_spring = get_products_by_tile_and_date(tile, mongo_products_collection, spring_start, spring_end, max_cloud_percentage)
        product_metadata_cursor_summer = get_products_by_tile_and_date(tile, mongo_products_collection, summer_start, summer_end, max_cloud_percentage)
        product_metadata_cursor_autumn = get_products_by_tile_and_date(tile, mongo_products_collection, autumn_start, autumn_end, max_cloud_percentage)

        product_per_season = {
                "spring": list(product_metadata_cursor_spring)[:5],
                "autumn": list(product_metadata_cursor_autumn)[:5],
                "summer": list(product_metadata_cursor_summer)[:5],
            }
    
        if len(product_per_season["spring"]) == 0 or len(product_per_season["autumn"]) == 0 or len(product_per_season["summer"]) == 0:
            print(f"There is no valid data for tile {tile}. Skipping it...")
            continue

        print(f"Working through tile {tile}")
    

        # Dataframe for storing data of a tile
        tile_df = None    

        dems_raster_names = ["slope", "aspect", "dem",]
        for dem_name in dems_raster_names:
            # Add dem and aspect data
            if (not predict) or (predict and dem_name in pc_columns):
                dem_path = get_dem_from_tile(tile,mongo_products_collection,minio_client, dem_name)

                # Predict doesnt work yet in some specific tiles where tile is not fully contained in aster rasters
                kwargs = _get_kwargs_raster(dem_path)    
                crop_mask, label_lon_lat = mask_polygons_by_tile(polygons_per_tile, tile, kwargs)
                if predict:
                    crop_mask = np.zeros_like(crop_mask)

                band_normalize_range = normalize_range.get(dem_name,None)
                raster = read_raster(
                                band_path=dem_path,
                                rescale=True,
                                normalize_range=band_normalize_range, 
                                path_to_disk=str(Path(settings.TMP_DIR,'visualization',f'{dem_name}.tif')),
                        )
                raster_masked = np.ma.masked_array(raster, mask=crop_mask)                    
                raster_masked = np.ma.compressed(raster_masked).flatten()
                raster_df = pd.DataFrame({dem_name: raster_masked})
                tile_df = pd.concat([tile_df, raster_df], axis=1)

        #Get crop mask for sentinel rasters and dataset labeled with database points in tile
        band_path = download_sample_band(tile, minio_client, mongo_products_collection)
        kwargs = _get_kwargs_raster(band_path)    
        crop_mask, label_lon_lat = mask_polygons_by_tile(polygons_per_tile, tile, kwargs)

        if predict:
            crop_mask = np.zeros_like(crop_mask)

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
            if not predict:
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

                # Only keep bands and indexes in indexes_used
                if (not is_band[i]) and (not any(raster_name.upper() == index_used.upper() for index_used in indexes_used)):
                    continue
                # Skip bands in skip_bands
                if is_band[i] and any(raster_name.upper() == band_skipped.upper() for band_skipped in skip_bands):
                    continue
                # Read only the first band to avoid duplication of different spatial resolution
                if any(raster_name.upper() == read_raster.upper() for read_raster in already_read):
                    continue
                already_read.append(raster_name)

                print(f"Downloading raster {raster_name} from minio into {temp_path}")
                safe_minio_execute(
                    func = minio_client.fget_object,
                    bucket_name=current_bucket,
                    object_name=raster_path,
                    file_path=str(temp_path),
                )
                kwargs = _get_kwargs_raster(str(temp_path))
                spatial_resolution = kwargs["transform"][0]
                if spatial_resolution == 10:
                    kwargs_10m = kwargs

                band_normalize_range = normalize_range.get(raster_name,None)
                if is_band[i] and (band_normalize_range is None):
                    band_normalize_range = (0,7000)

                path_to_disk = None
                if visualization:
                    path_to_disk = str(Path(settings.TMP_DIR,'visualization',raster_filename))
                raster = read_raster(
                            band_path=temp_path, 
                            rescale=True,
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
            mapping = {"nodata":0,"beaches":1,"bosqueRibera":2,"cities":3,"dehesas":4,"matorral":5,"pastos":6,"plantacion":7,"rocks":8,"water":9,"wetland":10,"agricola":11, "bosque":12, "bosqueAbierto": 13}
            for class_, value in mapping.items():
                encoded_predictions = np.where(encoded_predictions == class_, value, encoded_predictions)

            kwargs_10m["driver"] = "GTiff"
            classification_name = f"classification_{tile}.tif"
            classification_path = str(Path(settings.TMP_DIR,classification_name))
            with rasterio.open(classification_path, "w", **kwargs_10m) as classification_file:
                classification_file.write(encoded_predictions)
            print(f"{classification_name} saved")
            safe_minio_execute(
                    func = minio_client.fput_object,
                    bucket_name = settings.MINIO_BUCKET_CLASSIFICATIONS,
                    object_name =  f"{settings.MINIO_DATA_FOLDER_NAME}/{classification_name}",
                    file_path=classification_path,
                    content_type="image/tif"
            )
            
        
    if not predict:
        print(final_df.head()) 
        file_name =  "dataset.csv"
        final_df.to_csv(file_name, index=False)
        safe_minio_execute(
                func = minio_client.fput_object,
                bucket_name = settings.MINIO_BUCKET_DATASETS,
                object_name =  f"{settings.MINIO_DATA_FOLDER_NAME}/{file_name}",
                file_path=file_name,
                content_type="text/csv"
            )


if __name__ == '__main__':
    workflow(visualization=False, predict=False, tiles_to_predict=None)
