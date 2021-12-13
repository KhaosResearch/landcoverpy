from ctypes import ArgumentError
from enum import unique
from logging import exception
from typing import Iterable, List, Tuple
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from config import settings
from datetime import datetime
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from pathlib import Path
from tqdm import tqdm
from mgrs import MGRS
from config import settings
from minio import Minio
import pyproj
from shapely.geometry import Point
from shapely.ops import transform, shape
import warnings
import rasterio
from functools import partial
from hashlib import sha256
from itertools import compress
from raw_index_calculation_composite import calculate_raw_indexes
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA




def get_minio():
   # Connect with minio
    return Minio(
        settings.MINIO_HOST + ":" + str(settings.MINIO_PORT),
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False,
    )

def get_products_by_tile_and_date(tile: str, mongo_collection: Collection, start_date: datetime, end_time: datetime) -> Cursor:
    product_metadata_cursor = mongo_collection.find({
                    "title": {
                        "$regex": f"_T{tile}_"
                    },
                    "date": {
                        "$gte": start_date,
                        "$lte": end_time,
                    },
                })

    return product_metadata_cursor


def connect_mongo_products_collection():
    # Return a collection from Mongo DB
    mongo_client = MongoClient(
        "mongodb://" + settings.MONGO_HOST + ":" + str(settings.MONGO_PORT) + "/",
        username=settings.MONGO_USERNAME,
        password=settings.MONGO_PASSWORD,
    )

    return mongo_client[settings.MONGO_DB][settings.MONGO_PRODUCTS_COLLECTION]

def connect_mongo_composites_collection():
    # Return a collection from Mongo DB
    mongo_client = MongoClient(
        "mongodb://" + settings.MONGO_HOST + ":" + str(settings.MONGO_PORT) + "/",
        username=settings.MONGO_USERNAME,
        password=settings.MONGO_PASSWORD,
    )

    return mongo_client[settings.MONGO_DB][settings.MONGO_COMPOSITES_COLLECTION]

def group_polygons_by_tile(geojson_file: Path) -> dict:
    '''
    Extract all features from a GeoJSON file and group them on the tiles they intersect
    '''
    geojson = read_geojson(geojson_file)
    tiles = {} 

    print(f"Querying relevant tiles for {len(geojson['features'])} features")
    for feature in tqdm(geojson["features"]):
        small_geojson = {"type": "FeatureCollection", "features": [feature] }
        footprint = geojson_to_wkt(small_geojson)
        geometry = small_geojson['features'][0]['geometry']
        classification_label = small_geojson['features'][0]['properties']['Nombre']
        cover_percent = small_geojson['features'][0]['properties']['Cobertura']
        intersection_tiles = _get_mgrs_from_geometry(geometry)

        for tile in intersection_tiles:
            if tile not in tiles:
                tiles[tile] = []

            tiles[tile].append({
                "label": classification_label,
                "cover": cover_percent,
                "geometry": geometry,
            })

    return tiles

def _get_mgrs_from_geometry(geometry: dict):
    '''
    Get the MGRS coordinates for the 4 corners of the bounding box of a geometry

    This wont work for geometry bigger than a tile. A chunk of the image could not fit between the products of the 4 corners
    '''
    tiles = set() 
    corners = _get_corners(geometry)
    for point in corners.values():
        tiles.add(MGRS().toMGRS(*point, MGRSPrecision=0))

    return tiles

def _get_corners(geometry: dict):
    '''
    Get the coordinates of the 4 corners of the bounding box of a geometry
    '''
    coordinates = geometry["coordinates"][0] # Takes only the outer ring of the polygon: https://geojson.org/geojson-spec.html#polygon
    lon = [] 
    lat = [] 
    for coordinate in coordinates:
        lon.append(coordinate[0])
        lat.append(coordinate[1])

    max_lon = max(lon)
    min_lon = min(lon)
    max_lat = max(lat)
    min_lat = min(lat)

    return {
        "top_left": (max_lat, min_lon),
        "top_right": (max_lat, max_lon),
        "bottom_left": (min_lat, min_lon),
        "bottom_right": (min_lat, max_lon),
    } 

def _get_centroid(geometry: dict):
    '''
    Only works for non-self-intersecting closed polygon. The vertices are assumed
    to be numbered in order of their occurrence along the polygon's perimeter;
    furthermore, the vertex ( xn, yn ) is assumed to be the same as ( x0, y0 ),
    meaning i + 1 on the last case must loop around to i = 0.

    Source: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
    '''
    coordinates = geometry["coordinates"][0] # Takes only the outer ring of the polygon: https://geojson.org/geojson-spec.html#polygon
    lon = [] 
    lat = [] 
    for coordinate in coordinates:
        lon.append(coordinate[0])
        lat.append(coordinate[1])

    # Calculate sums
    sum_lon = 0
    sum_lat = 0
    sum_A = 0
    for i in range(len(lon)-1):
        sum_lon += (lon[i] + lon[i+1]) * ((lon[i] * lat[i+1]) - (lon[i+1] * lat[i]))
        sum_lat += (lat[i] + lat[i+1]) * ((lon[i] * lat[i+1]) - (lon[i+1] * lat[i]))
        sum_A +=                         ((lon[i] * lat[i+1]) - (lon[i+1] * lat[i]))

    # Calculate area inside the polygon's signed area
    A = sum_A/2

    # Calculate centroid coordinates
    Clon = (1/(6*A)) * sum_lon
    Clat = (1/(6*A)) * sum_lat

    return (Clat, Clon)

def get_product_rasters_paths(product_metadata: dict, minio_client: Minio, minio_bucket: str) -> Tuple[Iterable[str], Iterable[bool]]:

    product_title = product_metadata["title"]

    if minio_bucket == settings.MINIO_BUCKET_NAME_PRODUCTS:
        year = product_metadata["date"].strftime("%Y")
        month = product_metadata["date"].strftime("%B")
        product_dir = f"{year}/{month}/{product_title}"

    elif minio_bucket == settings.MINIO_BUCKET_NAME_COMPOSITES:
        product_dir = product_title

    bands_dir = f"{product_dir}/raw/"
    indexes_dir = f"{product_dir}/indexes/{product_title}/"

    bands_paths = minio_client.list_objects(minio_bucket, prefix=bands_dir)
    indexes_path = minio_client.list_objects(minio_bucket, prefix=indexes_dir)

    rasters = []
    is_band = []
    for index_path in indexes_path:
        rasters.append(index_path.object_name)
        is_band.append(False)
    for band_path in bands_paths:
        rasters.append(band_path.object_name)
        is_band.append(True)

    return (rasters, is_band)

def _project_shape(geom, scs: str = 'epsg:4326', dcs: str = 'epsg:32630'):
    """ Project a shape from a source coordinate system to another one.
    The source coordinate system can be obtain with `rasterio` as illustrated next:

    >>> import rasterio
    >>> print(rasterio.open('example.jp2').crs)

    This is useful when the geometry has its points in "normal" coordinate reference systems while the geoTiff/jp2 data
    is expressed in other coordinate system..

    :param geom: Geometry, e.g., [{'type': 'Polygon', 'coordinates': [[(1,2), (3,4), (5,6), (7,8), (9,10)]]}]
    :param scs: Source coordinate system.
    :param dcs: Destination coordinate system.
    """
    # TODO remove this warning catcher 
    # This disables FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6 in_crs_string = _prepare_from_proj_string(in_crs_string)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        project = partial(
            pyproj.transform,
            pyproj.Proj(init=scs),
            pyproj.Proj(init=dcs))

    return transform(project, shape(geom))

def read_raster(band_path: str, mask_geometry: dict = None, rescale: bool = False, no_data_value: int = 0, path_to_disk: str = None, normalize_raster: bool = False):
    '''
    Reads a Sentinel band as a numpy array. It scales all bands to a 10m/pxl resolution.
    If mask_geometry is given, the resturning band is cropped to the mask
    '''
    band_name = get_raster_name_from_path(str(band_path))
    print(f"Reading raster {band_name}")
    with rasterio.open(band_path) as band_file:
        # Read file 
        kwargs = band_file.meta
        destination_crs = band_file.crs
        band = band_file.read().astype(np.float32)
    print(f"Done")

    # Just in case...
    if len(band.shape) == 2:
        band = band.reshape((kwargs['count'],*band.shape))

    if 'JP2' in kwargs['driver']:
            kwargs['dtype'] = 'float32'
            kwargs['driver'] = 'GTiff'
            kwargs['nodata'] = 0
            band[band == 0] = np.nan
            band = band.astype(np.float32)

            if path_to_disk is not None:
                path_to_disk = path_to_disk[:-3] + 'tif'

    if normalize_raster:
        band = normalize(band)
        
    if rescale:
        img_resolution = kwargs["transform"][0]
        scale_factor = img_resolution/10
        # Scale the image to a resolution of 10m per pixel
        if img_resolution != 10:
            
            new_kwargs = kwargs.copy()
            new_kwargs["height"] = int(kwargs["height"] * scale_factor)
            new_kwargs["width"] = int(kwargs["width"] * scale_factor)
            new_kwargs["transform"] = rasterio.Affine(10, 0.0, kwargs["transform"][2], 0.0, -10, kwargs["transform"][5])
           
            rescaled_raster = np.ndarray(shape=(new_kwargs["height"],new_kwargs["width"]),dtype=np.float32)


            print(f"Rescaling raster, from: {img_resolution}m to 10.0m")
            reproject(
                source=band,
                destination=rescaled_raster,
                src_transform=kwargs['transform'],
                src_crs=kwargs['crs'],
                dst_resolution=(new_kwargs['width'], new_kwargs['height']),
                dst_transform=new_kwargs['transform'],
                dst_crs=new_kwargs['crs'],
                resampling=Resampling.nearest,
            )
            band = rescaled_raster.reshape((new_kwargs['count'],*rescaled_raster.shape))
            kwargs = new_kwargs
            # Update band metadata
            print("Done")

    # Create a temporal memory file to mask the band
    # This is necessary because the band is previously read to scale its resolution
    if mask_geometry:
        print(f"Cropping raster {band_name}")
        projected_geometry = _project_shape(mask_geometry, dcs=destination_crs)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**kwargs) as memfile_band:
                memfile_band.write(band)
                masked_band, _ = rasterio.mask.mask(memfile_band, shapes=[projected_geometry], crop=True, nodata=np.nan)
                masked_band = masked_band.astype(np.float32)
                band = masked_band
        
        new_kwargs = kwargs.copy()
        corners = _get_corners(mask_geometry)
        top_left_corner = corners['top_left']
        top_left_corner = (top_left_corner[1],top_left_corner[0])
        project = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), new_kwargs['crs'], always_xy=True).transform
        top_left_corner = transform(project, Point(top_left_corner))
        new_kwargs["transform"] = rasterio.Affine(new_kwargs["transform"][0], 0.0, top_left_corner.x, 0.0, new_kwargs["transform"][4], top_left_corner.y)
        new_kwargs["width"] = band.shape[2]
        new_kwargs["height"] = band.shape[1]
        kwargs = new_kwargs
        print("Done")
    band[band == no_data_value] = np.nan

    if path_to_disk is not None:
        with rasterio.open(path_to_disk, "w", **kwargs) as dst_file:
            dst_file.write(band)

    return band

def normalize(matrix):
    # Normalize a numpy matrix
    try:
        matrix[matrix == -np.inf] = np.nan
        matrix[matrix == np.inf] = np.nan
        normalized_matrix = (matrix.astype(dtype=np.float32) - np.nanmean(matrix)) / np.nanstd(matrix)
    except exception:
        normalized_matrix = matrix
    return normalized_matrix


def composite(band_paths: Iterable[str], band_name: str, method: str = "median"):
    """
    Calculates de composite between a series of bands

    :param band_paths: List of paths to calculate the composite from.
    :param method: To calculate the composite. Values: "median", "mean".
    """
    composite_bands = []
    composite_kwargs = None
    for band_path in band_paths:
        if composite_kwargs is None:
            composite_kwargs = _get_kwargs_raster(band_path)
        composite_bands.append(read_raster(band_path))
        print(f"Removing file {str(band_path)}")
        Path.unlink(Path(band_path))

    shapes = [np.shape(band) for band in composite_bands] 

    # Check if all arrays are of the same shape 
    if not np.all(np.array(list(map(lambda x: x == shapes[0], shapes)))):  
        raise ValueError(f"Not all bands have the same shape\n{shapes}")

    if 'SCL' in band_name:
        composite_out =  composite_bands[0] # TODO union de los pixeles clasificados como 4
    elif method == "mean":
        composite_out = np.mean(composite_bands, axis=0)
    elif method == "median":
        composite_out = np.median(composite_bands, axis=0)
    else:
        raise ValueError(f"Method '{method}' is not recognized.")

    return (composite_out, composite_kwargs)

def _get_id_composite(products_ids: List[str]) -> str:
    products_ids.sort()
    concat_ids = "".join(products_ids)
    hashed_ids = sha256(concat_ids.encode('utf-8')).hexdigest()
    return hashed_ids

def _get_title_composite(products_dates: List[str], products_tiles: List[str], composite_id: str) -> str:
    if not all(product_tile==products_tiles[0] for product_tile in products_tiles):
        raise ValueError(f"Error creating composite, products have different tile: {products_tiles}")
    tile = products_tiles[0]
    first_product_date, last_product_date = min(products_dates), max(products_dates)
    first_product_date = first_product_date.split('T')[0]
    last_product_date = last_product_date.split('T')[0]
    composite_title = f"S2X_MSIL2A_{first_product_date}_NXXX_RXXX_{tile}_{last_product_date}_{composite_id[:8]}"
    return composite_title

# Intentar leer de mongo si existe algun composite con esos products
def get_composite(products_metadata: Iterable[dict], mongo_collection: Collection) -> dict:
    products_ids = [products_metadata["id"] for products_metadata in products_metadata] 
    hashed_id_composite = _get_id_composite(products_ids)
    composite_metadata = mongo_collection.find_one({"id":hashed_id_composite})
    return composite_metadata

def create_composite(products_metadata: Iterable[dict], minio_client: Minio, bucket_products: str, bucket_composites: str, mongo_composites_collection: Collection) -> None:

    products_ids = [] 
    products_titles = []
    products_dates = []
    products_tiles = []
    bands_paths_products = []

    for product_metadata in products_metadata:

        product_title = product_metadata["title"]
        products_ids.append(product_metadata["id"])
        products_titles.append(product_title)
        products_dates.append(product_title.split("_")[2])
        products_tiles.append(product_title.split("_")[5])

        (rasters_paths,is_band) = get_product_rasters_paths(product_metadata, minio_client, bucket_products)
        bands_paths_products.append(list(compress(rasters_paths, is_band)))

    composite_id = _get_id_composite(products_ids)
    composite_title = _get_title_composite(products_dates, products_tiles, composite_id) 
    temp_path_composite = Path(settings.TMP_DIR, composite_title + '.SAFE')

    print("Creating composite of ", len(products_titles), " products: ", products_titles)
    uploaded_composite_band_paths = []
    temp_paths_composite_bands = []
    temp_product_dirs = []
    result = None
    try:
        num_bands = len(bands_paths_products[0])
        for i_band in range(num_bands):
            products_i_band_path = [bands_paths_product[i_band] for bands_paths_product in bands_paths_products]
            band_name = get_raster_name_from_path(products_i_band_path[0])
            band_filename = get_raster_filename_from_path(products_i_band_path[0])

            temp_path_composite_band = Path(temp_path_composite, band_filename)

            temp_path_list = []
            for product_i_band_path in products_i_band_path:
                product_title = product_i_band_path.split('/')[2]

                temp_dir_product = f"{settings.TMP_DIR}/{product_title}.SAFE"
                temp_path_product_band = f"{temp_dir_product}/{band_filename}"
                print(f"Reading raster: -> {bucket_products}:{product_i_band_path} into {temp_path_product_band}")
                minio_client.fget_object(
                                bucket_name=bucket_products,
                                object_name=product_i_band_path,
                                file_path=str(temp_path_product_band),
                            )
                if temp_dir_product not in temp_product_dirs:
                    temp_product_dirs.append(temp_dir_product)
                temp_path_list.append(temp_path_product_band)

            composite_i_band, kwargs_composite = composite(temp_path_list, band_name)

            # Save raster to disk
            if not Path.is_dir(temp_path_composite):
                Path.mkdir(temp_path_composite)
            # Update kwargs to reflect change in data type.
            temp_paths_composite_bands.append(temp_path_composite_band)
            with rasterio.open(temp_path_composite_band, "w", **kwargs_composite) as file_composite:
                print(composite_i_band.shape, kwargs_composite["count"])
                file_composite.write(composite_i_band)


            # Upload raster to minio
            minio_band_path = f"{composite_title}/raw/{band_filename}"
            minio_client.fput_object(
                bucket_name = bucket_composites,
                object_name =  minio_band_path,
                file_path=temp_path_composite_band,
                content_type="image/jp2"
            )
            uploaded_composite_band_paths.append(minio_band_path)
            print(f"Uploaded raster: -> {temp_path_composite_band} into {bucket_composites}:{minio_band_path}") 

        composite_metadata = dict()
        composite_metadata["id"] = composite_id
        composite_metadata["title"] = composite_title
        composite_metadata["products"] = [dict(id=product_id, title=products_title) for (product_id,products_title) in zip(products_ids,products_titles)]
        composite_metadata["first_date"] = sentinel_date_to_datetime(min(products_dates))
        composite_metadata["last_date"] = sentinel_date_to_datetime(max(products_dates))

        # Upload metadata to mongo
        result = mongo_composites_collection.insert_one(composite_metadata)
        print("Inserted data in mongo, id: ", result.inserted_id)

        # Compute indexes
        calculate_raw_indexes(_get_id_composite([products_metadata["id"] for products_metadata in products_metadata]))

    except (Exception,KeyboardInterrupt) as e:
        print("Removing uncompleted composite from minio")
        for composite_band in uploaded_composite_band_paths:
            minio_client.remove_object(
                bucket_name=bucket_composites,
                object_name=composite_band
            )
        products_ids = [products_metadata["id"] for products_metadata in products_metadata] 
        mongo_composites_collection.delete_one({'id':_get_id_composite(products_ids)})
        raise e

    finally:
        for composite_band in temp_paths_composite_bands:
            print(f"Removing file {str(composite_band)}")
            Path.unlink(Path(composite_band))
        print(f"Removing folder {str(temp_path_composite)}")
        Path.rmdir(Path(temp_path_composite))
        for product_dir in temp_product_dirs:
            print(f"Removing folder {str(product_dir)}")
            Path.rmdir(Path(product_dir))


def get_raster_filename_from_path(raster_path):
    return raster_path.split('/')[-1]

def get_raster_name_from_path(raster_path):
    raster_filename = get_raster_filename_from_path(raster_path)
    return raster_filename.split('.')[0].split('_')[0]

def _get_kwargs_raster(raster_path):
    with rasterio.open(raster_path) as raster_file:
        kwargs = raster_file.meta
        return kwargs

def sentinel_date_to_datetime(date: str):
    date_datetime = datetime(int(date[0:4]),int(date[4:6]),int(date[6:8]),int(date[9:11]),int(date[11:13]),int(date[13:15])) #YYYYMMDDTHHMMSS
    return date_datetime


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
    col_names_list = [column_names[id_] for id_ in col_ids]
    col_names_list = list(dict.fromkeys(col_names_list))
    col_names_list.sort()
    return col_names_list

    