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
from shapely.ops import transform, shape
import warnings
import numpy as np
import rasterio
from functools import partial
from hashlib import sha256
from itertools import compress
from raw_index_calculation_composite import calculate_raw_indexes





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

def read_raster(band_path: str, mask_geometry: dict = None, rescale: bool = False, no_data_value: int = 0):
    '''
    Reads a Sentinel band as a numpy array. It scales all bands to a 10m/pxl resolution.
    If mask_geometry is given, the resturning band is cropped to the mask
    '''
    with rasterio.open(band_path) as band_file:
        # Read file 
        kwargs = band_file.meta
        destination_crs = band_file.crs
        band = band_file.read().astype(np.float32)

    # Just in case...
    if len(band.shape) == 2:
        band = band.reshape((1,*band.shape))
        
    if rescale:
        img_resolution = kwargs["transform"][0]
        scale_factor = img_resolution/10

        # Scale de image to a resolution of 10m per pixel
        if img_resolution > 10:
            band = np.repeat(np.repeat(band, scale_factor, axis=1), scale_factor, axis=2)
            # Update band metadata
            kwargs["height"] *= scale_factor
            kwargs["width"] *= scale_factor
            kwargs["transform"] = rasterio.Affine(10, 0.0, kwargs["transform"][2], 0.0, -10, kwargs["transform"][5])

    # Create a temporal memory file to mask the band
    # This is necessary because the band is previously read to scale its resolution
    if mask_geometry:
        projected_geometry = _project_shape(mask_geometry, dcs=destination_crs)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**kwargs) as memfile_band:
                memfile_band.write(band)
                masked_band, _ = rasterio.mask.mask(memfile_band, shapes=[projected_geometry], crop=True)
                masked_band = masked_band.astype(np.float32)

    band[band == no_data_value] = np.nan

    return band

def normalize(matrix):
    # Normalize a numpy matrix
    try:
        normalized_matrix = (matrix - np.nanmean(matrix)) / np.nanstd(matrix)
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

def _get_title_composite(products_dates: List[str], products_tiles: List[str]) -> str:
    if not all(product_tile==products_tiles[0] for product_tile in products_tiles):
        raise ValueError(f"Error creating composite, products have different tile: {products_tiles}")
    tile = products_tiles[0]
    first_product_date, last_product_date = min(products_dates), max(products_dates)
    composite_title = f"S2X_MSIL2A_{first_product_date}_NXXX_RXXX_{tile}_{last_product_date}"
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

    composite_title = _get_title_composite(products_dates, products_tiles) 
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
            print("Inserted band in minio: ", minio_band_path)    

        composite_metadata = dict()
        composite_metadata["id"] = _get_id_composite(products_ids)
        composite_metadata["title"] = composite_title
        composite_metadata["products"] = [dict(id=product_id, title=products_title) for (product_id,products_title) in zip(products_ids,products_titles)]
        composite_metadata["first_date"] = min(products_dates)
        composite_metadata["last_date"] = max(products_dates)

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
        [Path.unlink(Path(composite_band)) for composite_band in temp_paths_composite_bands]
        Path.rmdir(Path(temp_path_composite))
        [Path.rmdir(Path(product_dir)) for product_dir in temp_product_dirs]


def get_raster_filename_from_path(raster_path):
    return raster_path.split('/')[-1]

def get_raster_name_from_path(raster_path):
    raster_filename = get_raster_filename_from_path(raster_path)
    return raster_filename.split('.')[0]

def _get_kwargs_raster(raster_path):
    with rasterio.open(raster_path) as raster_file:
        kwargs = raster_file.meta
        return kwargs