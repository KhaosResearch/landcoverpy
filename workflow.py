import io
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import rasterio
import pandas as pd
import pyproj
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

def get_minio():
   # Connect with minio
    return Minio(
        settings.MINIO_HOST + ":" + str(settings.MINIO_PORT),
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False,
    )

def get_mongo_collection():
    # Return a collection from Mongo DB
    mongo_client = MongoClient(
        "mongodb://" + settings.MONGO_HOST + ":" + str(settings.MONGO_PORT) + "/",
        username=settings.MONGO_USERNAME,
        password=settings.MONGO_PASSWORD,
    )

    return mongo_client[settings.MONGO_DB][settings.MONGO_COLLECTION]

def find_product_image(pattern: str, products_folder: str = settings.TMP_DIR) -> Path:
    """
    Finds image matching a pattern in the product folder with glob.
    :param pattern: A pattern to match.
    :return: A Path object pointing to the first found image.
    """
    return (
        [
            f
            for f in Path(products_folder).glob(
                "GRANULE/*/IMG_DATA/**/*" + pattern + ".jp2"
            )
        ]
        + [f for f in Path(products_folder).glob("*" + pattern + ".jp2")]
    )[0]

def project_shape(geom, scs: str = 'epsg:4326', dcs: str = 'epsg:32630'):
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

def __read_band(band_path: str, mask_geometry: dict = None, scs: str = "epsg:32629", dcs: str = "epsg:4326"):
    '''
    Reads a Sentinel band as a numpy array. It scales all bands to a 10m/pxl resolution.
    If mask_geometry is given, the resturning band is cropped to the mask
    '''

    with rasterio.open(band_path) as band_file:
        kwargs = band_file.meta
        img_resolution = kwargs["transform"][0]
        scale_factor = img_resolution/10

        destination_crs = band_file.crs

        # Read file 
        band = band_file.read(1).astype(np.float32)
        band = band.reshape((1,*band.shape))

        # Scale de image to a resolution of 10m per pixel
        if img_resolution > 10:
            band = np.repeat(np.repeat(band, scale_factor, axis=1), scale_factor, axis=2)
            # Update band metadata
            kwargs["height"] *= scale_factor
            kwargs["width"] *= scale_factor
            kwargs["transform"] = rasterio.Affine(10, 0.0, kwargs["transform"][2], 0.0, -10, kwargs["transform"][5])

    # Mask geometry if supplied
    if mask_geometry:
        projected_geometry = project_shape(mask_geometry, dcs=destination_crs)
        # Create a temporal memory file to mask the band
        # This is necessary because the band is previously read to scale its resolution
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**kwargs) as memfile_band:
                memfile_band.write(band)
                masked_band, _ = rasterio.mask.mask(memfile_band, shapes=[projected_geometry], crop=True)
                masked_band = masked_band.astype(np.float32)

        band = masked_band

    band[band == 0] = np.nan 
    return band[0] 

def normalize(matrix):
    # Normalize a numpy matrix
    return (matrix - np.nanmean(matrix)) / np.nanstd(matrix)

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

def _get_mgrs_from_geometry(geometry: dict):
    '''
    Get the MGRS coordinates for the 4 corners of the bounding box of a geometry

    This wont work for geometry bigger than a tile. A chunck of the image could not fit between the products of the 4 corners
    '''
    tiles = set() 
    corners = _get_corners(geometry)
    for point in corners.values():
        tiles.add(MGRS().toMGRS(*point, MGRSPrecision=0))

    return tiles

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
    polygons_per_tile = group_polygons_by_tile(Path("forest-db/Forest_DB_SNieves.geojson"))

    # Tiles related to the traininig zone
    tiles = polygons_per_tile.keys() 
    # This parameters should be coming from somewhere else
    bands = ["AOT_10m", "B01_60m", "B02_10m", "B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B08_10m", "B09_60m", "B11_20m", "B12_20m", "B8A_20m", "WVP_10m"] # Removed , "SCL_20m"
    spring_start = datetime(2020, 3, 1)
    spring_end = datetime(2020, 3, 31)
    summer_start = datetime(2020, 7, 1)
    summer_end = datetime(2020, 7, 31)
    autumn_start = datetime(2020, 10, 1)
    autumn_end = datetime(2020, 10, 31)
    
    # Step 1 
    minio_client = get_minio()
    mongo_col = get_mongo_collection()
    
    # Model input
    train_df = None

    # Search product metadata in Mongo
    for tile in tiles:
        for geometry_id in (0,4, 69): # Take some ids to speed up the demo  
            print(f"Working through tiles {tile}")
            # TODO Filter por fecha, take 3 of every year y añadir todas las bandas
            # Query sample {"title": {"$regex": "_T30SUF_"}, "date": {"$gte": ISODate("2020-07-01T00:00:00.000+00:00"),"$lte": ISODate("2020-07-31T23:59:59.999+00:00")}}
            product_data_spring = mongo_col.find({
                "title": {
                    "$regex": f"_T{tile}_"
                },
                "date": {
                    "$gte": spring_start,
                    "$lte": spring_end,
                },
            })
            product_data_summer = mongo_col.find({
                "title": {
                    "$regex": f"_T{tile}_"
                },
                "date": {
                    "$gte": summer_start,
                    "$lte": summer_end,
                },
            })
            product_data_autumn = mongo_col.find({
                "title": {
                    "$regex": f"_T{tile}_"
                },
                "date": {
                    "$gte": autumn_start,
                    "$lte": autumn_end,
                },
            })

            product_per_season = {
                "spring": list(product_data_spring),
                "summer": list(product_data_summer),
                "autumn": list(product_data_autumn),
            } 

            season_df = None

            for season, products in product_per_season.items():
                if len(products) == 1:
                    product = products[0]
                else:
                # TODO integrate the making of a composite
                # TODO Should the composites be made before hand?  
                # product = _make_composite(products)
                    warnings.warn(f"{len(products)} products available for {season}, but composites not implemented, taking only first")
                    product = products[0]

                year = product["date"].strftime("%Y")
                month = product["date"].strftime("%B")
                minio_dir = year + "/" + month + "/"
                bands_dir = minio_dir +  product["title"] + "/raw/"
                temp_product_folder = settings.TMP_DIR + "/" + product["title"] + ".SAFE"

                # Create product dataframe
                single_product_df = None

                # Get all sentinel bands
                for band_name in bands:
                    band_file = band_name + ".jp2"
                    print("Reading band: ->", settings.MINIO_BUCKET_NAME, bands_dir + band_file, temp_product_folder + "/" + band_file)
                    
                    minio_client.fget_object(
                        settings.MINIO_BUCKET_NAME,
                        bands_dir + band_file,
                        temp_product_folder + "/" + band_file,
                    )

                    # Once all bands are loaded, they need to be passed to feature selection (PCA, regression, matusita)
                    band = __read_band(temp_product_folder + "/" + band_file, mask_geometry = polygons_per_tile[tile][geometry_id]["geometry"])
                    
                    normalized_band = normalize(band)
                    
                    band_df = pd.DataFrame({f"{season}_{band_name}": normalized_band.flatten()})
                    
                    band_df= band_df.dropna()
                    
                    if single_product_df is None:
                        single_product_df = band_df
                    else: 
                        single_product_df = pd.concat([single_product_df, band_df], axis=1)
                
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
    train_df.to_csv("dataset.csv")
    # Temporal code to load premade dataset for training 
    # train_df = pd.read_csv("dataset2.csv")
    # train_df = train_df.drop("Unnamed: 0", axis=1)
    # print(train_df.head())
    if training:
        # Prepare data for training
        x_train_data = train_df.drop("class", axis=1)
        y_train_data = train_df["class"] 

        # Filter bands according to PCA, matusita etc
        pc_columns = pca(x_train_data)
        print(pc_columns)
        reduced_x_train_data = x_train_data[pc_columns]
        
        # Train model 
        clf = RandomForestClassifier()
        clf.fit(reduced_x_train_data, y_train_data)
        joblib.dump(clf, 'model.pkl', compress=1)
    else: # Prediction
        
        reduced_predict_data = train_df[['B02_10m', 'AOT_10m']] # This should be coming from the PCA used during training
        clf = joblib.load('model.pkl')
        results = clf.predict(reduced_predict_data)
        print(results)

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