import io
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import joblib
import rasterio
import pandas as pd
import pyproj
from minio import Minio
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
    Reads a Sentinel band as a numpy array
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
        # This is necessary if the band is previously read to scale its resolution
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

def _dimension_reduction(percentage_variance: list, variance_explained: int):
    cumulative_variance = 0
    n_components = 0
    while cumulative_variance < variance_explained:
        cumulative_variance += percentage_variance[n_components]
        n_components += 1
    return n_components

def pca(data: pd.DataFrame, variance_explained: int = 75):
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

    geojson = read_geojson(Path("forest-db/Forest_DB_SNieves.geojson"))
    tiles = set()
    cache = {} 

    # # Initialize Sentinel client
    # sentinel_api = SentinelAPI(
    #     settings.SENTINEL_USERNAME,
    #     settings.SENTINEL_PASSWORD,
    #     settings.SENTINEL_HOST,
    #     show_progressbars=False,
    # )

    print(f"Querying relevant tiles for {len(geojson['features'])} features")
    for feature in tqdm(geojson["features"]):
        small_geojson = {"type": "FeatureCollection", "features":[feature]}
        footprint = geojson_to_wkt(small_geojson)
        geometry = small_geojson['features'][0]['geometry']
        classification_label = small_geojson['features'][0]['properties']['Nombre']
        
        break
        # TODO: Find centroid of footprint and find the associated tile by coordinates
         
        # # Search is limited to those scenes that intersect with the AOI (area of interest) polygon
        # products = sentinel_api.query(
        #     area=footprint,
        #     filename="S2*",
        #     producttype="S2MSI2A",
        #     platformname="Sentinel-2",
        #     cloudcoverpercentage=(0, 100),
        #     date=(datetime(2020,5,1), datetime(2020,5,31)),
        # )

        # # Get the list of products
        # products_df = sentinel_api.to_dataframe(products)
        # tiles.update(set(map(lambda x: x.split('_')[5], products_df['title'])))

    # Tiles related to the traininig zone
    tiles = ["T30SUF"] # ["T29SPA", "T29SPB", "T29SPC", "T29SPD", "T29SQA", "T29SQB", "T29SQC", "T29SQD", "T29SQV", "T29TMG", "T29TMH", "T29TMJ", "T29TNG", "T29TNH", "T29TNJ", "T29TPE", "T29TPF", "T29TPG", "T29TPH", "T29TPJ", "T29TQE", "T29TQF", "T29TQG", "T29TQH", "T29TQJ", "T30STE", "T30STF", "T30STG", "T30STH", "T30STJ", "T30SUE", "T30SUF", "T30SUG", "T30SUH", "T30SUJ", "T30SVF", "T30SVG", "T30SVH", "T30SVJ", "T30SWF", "T30SWG", "T30SWH", "T30SWJ", "T30SXF", "T30SXG", "T30SXH", "T30SXJ", "T30SYG", "T30SYH", "T30SYJ", "T30TTK", "T30TTL", "T30TTM", "T30TUK", "T30TUL", "T30TUM", "T30TUN", "T30TUP", "T30TVK", "T30TVL", "T30TVM", "T30TVN", "T30TVP", "T30TWK", "T30TWL", "T30TWM", "T30TWN", "T30TWP", "T30TXK", "T30TXL", "T30TXM", "T30TXN", "T30TXP", "T30TYK", "T30TYL", "T30TYM", "T30TYN", "T31SBC", "T31SBD", "T31TBE", "T31TBF", "T31TBG", "T31TCE", "T31TCF", "T31TCG", "T31TCH", "T31TDF", "T31TDG", "T31TDH", "T31TEG", "T31TEH"] 
    bands = ["AOT_10m", "B01_60m", "B02_10m"] #, "B03_10m", "B04_10m", "B05_20m", "B06_20m", "B07_20m", "B08_10m", "B09_60m", "B11_20m", "B12_20m", "B8A_20m", "WVP_10m"] # Removed , "SCL_20m"
    # Step 1 
    minio_client = get_minio()
    mongo_col = get_mongo_collection()
    
    # Model input
    train_data = None

    # Search product metadata in Mongo
    for tile in tiles:
        print(f"Working through tiles {tile}")
        product_data = mongo_col.find({"title": {"$regex": f"_{tile}_"}})

        for product in product_data:
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
                band = __read_band(temp_product_folder + "/" + band_file, mask_geometry = geometry)
                
                normalized_band = normalize(band)
                
                band_df = pd.DataFrame({band_name: normalized_band.flatten()})
                
                band_df= band_df.dropna()
                
                if single_product_df is None:
                    single_product_df = band_df
                else: 
                    single_product_df = pd.concat([single_product_df, band_df], axis=1)
            
            if training:
                # Add classification label
                single_product_df["class"] = classification_label

            if train_data is None:
                train_data = single_product_df
            else: 
                train_data = pd.concat([train_data, single_product_df], axis=0)
                break
        
        if training:
            # Prepare data for training
            x_train_data = train_data.drop("class", axis=1)
            y_train_data = train_data["class"] 

            # Filter bands according to PCA, matusita etc
            pc_columns = pca(x_train_data)
            print(pc_columns)
            reduced_x_train_data = x_train_data[pc_columns]
            
            # Train model 
            clf = RandomForestClassifier()
            clf.fit(reduced_x_train_data, y_train_data)
            joblib.dump(clf, 'model.pkl', compress=1)
        else: # Prediction
            
            reduced_predict_data = train_data[['B02_10m', 'AOT_10m']]
            clf = joblib.load('model.pkl')
            results = clf.predict(reduced_predict_data)
            print(results)

        
        print("STOP")
        break


if __name__ == '__main__':
    print("Training")
    workflow(training=True)
    print("Testing")
    workflow(training=False)