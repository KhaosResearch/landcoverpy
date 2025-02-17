import os
from os.path import join
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
from landcoverpy.utilities.band_arithmetic import (
    bri,
    bsi,
    cloud_mask,
    cri1,
    evi,
    evi2,
    moisture,
    mndwi,
    ndre,
    ndsi,
    ndvi,
    ndwi,
    ndyi,
    osavi,
    ri,
    true_color
)
from landcoverpy.minio import MinioConnection
from landcoverpy.mongo import MongoConnection

from landcoverpy.config import settings

indexes_bands = dict(
    moisture={"b8a": "B8A_20m", "b11": "B11_20m"},
    ndvi={"b4": "B04_10m", "b8": "B08_10m"},
    ndwi={"b3": "B03_10m", "b8": "B08_10m"},
    ndsi={"b3": "B03_20m", "b11": "B11_20m"},
    evi={"b2": "B02_10m", "b4": "B04_10m", "b8": "B08_10m"},
    cloudmask={"scl": "SCL_20m"},
    osavi={"b4": "B04_10m", "b8": "B08_10m"},
    evi2={"b4": "B04_10m", "b8": "B08_10m"},
    ndre={"b5": "B05_60m", "b9": "B09_60m"},
    ndyi={"b2": "B02_10m", "b3": "B03_10m"},
    mndwi={"b3": "B03_20m", "b11": "B11_20m"},
    bri={"b3": "B03_10m", "b5": "B05_20m", "b8": "B08_10m"},
    bsi={"b2": "B02_10m", "b4": "B04_10m", "b8": "B08_10m", "b11": "B11_20m"},
    tci={"b2": "B02_10m", "b3": "B03_10m", "b4": "B04_10m"},
    ri={"b3": "B03_10m", "b4": "B04_10m"},
    cri1={"b2": "B02_10m", "b3": "B03_10m"},
)


def find_product_image(band_name: str, product_title: str) -> Path:
    """
    Finds image matching a pattern in the product folder with glob.
    :param pattern: A pattern to match.
    :return: A Path object pointing to the first found image.
    """
    product_folder = join(settings.TMP_DIR, product_title)
    return ([f for f in Path(product_folder).glob("*" + band_name + "*")])[0]

def get_index(index_name, bands_dict, product_title, minio_folder_name):

    band_extension = ".tif"

    indexes_folder = join(settings.TMP_DIR, product_title, minio_folder_name)
    output = Path(indexes_folder + "/" + index_name + ".tif")
    try:
        if index_name == "moisture":
            index_value = moisture(
                b8a=find_product_image(bands_dict["b8a"], product_title),
                b11=find_product_image(bands_dict["b11"], product_title),
                output=output,
            )
        elif index_name == "ndvi":
            index_value = ndvi(
                b4=find_product_image(bands_dict["b4"], product_title),
                b8=find_product_image(bands_dict["b8"], product_title),
                output=output,
            )
        elif index_name == "ndwi":
            index_value = ndwi(
                b3=find_product_image(bands_dict["b3"], product_title),
                b8=find_product_image(bands_dict["b8"], product_title),
                output=output,
            )
        elif index_name == "ndsi":
            index_value = ndsi(
                b3=find_product_image(bands_dict["b3"], product_title),
                b11=find_product_image(bands_dict["b11"], product_title),
                output=output,
            )
        elif index_name == "evi":
            index_value = evi(
                b2=find_product_image(bands_dict["b2"], product_title),
                b4=find_product_image(bands_dict["b4"], product_title),
                b8=find_product_image(bands_dict["b8"], product_title),
                output=output,
            )
        elif index_name == "cloudmask":
            index_value = cloud_mask(
                scl=find_product_image(bands_dict["scl"], product_title),
                output=output,
            )
        elif index_name == "osavi":
            index_value = osavi(
                b4=find_product_image(bands_dict["b4"], product_title),
                b8=find_product_image(bands_dict["b8"], product_title),
                Y=0.16,
                output=output,
            )
        elif index_name == "evi2":
            index_value = evi2(
                b4=find_product_image(bands_dict["b4"], product_title),
                b8=find_product_image(bands_dict["b8"], product_title),
                output=output,
            )
        elif index_name == "ndre":
            index_value = ndre(
                b5=find_product_image(bands_dict["b5"], product_title),
                b9=find_product_image(bands_dict["b9"], product_title),
                output=output,
            )
        elif index_name == "ndyi":
            index_value = ndyi(
                b2=find_product_image(bands_dict["b2"], product_title),
                b3=find_product_image(bands_dict["b3"], product_title),
                output=output,
            )
        elif index_name == "mndwi":
            index_value = mndwi(
                b3=find_product_image(bands_dict["b3"], product_title),
                b11=find_product_image(bands_dict["b11"], product_title),
                output=output,
            )
        elif index_name == "bri":
            index_value = bri(
                b3=find_product_image(bands_dict["b3"], product_title),
                b5=find_product_image(bands_dict["b5"], product_title),
                b8=find_product_image(bands_dict["b8"], product_title),
                output=output,
            )
        elif index_name == "bsi":
            index_value = bsi(
                b2=find_product_image(bands_dict["b2"], product_title),
                b4=find_product_image(bands_dict["b4"], product_title),
                b8=find_product_image(bands_dict["b8"], product_title),
                b11=find_product_image(bands_dict["b11"], product_title),
                output=output,
            )
        elif index_name == "ri":
            index_value = ri(
                b3=find_product_image(bands_dict["b3"], product_title),
                b4=find_product_image(bands_dict["b4"], product_title),
                output=output,
            )
        elif index_name == "cri1":
            index_value = cri1(
                b2=find_product_image(bands_dict["b2"], product_title),
                b3=find_product_image(bands_dict["b3"], product_title),
                output=output,
            )
        elif index_name == "tci":
            index_value = true_color(
                b=find_product_image(bands_dict["b2"], product_title),
                g=find_product_image(bands_dict["b3"], product_title),
                r=find_product_image(bands_dict["b4"], product_title),
                output=output,
            )

        index_value = (
            np.nanmean(index_value)
            if index_value is not None and not isinstance(index_value, float)
            else index_value
        )
    except Exception as e:
        raise e

    minio_client = MinioConnection()
    minio_bucket_name = settings.MINIO_BUCKET_NAME_COMPOSITES

    date = datetime.strptime(product_title.split('_')[2], "%Y%m%d")
    year = date.strftime("%Y")
    month = date.strftime("%B")
    tile_id = product_title.split("_T")[1][0:5]
    minio_dir = join(tile_id, year, month, "")
    minio_dir = join(minio_dir, "composites", "")

    tif_minio_path = join(minio_dir, product_title, minio_folder_name ,index_name + ".tif")


    minio_client.fput_object(
        minio_bucket_name,
        tif_minio_path,
        indexes_folder + "/" + index_name + ".tif",
        content_type="image/tif",
    )

    band = dict()
    for k, v in bands_dict.items():
        band[k] = {}
        band[k]["rasterS3Bucket"] = minio_bucket_name
        band[k]["rasterS3Key"] = join(minio_dir, product_title, "raw" , v + band_extension)

    index_dict = {
        "name": index_name,
        "rasterS3Bucket": minio_bucket_name,
        "rasterS3Key": tif_minio_path,
        "bands": band,
        "rasterMeanValue": float(index_value) if index_value is not None else index_value
    }

    return index_dict


def calculate_raw_index(
    product_title: str,
    index: list,
    minio_folder_name: str = "indexes"
):
    """
    Example: python raw_index_calculation.py --uid dad7f379-de8c-49ec-b4cf-44348d0f418c --index ndvi --index ndsi --temp-dir ./data
    """
    
    # Connect with mongo
    mongo_col = MongoConnection().get_composite_collection_object()
    band_extension = ".tif"
    date = datetime.strptime(product_title.split('_')[2], "%Y%m%d")        

    # Search product metadata in Mongo
    product_data = mongo_col.find_one({"title": product_title})

    temp_dir = settings.TMP_DIR
    product_local_folder = join(temp_dir, product_title)

    # Create folder to store tif files
    indexes_folder = join(product_local_folder, minio_folder_name)
    Path(indexes_folder).mkdir(exist_ok=True, parents=True)

    # Determine the Minio folder
    year = date.strftime("%Y")
    month = date.strftime("%B")
    tile_id = product_data["title"].split("_T")[1][0:5]
    minio_dir = join(tile_id, year, month, "")
    minio_dir = join(minio_dir, "composites", "")
    bands_dir = join(minio_dir, product_title, "raw", "")

    # Create dictionary of indexes
    index_dicts = {}

    if minio_folder_name not in product_data:
        product_data[minio_folder_name] = []

    minio_client = MinioConnection()
    minio_bucket_name = settings.MINIO_BUCKET_NAME_COMPOSITES

    for idx in index:
        index_name = idx.lower()

        if index_name in product_data[minio_folder_name]:
            print("The index " + index_name + " is already calculated")
            continue

        for v in indexes_bands[index_name].values():

            band_file = v + band_extension
            local_band_path = join(product_local_folder, band_file)

            if not os.path.exists(local_band_path):
                minio_client.fget_object(
                    minio_bucket_name,
                    join(bands_dir, band_file),
                    local_band_path
                )

        dict_key = index_name.replace("-", "")
        index_dicts[index_name] = get_index(
            index_name=index_name,
            bands_dict=indexes_bands[dict_key],
            product_title=product_title,
            minio_folder_name=minio_folder_name,
        )


    # merge existing indexes in the product_data with the new ones
    index_dicts.update(product_data[minio_folder_name])

    # Push the index_dicts to MongoDB
    mongo_col.update_one(
        {"title": product_title},
        {"$set": {minio_folder_name: index_dicts}}
    )

    # Remove product from local folder
    try:
        shutil.rmtree(product_local_folder)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
