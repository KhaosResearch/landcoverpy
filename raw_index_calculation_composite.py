import os
import shutil
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from greensenti.cli.compute_index import *
from minio import Minio
from pymongo import MongoClient
from config import settings
from utils import safe_minio_execute

load_dotenv()

indexes_bands = dict(
    moisture={"b8a": "B8A_20m", "b11": "B11_20m"},
    ndvi={"b4": "B04_10m", "b8": "B08_10m"},
    ndwi={"b3": "B03_10m", "b8": "B08_10m"},
    ndsi={"b3": "B03_20m", "b11": "B11_20m"},
    evi={"b2": "B02_10m", "b4": "B04_10m", "b8": "B08_10m"},
    coverpercentage={"b3": "B03_20m", "b4": "B04_20m", "b11": "B11_20m"},
    osavi={"b4": "B04_10m", "b8": "B08_10m"},
    evi2={"b4": "B04_10m", "b8": "B08_10m"},
    ndre={"b5": "B05_60m", "b9": "B09_60m"},
    ndbg={"b2": "B02_10m", "b3": "B03_10m"},
    mndwi={"b3": "B03_20m", "b11": "B11_20m"},
    bri={"b3": "B03_10m", "b5": "B05_20m", "b8": "B08_10m"},
)


def calculate_raw_indexes(uid: str):

    index = ["Moisture", "NDVI", "NDWI", "NDSI", "EVI", "Cover-Percentage", "OSAVI", "EVI2", "NDRE", "NDBG", "MNDWI", "BRI"]

    minio_bucket_name = settings.MINIO_BUCKET_NAME_COMPOSITES

    # Connect with mongo
    mongo_client = MongoClient(
        "mongodb://" + str(settings.MONGO_HOST) + ":" + str(settings.MONGO_PORT) + "/",
        username=settings.MONGO_USERNAME,
        password=settings.MONGO_PASSWORD,
    )
    mongo_db = mongo_client[settings.MONGO_DB]
    mongo_col = mongo_db[settings.MONGO_COMPOSITES_COLLECTION]

    # Search product metadata in Mongo
    product_data = mongo_col.find_one({"id": uid})

    # Find if the file is already unzipped in the temporary folder
    temp_dir = str(settings.TMP_DIR)
    title = product_data["title"]
    unzip_folder = temp_dir + '/' + title + ".SAFE"
    exists_unzip = Path(unzip_folder).is_dir()

    # Declare function for image search
    def find_product_image(pattern: str) -> Path:
        """
        Finds image matching a pattern in the product folder with glob.
        :param pattern: A pattern to match.
        :return: A Path object pointing to the first found image.
        """
        return ([f for f in Path(unzip_folder).glob("*" + pattern + ".tif")]
        )[0]

    # Connect with minio
    client = Minio(
        str(settings.MINIO_HOST) + ":" + str(settings.MINIO_PORT),
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False,
    )

    # Create folder to store tif files
    indexes_folder = unzip_folder + "/INDEXES"
    Path(indexes_folder).mkdir(exist_ok=True, parents=True)

    # Determine the Minio folder
    bands_dir = title + "/raw/"

    def get_index(index_name, bands_dict):
        
        output = Path(indexes_folder + "/" + index_name + ".tif")
        try:
            if index_name == "moisture":
                index_value = moisture(
                    b8a=find_product_image(bands_dict["b8a"]),
                    b11=find_product_image(bands_dict["b11"]),
                    output=output,
                )
            elif index_name == "ndvi":
                index_value = ndvi(
                    b4=find_product_image(bands_dict["b4"]),
                    b8=find_product_image(bands_dict["b8"]),
                    output=output,
                )
            elif index_name == "ndwi":
                index_value = ndwi(
                    b3=find_product_image(bands_dict["b3"]),
                    b8=find_product_image(bands_dict["b8"]),
                    output=output,
                )
            elif index_name == "ndsi":
                index_value = ndsi(
                    b3=find_product_image(bands_dict["b3"]),
                    b11=find_product_image(bands_dict["b11"]),
                    output=output,
                )
            elif index_name == "evi":
                index_value = evi(
                    b2=find_product_image(bands_dict["b2"]),
                    b4=find_product_image(bands_dict["b4"]),
                    b8=find_product_image(bands_dict["b8"]),
                    output=output,
                )
            elif index_name == "cover-percentage":
                index_value = cloud_cover_percentage(
                    b3=find_product_image(bands_dict["b3"]),
                    b4=find_product_image(bands_dict["b4"]),
                    b11=find_product_image(bands_dict["b11"]),
                    tau=0.2,
                    output=output,
                )
            elif index_name == "osavi":
                index_value = osavi(
                    b4=find_product_image(bands_dict["b4"]),
                    b8=find_product_image(bands_dict["b8"]),
                    Y=0.16,
                    output=output,
                )
            elif index_name == "evi2":
                index_value = evi2(
                    b4=find_product_image(bands_dict["b4"]),
                    b8=find_product_image(bands_dict["b8"]),
                    output=output,
                )
            elif index_name == "ndre":
                index_value = ndre(
                    b5=find_product_image(bands_dict["b5"]),
                    b9=find_product_image(bands_dict["b9"]),
                    output=output,
                )
            elif index_name == "ndbg":
                index_value = ndbg(
                    b2=find_product_image(bands_dict["b2"]),
                    b3=find_product_image(bands_dict["b3"]),
                    output=output,
                )
            elif index_name == "mndwi":
                index_value = mndwi(
                    b3=find_product_image(bands_dict["b3"]),
                    b11=find_product_image(bands_dict["b11"]),
                    output=output,
                )
            elif index_name == "bri":
                index_value = bri(
                    b3=find_product_image(bands_dict["b3"]),
                    b5=find_product_image(bands_dict["b5"]),
                    b8=find_product_image(bands_dict["b8"]),
                    output=output,
                )

        except Exception as e:
            print(f"{title} {index_name} failed: {e}")

        metadata_path = "minio://" + minio_bucket_name + "/"
        tif_minio_path = (
            title + "/indexes/" + title + "/" + index_name + ".tif"
        )
        tif_meta_minio_path = metadata_path + tif_minio_path
        band_meta_minio_path = "minio://" + minio_bucket_name + "/" + bands_dir

        safe_minio_execute(
            func = client.fput_object,
            minio_bucket_name,
            tif_minio_path,
            indexes_folder + "/" + index_name + ".tif",
            content_type="image/tif"
        )

        band = dict()
        for k, v in bands_dict.items():
            band[k] = band_meta_minio_path + v + ".tif"

        index_dict = dict(
            name=index_name,
            objectName=None,
            rawObjectName=tif_meta_minio_path,
            band=band,
            mask=None,
            value=float(index_value),
        )

        return index_dict

    # Create list of indexes
    l_indexes = []

    for idx in index:
        index_name = idx.lower()

        if index_name == "cover-percentage":
            dict_key = "coverpercentage"
        else:
            dict_key = index_name

        l_indexes.append(
            get_index(
                index_name=index_name,
                bands_dict=indexes_bands[dict_key]
            )
        )

    mongo_col.update_one({"id": uid}, {"$push": {"indexes": {"$each": l_indexes}}})
    
    # Remove product from local folder
    try:
        print(f"Removing folder {str(indexes_folder)} recursively")
        shutil.rmtree(indexes_folder)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))