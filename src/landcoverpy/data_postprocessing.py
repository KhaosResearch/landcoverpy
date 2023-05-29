from os.path import join

import pandas as pd

from landcoverpy.config import settings
from landcoverpy.minio import MinioConnection


def postprocess_dataset(input_dataset: str, output_land_cover_dataset: str, forest_classification: bool = False, output_dataset_forest: str = None):
    """
    Once the training dataset is computed, it is possible that it needs some preprocessing.
    This script is in charge of all kind of preprocessing needed. 
    
    The steps that this method does are:
        - Read input dataset `input_dataset` from minio
        - Postprocess the dataset for land cover
        - Write resulting dataset to Minio as `output_dataset`
        - If `forest_classification` is True:
            + Postprocess the dataset for forest classification
            + Write resulting dataset to Minio as `output_dataset_forest`
    """
    minio_client = MinioConnection()

    input_file_path = join(settings.TMP_DIR, input_dataset)

    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, input_dataset),
        file_path=input_file_path,
    )

    df = pd.read_csv(input_file_path, dtype={'forest_type': "string"})
    df_land_cover = postprocess_dataset_land_cover(df.copy())

    output_file_path = join(settings.TMP_DIR, output_land_cover_dataset)

    df_land_cover.to_csv(output_file_path, index=False)

    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, output_land_cover_dataset),
        file_path=output_file_path,
        content_type="text/csv",
    )

    if forest_classification:

        df_forest = postprocess_dataset_forest(df)

        output_file_path = join(settings.TMP_DIR, output_dataset_forest)

        df_forest.to_csv(output_file_path, index=False)

        minio_client.fput_object(
            bucket_name=settings.MINIO_BUCKET_DATASETS,
            object_name=join(settings.MINIO_DATA_FOLDER_NAME, output_dataset_forest),
            file_path=output_file_path,
            content_type="text/csv",
        )

        

def postprocess_dataset_land_cover(df: pd.DataFrame):
    """
    Postprocess the dataset for land cover:
        - Map classes `mixedForest`, `treePlantation` and `riparianForest` to `closedForest`
        - Map class `grass` to `dehesa`
        - Map class `tropical` to `cropland`
        - Reduce the quantity of rows labeled as `closedForest` in a 1:9 ratio.
        - Shuffle the whole dataset
        - Drop `forest_type` column
    """

    print("\nHistogram of land cover classes in input dataset\n")
    print(*sorted(df["class"].value_counts().to_dict().items()), sep="\n")

    # Sustituir clases
    df["class"].replace(
        to_replace=["mixedForest", "treePlantation", "riparianForest"],
        value="closedForest",
        inplace=True,
    )
    df["class"].replace(to_replace=["grass", "dehesa"], value="herbaceousVegetation", inplace=True)

    df["class"].replace(to_replace=["rocks", "beaches"], 
    value="bareSoil", inplace=True)

    df["class"].replace(to_replace=["tropical"], 
    value="cropland", inplace=True)

    # Reduce size closedForest
    df_bosque = df[df["class"] == "closedForest"].copy()
    df_not_bosque = df[df["class"] != "closedForest"].copy()
    df_bosque = df_bosque.drop_duplicates(subset=["longitude", "latitude"]).reset_index(
        drop=True
    )
    df = pd.concat([df_bosque, df_not_bosque])

    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    print("\nHistogram of land cover classes in output dataset\n")
    print(*sorted(df["class"].value_counts().to_dict().items()), sep="\n")

    df.drop(columns=["forest_type"], inplace = True)

    return df

def postprocess_dataset_forest(df: pd.DataFrame):
    """
    Postprocess the dataset for forest classification:
        - Filter by land cover classes `mixedForest`, `treePlantation`, `riparianForest`, `closedForest` and `openForest`
        - Remove pixels without `forest_type` value and "No arbolado" values
        - Shuffle the whole dataset
    """

    df_all_forests = df[df["class"].isin(["mixedForest", "treePlantation", "riparianForest", "closedForest", "openForest"])]
    df_all_forests = df_all_forests[~((df_all_forests["forest_type"].isna()) | (df_all_forests["forest_type"] == "No arbolado"))]
    
    # Open forest part
    df_open_forest = df_all_forests[df_all_forests["class"] == "openForest"].copy()

    print("\nHistogram of classes in open forest input dataset\n")
    print(*sorted(df_open_forest["forest_type"].value_counts().to_dict().items()), sep="\n")

    df_open_forest_encinares = df_open_forest[df_open_forest["forest_type"] == 'Encinares (Quercus ilex)'].copy()
    df_open_forest_not_encinared = df_open_forest[df_open_forest["forest_type"] != 'Encinares (Quercus ilex)'].copy()
    df_open_forest_encinares = df_open_forest_encinares.drop_duplicates(subset=["longitude", "latitude"]).reset_index(
        drop=True
    )
    df_open_forest = pd.concat([df_open_forest_not_encinared, df_open_forest_encinares])

    print("\nHistogram of classes in open forest output dataset\n")
    print(*sorted(df_open_forest["forest_type"].value_counts().to_dict().items()), sep=",\n")

    # Dense forest part
    df_dense_forest = df_all_forests[~df_all_forests["class"].isin(["openForest"])].copy()

    print("\nHistogram of classes in dense forest input dataset\n")
    print(*sorted(df_dense_forest["forest_type"].value_counts().to_dict().items()), sep="\n")

    df = pd.concat([df_dense_forest, df_open_forest])

    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df