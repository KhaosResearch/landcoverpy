from os.path import join

import pandas as pd

from etc_workflow.config import settings
from etc_workflow.utils import _get_minio, _safe_minio_execute


def postprocess_dataset(input_dataset: str, output_dataset: str, forest_classification: bool = False, output_dataset_forest: str = None):
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
    minio_client = _get_minio()

    input_file_path = join(settings.TMP_DIR, input_dataset)

    _safe_minio_execute(
        func=minio_client.fget_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, input_dataset),
        file_path=input_file_path,
    )

    df = pd.read_csv(input_file_path, dtype={'forest_type': "string"})
    df_land_cover = postprocess_dataset_land_cover(df.copy())

    output_file_path = join(settings.TMP_DIR, output_dataset)

    df_land_cover.to_csv(output_file_path, index=False)

    _safe_minio_execute(
        func=minio_client.fput_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, output_dataset),
        file_path=output_file_path,
        content_type="text/csv",
    )

    if forest_classification:

        df_forest = postprocess_dataset_forest(df)

        output_file_path = join(settings.TMP_DIR, output_dataset_forest)

        df_forest.to_csv(output_file_path, index=False)

        _safe_minio_execute(
            func=minio_client.fput_object,
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

    print("\nHistogram of land cover classes in input dataset\n", df["class"].value_counts())

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
    print("\nHistogram of land cover classes in output dataset\n", df["class"].value_counts()) 

    df.drop(columns=["forest_type"], inplace = True)

    return df

def postprocess_dataset_forest(df: pd.DataFrame):
    """
    Postprocess the dataset for forest classification:
        - Filter by land cover classes `mixedForest`, `treePlantation`, `riparianForest`, `closedForest` and `openForest`
        - Remove pixels without `forest_type` value and "No arbolado" values
        - If land cover class is openForest:
            + ["Sabinares de Juniperus phoenicea", "Sabinares de Juniperus phoenicea ssp. Turbinata"] -> `Sabinares de Juniperus`
            + Remove "Pinares de pino pinaster en región atlántica", Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Atlántica
                and "Bosques mixtos de frondosas autóctonas en region biogeográfica atlántica" 
            + Remove classes that have less than 6*9 pixels
        - If land cover class is treePlantation:
            + Remove classes that have less than 25*9 pixels
            + Remove "Pinares de pino pinaster en región atlántica"
            + Forest classification label has to be converted to ["plantacion" + `forest_type`]
        - Else (dense not plantation):
            + Remove "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Atlántica"
            + ["Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Alpina", "Mezcla de coníferas con frondosas, autoctónas con alóctonas" y "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Mediterránea"] -> `Mezcla de coníferas y frondosas`
        - Shuffle the whole dataset
    """

    df_all_forests = df[df["class"].isin(["mixedForest", "treePlantation", "riparianForest", "closedForest", "openForest"])]
    df_all_forests = df_all_forests[~((df_all_forests["forest_type"].isna()) | (df_all_forests["forest_type"] == "No arbolado"))]

    # Open forest part
    df_open_forest = df_all_forests[df_all_forests["class"] == "openForest"].copy()

    print("\nHistogram of classes in open forest input dataset\n", df_open_forest["forest_type"].value_counts())

    df_open_forest["forest_type"] = df_open_forest["forest_type"].replace(
        to_replace=["Sabinares de Juniperus phoenicea", "Sabinares de Juniperus phoenicea ssp. Turbinata"], 
        value="Sabinares de Juniperus"
    )

    classes_to_remove = [
            "Pinares de pino pinaster en región atlántica",
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Atlántica",
            "Bosques mixtos de frondosas autóctonas en region biogeográfica atlántica"
    ]

    classes_without_enough_data = list(df_open_forest["forest_type"].value_counts().loc[lambda x : x<6*9].index.values)
    classes_to_remove = classes_to_remove + classes_without_enough_data
    df_open_forest = df_open_forest[~df_open_forest["forest_type"].isin(classes_to_remove)]

    print("\nHistogram of classes in open forest output dataset\n", df_open_forest["forest_type"].value_counts())

    # Tree plantation part
    df_tree_plantation = df_all_forests[df_all_forests["class"] == "treePlantation"].copy()

    print("\nHistogram of classes in tree plantation input dataset\n", df_tree_plantation["forest_type"].value_counts())

    classes_to_remove = ["Pinares de pino pinaster en región atlántica"]
    classes_without_enough_data = list(df_tree_plantation["forest_type"].value_counts().loc[lambda x : x<25*9].index.values)
    classes_to_remove = classes_to_remove + classes_without_enough_data
    df_tree_plantation = df_tree_plantation[~df_tree_plantation["forest_type"].isin(classes_to_remove)]
    df_tree_plantation["forest_type"] = df_tree_plantation["forest_type"].apply(lambda x : "Plantacion - " + str(x))

    print("\nHistogram of classes in tree plantation output dataset\n", df_tree_plantation["forest_type"].value_counts())

    # Dense forest part
    df_dense_forest = df_all_forests[~df_all_forests["class"].isin(["treePlantation", "openForest"])].copy()

    print("\nHistogram of classes in dense forest input dataset\n", df_dense_forest["forest_type"].value_counts())

    df_dense_forest["forest_type"] = df_dense_forest["forest_type"].replace(
        to_replace=[
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Alpina",
            "Mezcla de coníferas con frondosas, autoctónas con alóctonas",
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Mediterránea"
        ], 
        value="Mezcla de coníferas y frondosas"
    )
    classes_to_remove = ["Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Atlántica"]
    df_dense_forest = df_dense_forest[~df_dense_forest["forest_type"].isin(classes_to_remove)]

    print("\nHistogram of classes in dense forest output dataset\n", df_dense_forest["forest_type"].value_counts())

    df = pd.concat([df_dense_forest, df_tree_plantation, df_open_forest])

    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df