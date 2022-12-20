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
        - Change class of pixels that fulfil (`forest_type` == "Otras especies de producción en mezcla" or "Eucaliptales") to "treePlantation"
        - If land cover class is openForest:
            + ["Sabinares de Juniperus phoenicea", "Sabinares de Juniperus phoenicea ssp. Turbinata"] -> `Sabinares de Juniperus phoenicea`
            + ["Pinares de pino pinaster en región atlántica", "Pinar de pino pinaster en región mediterránea"] -> Pinar de pino pinaster
            + ["Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Alpina", "Mezcla de coníferas con frondosas, autoctónas con alóctonas", "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Mediterránea", "Arbolado disperso coníferas y frondosas", "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Atlántica", "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Macaronésica" ] -> "Mezcla de coníferas y frondosas"
            + ["Arbolado disperso  de coníferas", "Mezcla de coníferas autóctonas en la región biogeográfica Mediterránea"] -> "Otras coníferas"
            + ["Bosques mixtos de frondosas autóctonas en region biogeográfica atlántica", "Bosques mixtos de frondosas autóctonas en region biogeográfica mediterranea", "Bosque mixto de frondosas autóctonas en la región biogeográfica Alpina" "Arbolado disperso de frondosas", "Otras mezclas de frondosas autóctonas macaronésicas", "Frondosas alóctonas con  autóctonas"] -> "Otras frondosas"
            + Remove classes that have less than 5*9 pixels
        - If land cover class is treePlantation:
            + ["Pinares de pino pinaster en región atlántica", "Pinar de pino pinaster en región mediterránea"] -> Pinar de pino pinaster
            + Remove classes that have less than 5*9 pixels
            + Forest classification label has to be converted to ["plantacion" + `forest_type`]
        - Else (dense not plantation):
            + ["Pinar de pino pinaster en región mediterránea", "Pinares de pino pinaster en región atlántica"] -> "Pinar de pino pinaster"
            + ["Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Atlántica", "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Alpina", "Mezcla de coníferas con frondosas, autoctónas con alóctonas" y "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Mediterránea"] -> `Mezcla de coníferas y frondosas`
        - Shuffle the whole dataset
    """

    df_all_forests = df[df["class"].isin(["mixedForest", "treePlantation", "riparianForest", "closedForest", "openForest"])]
    df_all_forests = df_all_forests[~((df_all_forests["forest_type"].isna()) | (df_all_forests["forest_type"] == "No arbolado"))]

    df_all_forests.loc[df_all_forests["forest_type"]=="Otras especies de producción en mezcla", "class"] = "treePlantation"
    df_all_forests.loc[df_all_forests["forest_type"]=="Eucaliptales", "class"] = "treePlantation"

    # Open forest part
    df_open_forest = df_all_forests[df_all_forests["class"] == "openForest"].copy()

    print("\nHistogram of classes in open forest input dataset\n")
    print(*sorted(df_open_forest["forest_type"].value_counts().to_dict().items()), sep="\n")

    df_open_forest["forest_type"] = df_open_forest["forest_type"].replace(
        to_replace=["Sabinares de Juniperus phoenicea", "Sabinares de Juniperus phoenicea ssp. Turbinata"], 
        value="Sabinares de Juniperus phoenicea"
    )
    df_open_forest["forest_type"] = df_open_forest["forest_type"].replace(
        to_replace=["Pinares de pino pinaster en región atlántica", "Pinar de pino pinaster en región mediterránea"], 
        value="Pinares de pino pinaster"
    )
    df_open_forest["forest_type"] = df_open_forest["forest_type"].replace(
        to_replace=[
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Alpina",
            "Mezcla de coníferas con frondosas, autoctónas con alóctonas",
            "Arbolado disperso coníferas y frondosas",
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Atlántica",
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Macaronésica",
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Mediterránea"
        ], 
        value="Mezcla de coníferas y frondosas"
    )
    df_open_forest["forest_type"] = df_open_forest["forest_type"].replace(
        to_replace=["Arbolado disperso  de coníferas", "Mezcla de coníferas autóctonas en la región biogeográfica Mediterránea"], 
        value="Otras coníferas"
    )
    df_open_forest["forest_type"] = df_open_forest["forest_type"].replace(
        to_replace=[
            "Bosques mixtos de frondosas autóctonas en region biogeográfica atlántica",
            "Bosques mixtos de frondosas autóctonas en region biogeográfica mediterranea",
            "Bosque mixto de frondosas autóctonas en la región biogeográfica Alpina",
            "Arbolado disperso de frondosas",
            "Otras mezclas de frondosas autóctonas macaronésicas",
            "Frondosas alóctonas con  autóctonas"
        ], 
        value="Otras frondosas"
    )

    classes_to_remove = []
    classes_without_enough_data = list(df_open_forest["forest_type"].value_counts().loc[lambda x : x<5*9].index.values)
    classes_to_remove = classes_to_remove + classes_without_enough_data
    df_open_forest = df_open_forest[~df_open_forest["forest_type"].isin(classes_to_remove)]

    print("\nHistogram of classes in open forest output dataset\n")
    print(*sorted(df_open_forest["forest_type"].value_counts().to_dict().items()), sep="\n")

    # Tree plantation part
    df_tree_plantation = df_all_forests[df_all_forests["class"] == "treePlantation"].copy()

    print("\nHistogram of classes in tree plantation input dataset\n")
    print(*sorted(df_tree_plantation["forest_type"].value_counts().to_dict().items()), sep="\n")

    df_tree_plantation["forest_type"] = df_tree_plantation["forest_type"].replace(
        to_replace=["Pinares de pino pinaster en región atlántica", "Pinar de pino pinaster en región mediterránea"], 
        value="Pinares de pino pinaster"
    )
    classes_to_remove = []
    classes_without_enough_data = list(df_tree_plantation["forest_type"].value_counts().loc[lambda x : x<5*9].index.values)
    classes_to_remove = classes_to_remove + classes_without_enough_data
    df_tree_plantation = df_tree_plantation[~df_tree_plantation["forest_type"].isin(classes_to_remove)]
    df_tree_plantation["forest_type"] = df_tree_plantation["forest_type"].apply(lambda x : "Plantacion - " + str(x))

    print("\nHistogram of classes in tree plantation output dataset\n")
    print(*sorted(df_tree_plantation["forest_type"].value_counts().to_dict().items()), sep="\n")

    # Dense forest part
    df_dense_forest = df_all_forests[~df_all_forests["class"].isin(["treePlantation", "openForest"])].copy()

    print("\nHistogram of classes in dense forest input dataset\n")
    print(*sorted(df_dense_forest["forest_type"].value_counts().to_dict().items()), sep="\n")

    df_dense_forest["forest_type"] = df_dense_forest["forest_type"].replace(
        to_replace=["Pinares de pino pinaster en región atlántica", "Pinar de pino pinaster en región mediterránea"], 
        value="Pinares de pino pinaster"
    )

    df_dense_forest["forest_type"] = df_dense_forest["forest_type"].replace(
        to_replace=[
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Alpina",
            "Mezcla de coníferas con frondosas, autoctónas con alóctonas",
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Mediterránea",
            "Mezcla de coníferas y frondosas autóctonas en la región biogeográfica Atlántica"
        ], 
        value="Mezcla de coníferas y frondosas"
    )
    
    #classes_to_remove = []
    #df_dense_forest = df_dense_forest[~df_dense_forest["forest_type"].isin(classes_to_remove)]

    print("\nHistogram of classes in dense forest output dataset\n")
    print(*sorted(df_dense_forest["forest_type"].value_counts().to_dict().items()), sep="\n")

    df = pd.concat([df_dense_forest, df_tree_plantation, df_open_forest])

    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df