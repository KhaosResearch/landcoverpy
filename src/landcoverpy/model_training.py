import json
from os.path import join
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from landcoverpy.config import settings
from landcoverpy.minio import MinioConnection
from landcoverpy.utilities.confusion_matrix import compute_confusion_matrix


def _feature_reduction(
    df_x: pd.DataFrame, df_y: pd.DataFrame, percentage_columns: int = 100
):
    """Feature reduction method. Receives the training dataset and returns a set of variables."""

    if percentage_columns < 100:
        n_columns = len(df_x.columns.tolist())
        n_features = int(n_columns * percentage_columns / 100)
        model = LogisticRegression(
            penalty="elasticnet", max_iter=10000, solver="saga", n_jobs=-1, l1_ratio=0.5
        )
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        fit = rfe.fit(df_x, df_y)
        used_columns = df_x.columns[fit.support_].tolist()
    else:
        used_columns = df_x.columns.tolist()

    return used_columns


def train_model_land_cover(n_jobs: int = 2):
    """Trains a Random Forest model using a land cover dataset."""

    land_cover_dataset = "dataset.csv"

    training_dataset_path = join(settings.TMP_DIR, land_cover_dataset)

    minio_client = MinioConnection()

    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, land_cover_dataset),
        file_path=training_dataset_path,
    )

    df = pd.read_csv(training_dataset_path)
    df = df.drop(settings.SL_PROPERTY,axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(np.nan)
    df = df.dropna()

    print(f"Training land-cover model using {len(df)} samples with unique {df[settings.LC_PROPERTY].unique()} targets")

    y_train_data = df[settings.LC_PROPERTY]
    not_training_data_columns = [col for col in df.columns if "product_name" in col] + [settings.LC_PROPERTY, "latitude", "longitude"]
    x_train_data = df.drop(not_training_data_columns, axis=1)

    used_columns = _feature_reduction(x_train_data, y_train_data)

    train_size = 0.85

    train_coordinates_label = []
    test_coordinates_label = []

    for label in y_train_data.unique():
        df_label = df[df[settings.LC_PROPERTY] == label]
        unique_locations_label = df_label.drop_duplicates(subset=["latitude","longitude"])
        unique_locations_label = unique_locations_label[['latitude', 'longitude']]
        unique_locations_label = unique_locations_label.sample(frac=1).reset_index(drop=True)

        split_index_label = int(len(unique_locations_label) * train_size)

        train_coordinates_label.append(unique_locations_label[:split_index_label])
        test_coordinates_label.append(unique_locations_label[split_index_label:])

    train_coordinates = pd.concat(train_coordinates_label)
    test_coordinates = pd.concat(test_coordinates_label)

    train_df = pd.merge(df, train_coordinates, on=['latitude', 'longitude'])
    test_df = pd.merge(df, test_coordinates, on=['latitude', 'longitude'])

    X_train = train_df[used_columns]
    X_test = test_df[used_columns]
    y_train = train_df[settings.LC_PROPERTY]
    y_test = test_df[settings.LC_PROPERTY]

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    X_train = X_train.reindex(columns=used_columns)
    print(X_train)
    clf.fit(X_train, y_train)
    y_true = clf.predict(X_test)

    labels = y_train_data.unique()

    confusion_image_filename = "confusion_matrix.png"
    out_image_path = join(settings.TMP_DIR, confusion_image_filename)
    compute_confusion_matrix(y_true, y_test, labels, out_image_path=out_image_path)

    minio_folder = "land-cover"

    # Save confusion matrix image to minio
    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, minio_folder, confusion_image_filename),
        file_path=out_image_path,
        content_type="image/png",
    )

    model_name = "model.joblib"
    model_path = join(settings.TMP_DIR, model_name)
    joblib.dump(clf, model_path)

    # Save model to minio
    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_name}",
        file_path=model_path,
        content_type="mlmodel/randomforest",
    )

    model_metadata = {
        "model": str(type(clf)),
        "n_jobs": n_jobs,
        "used_columns": list(used_columns),
        "classes": list(labels)
    }

    model_metadata_name = "metadata.json"
    model_metadata_path = join(settings.TMP_DIR, model_metadata_name)

    with open(model_metadata_path, "w") as f:
        json.dump(model_metadata, f)

    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_metadata_name}",
        file_path=model_metadata_path,
        content_type="text/json",
    )

def train_second_level_models(lc_classes: List[str], n_jobs: int = 2, n_trees: int = 100, max_depth: int = None, min_samples_leaf: int = 1):
    """Trains a Random Forest model using a forest dataset."""

    dataset = "dataset.csv"

    training_dataset_path = join(settings.TMP_DIR, dataset)

    minio_client = MinioConnection()
    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, dataset),
        file_path=training_dataset_path,
    )

    df = pd.read_csv(training_dataset_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(np.nan)
    df = df.dropna()


    for lc_class in lc_classes:

        df_class = df[df[settings.LC_PROPERTY] == lc_class]

        print(f"Training {lc_class} model using {len(df_class)} samples with unique {df_class[settings.SL_PROPERTY].unique()} targets")

        y_train_data = df_class[settings.SL_PROPERTY]

        not_training_data_columns = [col for col in df_class.columns if "product_name" in col] + [settings.LC_PROPERTY, settings.SL_PROPERTY, "latitude", "longitude"]
        x_train_data = df_class.drop(not_training_data_columns, axis=1)

        used_columns = _feature_reduction(x_train_data, y_train_data)

        train_size = 0.80

        train_coordinates_label = []
        test_coordinates_label = []

        for class_label in y_train_data.unique():
            df_class_label = df_class[df_class[settings.SL_PROPERTY] == class_label]
            unique_locations_label = df_class_label.drop_duplicates(subset=["latitude","longitude"])
            unique_locations_label = unique_locations_label[['latitude', 'longitude']]
            unique_locations_label = unique_locations_label.sample(frac=1).reset_index(drop=True)

            split_index_label = int(len(unique_locations_label) * train_size)

            train_coordinates_label.append(unique_locations_label[:split_index_label])
            test_coordinates_label.append(unique_locations_label[split_index_label:])

        train_coordinates = pd.concat(train_coordinates_label)
        test_coordinates = pd.concat(test_coordinates_label)

        train_df = pd.merge(df_class, train_coordinates, on=['latitude', 'longitude'])
        test_df = pd.merge(df_class, test_coordinates, on=['latitude', 'longitude'])

        X_train = train_df[used_columns]
        X_test = test_df[used_columns]
        y_train = train_df[settings.SL_PROPERTY]
        y_test = test_df[settings.SL_PROPERTY]


        # Train model
        clf = RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        X_train = X_train.reindex(columns=used_columns)
        print(X_train)
        clf.fit(X_train, y_train)
        y_true = clf.predict(X_test)

        labels = y_train_data.unique()

        confusion_image_filename = "confusion_matrix.png"
        out_image_path = join(settings.TMP_DIR, confusion_image_filename)
        compute_confusion_matrix(y_true, y_test, labels, out_image_path=out_image_path)

        # Save confusion matrix image to minio
        minio_client.fput_object(
            bucket_name=settings.MINIO_BUCKET_MODELS,
            object_name=join(settings.MINIO_DATA_FOLDER_NAME, lc_class, confusion_image_filename),
            file_path=out_image_path,
            content_type="image/png",
        )

        model_name = "model.joblib"
        model_path = join(settings.TMP_DIR, model_name)
        joblib.dump(clf, model_path)
        clf = joblib.load(model_path)
        print([estimator.get_n_leaves() for estimator in clf.estimators_])
        print([estimator.get_depth() for estimator in clf.estimators_])

        # Save model to minio
        minio_client.fput_object(
            bucket_name=settings.MINIO_BUCKET_MODELS,
            object_name=join(settings.MINIO_DATA_FOLDER_NAME, lc_class, model_name),
            file_path=model_path,
            content_type="mlmodel/randomforest",
        )

        model_metadata = {
            "model": str(type(clf)),
            "n_jobs": n_jobs,
            "n_estimators": n_trees,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "used_columns": list(used_columns),
            "classes": list(labels)
        }
        model_metadata_name = "metadata.json"
        model_metadata_path = join(settings.TMP_DIR, model_metadata_name)

        with open(model_metadata_path, "w") as f:
            json.dump(model_metadata, f)

        minio_client.fput_object(
            bucket_name=settings.MINIO_BUCKET_MODELS,
            object_name=join(settings.MINIO_DATA_FOLDER_NAME, lc_class, model_metadata_name),
            file_path=model_metadata_path,
            content_type="text/json",
        )
