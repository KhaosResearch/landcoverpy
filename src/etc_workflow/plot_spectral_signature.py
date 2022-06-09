from os.path import join
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot

from etc_workflow.config import settings
from etc_workflow.utils import _get_minio, _safe_minio_execute


def _plot_dataset(dataset: pd.DataFrame, out_plot_path: str, target_column: str):



    fig, ax = pyplot.subplots(figsize=(35, 15))
    ax.set(ylim=(-2, 2))
    sns.lineplot(
        ax=ax, x=dataset["raster"], y=dataset["mean"], hue=dataset[target_column], linewidth=3
    )
    ax.tick_params(labelrotation=90)

    df_classes = [
        group.reset_index()[[target_column, "raster", "mean", "std"]]
        for _, group in dataset.groupby(target_column)
    ]
    for df in df_classes:
        ax.fill_between(
            x=df["raster"],
            y1=df["mean"] + df["std"],
            y2=df["mean"] - df["std"],
            alpha=0.5,
            linewidth=0,
        )
    fig.get_figure().savefig(out_plot_path)


def compute_spectral_signature_plot(
    input_dataset: str,
    out_plot_path: str,
    classes_showed: List[str],
    is_forest: bool = False,
):

    dataset_path = join(settings.TMP_DIR, input_dataset)

    minio_client = _get_minio()

    _safe_minio_execute(
        func=minio_client.fget_object,
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, input_dataset),
        file_path=dataset_path,
    )

    df = pd.read_csv(dataset_path)
    df = df.drop(
        [
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
        ],
        axis=1,
    )

    
    if is_forest:
        class_column = "forest_type"
        df = df.drop("class",axis=1)
    else:
        class_column = "class"

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    means = df.groupby(class_column).mean()
    stds = df.groupby(class_column).std()
    means = means.reset_index()
    stds = stds.reset_index()
    stds = pd.melt(stds, id_vars=class_column, value_name="std", var_name="raster")
    means = pd.melt(means, id_vars=class_column, value_name="mean", var_name="raster")
    stds = stds["std"]
    df = means.join(stds)

    df = df[df[class_column].isin(classes_showed)]
    _plot_dataset(df, out_plot_path, class_column)
