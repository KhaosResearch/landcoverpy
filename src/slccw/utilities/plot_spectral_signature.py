from os.path import join
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot

from slccw.config import settings
from slccw.minio import MinioConnection


def _plot_dataset(
    dataset: pd.DataFrame, out_plot_path: str, out_legend_path: str, target_column: str
):

    fig, ax = pyplot.subplots(figsize=(16, 5))
    sns.lineplot(
        ax=ax,
        x=dataset["raster"],
        y=dataset["mean"],
        hue=dataset[target_column],
        linewidth=2,
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
            alpha=0.3,
            linewidth=0,
        )

    # Remove legend
    ax.get_legend().remove()

    # Set x-axis ticks labels
    bands = [
        "slope",
        "aspect",
        "dem",
        "spring_B01",
        "spring_B02",
        "spring_B03",
        "spring_B04",
        "spring_B05",
        "spring_B06",
        "spring_B07",
        "spring_B08",
        "spring_B8A",
        "spring_B09",
        "spring_B11",
        "spring_B12",
        "summer_B01",
        "summer_B02",
        "summer_B03",
        "summer_B04",
        "summer_B05",
        "summer_B06",
        "summer_B07",
        "summer_B08",
        "summer_B8A",
        "summer_B09",
        "summer_B11",
        "summer_B12",
        "autumn_B01",
        "autumn_B02",
        "autumn_B03",
        "autumn_B04",
        "autumn_B05",
        "autumn_B06",
        "autumn_B07",
        "autumn_B08",
        "autumn_B8A",
        "autumn_B09",
        "autumn_B11",
        "autumn_B12",
        "spring_AOT",
        "spring_WVP",
        "summer_AOT",
        "summer_WVP",
        "autumn_AOT",
        "autumn_WVP",
        "spring_evi2",
        "spring_mndwi",
        "spring_moisture",
        "spring_ndre",
        "spring_ndvi",
        "spring_ndyi",
        "spring_osavi",
        "spring_ri",
        "summer_evi2",
        "summer_mndwi",
        "summer_moisture",
        "summer_ndre",
        "summer_ndvi",
        "summer_ndyi",
        "summer_osavi",
        "summer_ri",
        "autumn_evi2",
        "autumn_mndwi",
        "autumn_moisture",
        "autumn_ndre",
        "autumn_ndvi",
        "autumn_ndyi",
        "autumn_osavi",
        "autumn_ri",
    ]

    ax.set_xticks(ax.get_xticks())  # just get ticks and reset whatever we already have
    ax.set_xticklabels(bands)  # set the new x ticks labels

    fig.get_figure().savefig(out_plot_path, bbox_inches="tight")

    # Isolate and save legend separately
    legend = fig.legend(prop=dict(size=19))
    fig.canvas.draw()
    legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
    legend_fig, legend_ax = pyplot.subplots(
        figsize=(legend_bbox.width, legend_bbox.height)
    )
    legend_squared = legend_ax.legend(
        *ax.get_legend_handles_labels(),
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=legend_fig.transFigure,
        frameon=False,
        fancybox=None,
        shadow=False,
        mode="expand",
        prop=dict(size=18)
    )
    legend_ax.axis("off")

    legend_fig.savefig(
        out_legend_path, bbox_inches="tight", bbox_extra_artists=[legend_squared],
    )


def compute_spectral_signature_plot(
    input_dataset: str,
    out_plot_path: str,
    out_legend_path: str,
    classes_showed: List[str],
    is_forest: bool = False,
):

    dataset_path = join(settings.TMP_DIR, input_dataset)

    minio_client = MinioConnection()

    minio_client.fget_object(
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
            "summer_cri1",
            "autumn_cri1",
            "spring_cri1",
        ],
        axis=1,
    )

    if is_forest:
        class_column = "forest_type"
        df = df.drop("class", axis=1)
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
    _plot_dataset(df, out_plot_path, out_legend_path, class_column)
