from distributed import Client

from etc_workflow.config import settings
from etc_workflow.data_postprocessing import postprocess_dataset
from etc_workflow.execution_mode import ExecutionMode
from etc_workflow.model_training import train_model_land_cover, train_model_forest
from etc_workflow.utilities.plot_spectral_signature import compute_spectral_signature_plot
from etc_workflow.workflow import workflow
from etc_workflow.utilities.aoi_tiles import get_list_of_tiles_in_mediterranean_basin


def run_compute_training_dataset_distributed():
    """Creates a training dataset in distributed mode"""

    client = Client(address=settings.DASK_CLUSTER_IP)
    workflow(execution_mode=ExecutionMode.TRAINING, client=client)


def run_compute_training_dataset():
    """Creates a training dataset in local mode"""

    workflow(execution_mode=ExecutionMode.TRAINING)


def run_model_training():
    """Postprocess the training dataset, trains a Random Forest using it and computes the spectral plot of classes `water`, `wetland`, `closedForest` and `shrubland`"""

    input_dataset = "dataset.csv"
    land_cover_dataset = "dataset_postprocessed.csv"
    forest_dataset = "dataset_forests.csv"

    postprocess_dataset(input_dataset, land_cover_dataset, True, forest_dataset)
    train_model_land_cover(land_cover_dataset, n_jobs = 1)
    train_model_forest(forest_dataset, use_open_forest = True, n_jobs = 1)
    train_model_forest(forest_dataset, use_open_forest = False, n_jobs = 1)

    compute_spectral_signature_plot(
        land_cover_dataset,
        out_plot_path="../plot.png",
        out_legend_path="../legend.png",
        classes_showed=["water", "wetland", "closedForest", "shrubland"],
    )


def run_predict_tiles_distributed():
    """Predicts all tiles appearing in the training dataset in distributed mode"""

    client = Client(address=settings.DASK_CLUSTER_IP)
    workflow(execution_mode=ExecutionMode.FOREST_PREDICTION, client=client, tiles_to_predict=get_list_of_tiles_in_mediterranean_basin())


def run_predict_tiles():
    """Predicts all tiles appearing in the training dataset in local mode"""

    workflow(execution_mode=ExecutionMode.FOREST_PREDICTION, tiles_to_predict=get_list_of_tiles_in_mediterranean_basin())
