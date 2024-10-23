import dask
from dask import delayed
from dask.distributed import Client, progress
import pdal
import json
from pathlib import Path
from landcoverpy.minio import MinioConnection
import subprocess
from tqdm import tqdm

# Setup directories globally
tmp_dir = Path("/mnt/home/am/landcoverpy/tmp")
elevation_dir = Path(tmp_dir, "elevation")
max_height_dir = Path(tmp_dir, "max_height")

# Create necessary directories
tmp_dir.mkdir(parents=True, exist_ok=True)
elevation_dir.mkdir(parents=True, exist_ok=True)
max_height_dir.mkdir(parents=True, exist_ok=True)

# Define selected features
covariance_features = [
    "Anisotropy",
    "DemantkeVerticality",
    "Eigenentropy",
    "Linearity",
    "Omnivariance",
    "Planarity",
    "Scattering",
    "EigenvalueSum",
    "SurfaceVariation",
    "Verticality"
]

normal_features = [
    "NormalX",
    "NormalY",
    "NormalZ",
    "Curvature"
]

density_features = [
    "RadialDensity"
]

# Create directories for each feature
for feature in covariance_features:
    feature_dir = Path(tmp_dir, feature)
    feature_dir.mkdir(parents=True, exist_ok=True)

def is_valid_laz_file(file_path):
    """Check if the LAZ file is valid by running pdal info."""
    try:
        result = subprocess.run(
            ["pdal", "info", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

@delayed
def process_laz_file(laz_file):
    """Process a single LAZ file to extract elevation, max height, and selected features."""
    local_laz_file = str(Path(tmp_dir, laz_file)) 
    minio_client = MinioConnection()
    minio_client.fget_object("pnoa-lidar", laz_file, local_laz_file)

    if not is_valid_laz_file(local_laz_file):
        print(f"Skipping {laz_file} as it is invalid or corrupted.")
        return

    # Prepare file paths for outputs
    out_elevation_path = str(Path(elevation_dir, f"{Path(laz_file).stem}.tif"))
    out_max_height_path = str(Path(max_height_dir, f"{Path(laz_file).stem}.tif"))

    # Base pipeline for elevation and max height
    base_pipeline = [
        {"type": "readers.las", "filename": local_laz_file},
        {"type": "filters.range", "limits": "Classification![7:7]"},
        {"type": "filters.sample", "radius": 1.0},
        {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 2.5},
    ]

    # Add covariance feature extraction filter
    covariance_features_filter = {
        "type": "filters.covariancefeatures",
        "knn": 8,
        "threads": 2,
        "feature_set": ",".join(covariance_features)  # Compute only the selected features
    }

    normal_filter = {
        "type": "filters.normal",
        "knn": 8  # Adjust the K nearest neighbors value as necessary
    }

    radial_density_filter = {
        "type": "filters.radialdensity",
        "radius": 1.0  # Adjust the radius as needed
    }

    # Writers for elevation and max height
    elevation_writer = {
        "type": "writers.gdal",
        "resolution": 10.0,
        "output_type": "max",
        "dimension": "Z",
        "filename": out_elevation_path,
        "data_type": "float"
    }

    max_height_writer = {
        "type": "filters.hag_nn"
    }, {
        "type": "filters.ferry",
        "dimensions": "HeightAboveGround=Z"
    }, {
        "type": "writers.gdal",
        "resolution": 10.0,
        "output_type": "max",
        "dimension": "HeightAboveGround",
        "filename": out_max_height_path,
        "data_type": "float"
    }

    # Build the full pipeline with covariance features
    feature_writers = []
    for feature in covariance_features + normal_features + density_features:
        feature_out_path = str(Path(tmp_dir, feature, f"{Path(laz_file).stem}.tif"))
        feature_writer = {
            "type": "writers.gdal",
            "resolution": 10.0,
            "output_type": "mean",
            "dimension": feature,
            "filename": feature_out_path,
            "data_type": "float"
        }
        feature_writers.append(feature_writer)

    # Combine all parts of the pipeline
    complete_pipeline = base_pipeline + [covariance_features_filter] + [normal_filter] + [radial_density_filter]  + feature_writers + [elevation_writer] + list(max_height_writer)

    # Execute the pipeline
    pipeline_json = json.dumps({"pipeline": complete_pipeline})
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()

    # Upload generated files to Minio
    minio_client.fput_object("pnoa-lidar", str(Path("rasters_sam", "elevation", f"{Path(laz_file).stem}.tif")), out_elevation_path)
    minio_client.fput_object("pnoa-lidar", str(Path("rasters_sam", "max_height", f"{Path(laz_file).stem}.tif")), out_max_height_path)

    # Upload feature files
    for feature in covariance_features + normal_features + density_features:
        feature_out_path = str(Path(tmp_dir, feature, f"{Path(laz_file).stem}.tif"))
        minio_client.fput_object("pnoa-lidar", str(Path("rasters_sam", feature, f"{Path(laz_file).stem}.tif")), feature_out_path)

    # Cleanup temporary files
    Path(local_laz_file).unlink()
    Path(out_elevation_path).unlink()
    Path(out_max_height_path).unlink()
    for feature in covariance_features + normal_features + density_features:
        Path(tmp_dir, feature, f"{Path(laz_file).stem}.tif").unlink()

def batch_process(tasks, batch_size):
    """Process tasks in batches to manage resource usage."""
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        progress(batch)
        dask.compute(*batch)

def main():
    client = Client(dashboard_address=':8787')

    minio_client = MinioConnection()

    # List LAZ files from Minio bucket
    laz_files = [
        obj.object_name for obj in minio_client.list_objects("pnoa-lidar", prefix="", recursive=True)
        if obj.object_name.endswith("RGB.laz")
    ]

    # Get processed files for each type
    processed_elevation_files = [
        Path(obj.object_name).stem for obj in minio_client.list_objects("pnoa-lidar", prefix="rasters_sam/elevation/", recursive=True)
    ]

    processed_max_height_files = [
        Path(obj.object_name).stem for obj in minio_client.list_objects("pnoa-lidar", prefix="rasters_sam/max_height/", recursive=True)
    ]

    # Get processed files for each selected feature
    processed_feature_files = {}
    for feature in covariance_features + normal_features + density_features:
        feature_processed_files = [
            Path(obj.object_name).stem for obj in minio_client.list_objects("pnoa-lidar", prefix=f"rasters_sam/{feature}/", recursive=True)
        ]
        processed_feature_files[feature] = set(feature_processed_files)

    # Determine fully processed files (intersection of all feature sets)
    fully_processed_files = set(processed_elevation_files) & set(processed_max_height_files)
    for feature in covariance_features + normal_features + density_features:    
        fully_processed_files &= processed_feature_files[feature]

    # Filter LAZ files to process only the ones not fully processed
    laz_files = [laz_file for laz_file in laz_files if Path(laz_file).stem not in fully_processed_files]


    print(f"Number of files to process: {len(laz_files)}")

    tasks = [process_laz_file(laz_file) for laz_file in laz_files]

    # Set batch size to limit simultaneous processing
    batch_size = 50
    batch_process(tasks, batch_size)

    client.close()

if __name__ == "__main__":
    main()
