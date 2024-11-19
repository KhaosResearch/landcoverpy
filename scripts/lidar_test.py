import pdal
import json
from pathlib import Path
from landcoverpy.minio import MinioConnection
from tqdm import tqdm

# Initialize Minio connection
client = MinioConnection()

# Define temporary directories for processing
tmp_dir = Path("/mnt/home/am/landcoverpy-lidar/landcoverpy/tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

# List of selected features to compute
selected_features = [
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

# Define output directory
for feature in selected_features:
    feature_dir = Path(tmp_dir, feature)
    feature_dir.mkdir(parents=True, exist_ok=True)

# List LAZ files from Minio bucket
laz_files = [
    obj.object_name for obj in client.list_objects("pnoa-lidar", prefix="", recursive=True)
    if obj.object_name.endswith("RGB.laz")
]

# remove the first 100 from list
laz_files = laz_files[100:]

for laz_file in tqdm(laz_files):
    print(f"Processing {laz_file}...")

    # Download the LAZ file locally
    local_laz_file = str(Path(tmp_dir, laz_file))
    client.fget_object("pnoa-lidar", laz_file, local_laz_file)

    # Define the base pipeline for reading and pre-processing
    pipeline_base = [
        {
            "type": "readers.las",
            "filename": local_laz_file
        },
        {
            "type": "filters.decimation",
            "step": 3
        },
        {
            "type": "filters.range",
            "limits": "Classification![7:7]"
        },
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": 8,
            "multiplier": 2.5
        },
        {
            "type": "filters.covariancefeatures",
            "knn": 8,
            "threads": 2,
            "feature_set": ",".join(selected_features)  # Compute only the selected features
        }
    ]

    # Define a single pipeline for computing and storing all features
    writers = []
    for feature in selected_features:
        out_feature_path = str(Path(tmp_dir, feature, f"{Path(laz_file).stem}.tif"))
        writers.append(
            {
                "type": "writers.gdal",
                "resolution": 10.0,
                "output_type": "mean",
                "dimension": feature,
                "filename": out_feature_path,
                "data_type": "float"
            }
        )

    # Combine the pipeline base and all writers in one single pipeline
    pipeline_json = json.dumps({"pipeline": pipeline_base + writers})

    # Execute the pipeline
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()

    print(f"All selected features saved for {laz_file}!")
    
    # Remove the local copy of the processed LAZ file
    Path(local_laz_file).unlink()
    
    print(f"Removed temporary file {local_laz_file}")