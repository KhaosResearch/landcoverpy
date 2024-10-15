import pdal
import json
from pathlib import Path
from landcoverpy.minio import MinioConnection
from tqdm import tqdm
import subprocess

client = MinioConnection()

tmp_dir = Path("/mnt/home/am/landcoverpy/tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

elevation_dir = Path(tmp_dir, "elevation")
elevation_dir.mkdir(parents=True, exist_ok=True)

max_height_dir = Path(tmp_dir, "max_height")
max_height_dir.mkdir(parents=True, exist_ok=True)

laz_files = [
    obj.object_name for obj in client.list_objects("pnoa-lidar", prefix="", recursive=True)
    if obj.object_name.endswith("RGB.laz")
]

processed_elevation_files = [
    Path(obj.object_name).stem for obj in client.list_objects("pnoa-lidar", prefix="rasters/elevation/", recursive=True)
]

processed_max_height_files = [
    Path(obj.object_name).stem for obj in client.list_objects("pnoa-lidar", prefix="rasters/max_height/", recursive=True)
]

processed_files = set(processed_elevation_files) & set(processed_max_height_files)
laz_files = [laz_file for laz_file in laz_files if Path(laz_file).stem not in processed_files]

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

for laz_file in tqdm(laz_files):

    print(f"Processing {laz_file}...")
    local_laz_file = str(Path(tmp_dir, laz_file)) 
    client.fget_object("pnoa-lidar", laz_file, local_laz_file)

    # Check if the file is valid before processing it with PDAL
    if not is_valid_laz_file(local_laz_file):
        print(f"Skipping {laz_file} as it is invalid or corrupted.")
        continue

    out_elevation_path = str(Path(elevation_dir, Path(laz_file).stem + ".tif"))
    out_max_height_path = str(Path(max_height_dir, Path(laz_file).stem + ".tif"))

    pipeline_json = json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": local_laz_file
            },
            {
                "type": "filters.decimation",
                "step": 3
            },
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": 8,
                "multiplier": 2.5
            },
            {
                "type": "filters.range",
                "limits": "Classification![7:7]"
            },
            {
                "type": "writers.gdal",
                "resolution": 10.0,
                "output_type": "max",
                "dimension": "Z",
                "filename": out_elevation_path,
                "data_type": "float"
            },
            {
                "type": "filters.hag_nn"
            },
            {
                "type": "filters.ferry",
                "dimensions": "HeightAboveGround=Z"
            },
            {
                "type": "writers.gdal",
                "resolution": 10.0,
                "output_type": "max",
                "dimension": "HeightAboveGround",
                "filename": out_max_height_path,
                "data_type": "float"
            }
        ]
    })

    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    client.fput_object("pnoa-lidar", str(Path("rasters", "elevation", Path(laz_file).stem + ".tif")), out_elevation_path)
    client.fput_object("pnoa-lidar", str(Path("rasters", "max_height", Path(laz_file).stem + ".tif")), out_max_height_path)

    # Remove local files
    Path(local_laz_file).unlink()
    Path(out_elevation_path).unlink()
    Path(out_max_height_path).unlink()

    print("Done!")
