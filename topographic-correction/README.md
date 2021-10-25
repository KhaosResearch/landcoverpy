# Python script for performing topographic/terrain correction for Sentinel 2A products
This scripts uses the SAGA library for topographic correction. This operation requires a Digital Elevation Model (DEM), which must be reprojected and resampled to the product's SRC, and also the product's azimuth and height.

Install the python libraries.
```
pip install -r requirements.txt
```

For Ubuntu installations, the SAGA library can be found at the APT repositories.
```
sudo apt update
sudo apt install saga
```

Currently paths are located at *topographic-correction/tc.py*.
Please modify them and run *tc.py* for computing the topographic correction.

## TODO
- Integrate data source (MinIO).
- Include Typer.
- Develop test cases.