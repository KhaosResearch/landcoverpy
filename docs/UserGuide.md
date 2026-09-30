# User Guide: Setup, Configuration & Execution

This practical guide walks you through setting up, configuring, and running **LandCoverPy** using Docker Compose.

---

## 1. Prerequisites

Before running the workflow, ensure you have:
1. **Docker Engine & Docker Compose** installed (`docker compose version` $\ge 2.0$).
2. A **Google Cloud Service Account Key** in JSON format (`gcloud-user.json`) with permissions to access public Sentinel-2 data buckets.

---

## 2. Installation & Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/KhaosResearch/landcoverpy.git
cd landcoverpy

# 2. Place your Google Cloud credential
cp /path/to/your/key.json demo_deployment/app_data/gcloud-user.json

# 3. Start the infrastructure (MinIO S3 and MongoDB)
cd demo_deployment
docker compose up -d minio mongo

# 4. Start the pipeline
docker compose up landcoverpy
```

### Accessing the Web Interfaces

- **MinIO Object Storage Console:** [http://localhost:31115](http://localhost:31115)  
  - *Default Credentials:* User: `adminadmin` | Password: `adminadmin`
  - *Buckets:* `s2-composites` (seasonal rasters), `classification-maps` (final GeoTIFFs), `models` (trained ML weights).
- **MongoDB:** Accessible on port `31114` (used internally by workers).

---

## 3. Input Files Reference (`demo_deployment/app_data/`)

To customize the pipeline for your study area or tree species, adapt the files in `app_data/`:

| File | Format | Description | Example |
| :--- | :--- | :--- | :--- |
| `gcloud-user.json` | JSON | Service account key for Google Cloud Sentinel-2 access. | Standard GCP credentials file. |
| `dataset.csv` | CSV | Field inventory / ground-truth points for training. | Columns: `latitude`, `longitude`, `category`, `subcategory`. |
| `seasons.json` | JSON | Start and end dates for each seasonal composite. | `{"spring": {"start": "2021-03-01", "end": "2021-04-30"}}` |
| `lc_labels.json` | JSON | Level-1 macro class names mapped to numeric IDs ($\ge 1$). | `{"FOR": 1, "WATER": 5, "CROPLAND": 3}` |
| `sl_labels.json` | JSON | Level-2 subcategory names mapped to numeric IDs ($\ge 2$). | `{"F-Pinus brutia": 25, "F-Quercus ilex": 30}` |

<details>
<summary><b>📂 Format Details for <code>dataset.csv</code></b></summary>

```csv
latitude,longitude,category,subcategory
37.1234,-3.5678,FOR,F-Pinus halepensis
36.8765,-4.1234,FOR,F-Quercus suber
38.0012,-2.9987,WATER,
37.4521,-3.2109,CROPLAND,
```
- `category` is mandatory for all training points (Level 1).
- `subcategory` is required for points where Level-2 classification is applied (e.g. tree species inside `FOR`).
</details>

---

## 4. Useful Execution Commands & Modes

### Processing Specific Seasons
By default, the pipeline iterates through all seasons in `seasons.json`. You can restrict execution to one or more seasons using `TARGET_SEASON`:
```bash
# Run only flowering and summer
docker run --rm --net=demo_deployment_default \
  -e TARGET_SEASON="flowering,summer" \
  -e MONGO_HOST="mongo" -e MINIO_HOST="minio" \
  -v ./app_data:/app/data \
  ghcr.io/khaosresearch/demo-landcoverpy:latest python -u download_products.py
```

### Running a Helper Worker in Reverse Order ($Z \rightarrow A$)
When running multiple machines on the same area, set `REVERSE_TILES=true` on the second machine. It will process the tiles in reverse alphabetical order, while the atomic locking mechanism in MongoDB prevents collisions when they meet in the middle:
```bash
docker run -d --name landcoverpy_helper \
  -e REVERSE_TILES=true \
  -e TARGET_SEASON="spring" \
  -v ./app_data:/app/data \
  ghcr.io/khaosresearch/demo-landcoverpy:latest python -u download_products.py
```
---

## 5. Key Environment Variables

Configure these variables in [`demo_deployment/docker-compose.yaml`](file:///home/diego/Virginia/landcoverpy/demo_deployment/docker-compose.yaml) or pass them to `docker run`:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `MAX_PRODUCTS_COMPOSITE` | `5` | Maximum cloud-free captures to blend into each seasonal composite. |
| `MIN_USEFUL_DATA_PERCENTAGE` | `30` | Minimum valid pixel percentage required to ingest a scene. |
| `GDAL_CACHEMAX` | `512` | Memory limit (MB) for GDAL raster operations. |
| `REVERSE_TILES` | `false` | When `true`, reverses tile ordering ($Z \rightarrow A$). |
| `TARGET_SEASON` | `""` | Comma-separated season filter (e.g. `"spring"` or `"summer,autumn"`). |
