from pathlib import Path

from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    # Mongo-related settings
    MONGO_HOST: str = "0.0.0.0"
    MONGO_PORT: int = 27017
    MONGO_USERNAME: str = "user"
    MONGO_PASSWORD: str = "pass"
    MONGO_DB: str = "test"
    MONGO_PRODUCTS_COLLECTION: str = "test"
    MONGO_COMPOSITES_COLLECTION: str = "test"

    # Minio-related settings
    MINIO_HOST: str = None
    MINIO_PORT: str = None
    MINIO_BUCKET_NAME_PRODUCTS: str = None
    MINIO_BUCKET_NAME_COMPOSITES: str = None
    MINIO_BUCKET_NAME_ASTER: str = None
    MINIO_BUCKET_NAME_DEM: str = None
    MINIO_BUCKET_DATASETS: str = None
    MINIO_BUCKET_MODELS: str = None
    MINIO_BUCKET_CLASSIFICATIONS: str = None
    MINIO_BUCKET_TILE_METADATA: str = None
    MINIO_BUCKET_GEOJSONS: str = None
    MINIO_DATA_FOLDER_NAME: str = None
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "minio"

    # Sentinel download API
    SENTINEL_HOST: str = "https://scihub.copernicus.eu/dhus"
    SENTINEL_USERNAME: str = ""
    SENTINEL_PASSWORD: str = ""

    # Temporal directory
    TMP_DIR: str = "/tmp"

    # File containing validated data (.kmz or .geojson)
    DB_FILE: str = "/data.kmz"
    # Column (in case of db_file is a csv) or attribute (in case of db_file is a kmz or geojson) containing the LC class labels
    LC_PROPERTY: str = "LC"
    # Same for second level class labels
    SL_PROPERTY: str = "SL"

    # JSON files containing LC class labels to numbers mapping (0 is reserved to nodata)
    LC_LABELS_FILE: str = "/lc_labels.json"
    # Same for second level class labels (0 is reserved to nodata and 1 is reserved to noclassified)
    SL_LABELS_FILE: str = "/sl_labels.json"

    # For running in a distributed environment
    DASK_CLUSTER_IP: str = "0.0.0.0:0000"

    # File containing the definition of the seasons
    SEASONS_FILE: str = "/seasons.json"

    # Product filtering parameters
    MIN_USEFUL_DATA_PERCENTAGE: float = 0.0

    # Maximum number of products used in a composite
    MAX_PRODUCTS_COMPOSITE: int = 1

    class Config:
        env_file = ".env"
        file_path = Path(env_file)
        if not file_path.is_file():
            print("⚠️ `.env` not found in current directory")
            print("⚙️ Loading settings from environment")
        else:
            print(f"⚙️ Loading settings from dotenv @ {file_path.absolute()}")


settings = _Settings()
