from pathlib import Path

from pydantic import BaseSettings


class _Settings(BaseSettings):
    # Mongo-related settings
    MONGO_HOST: str = "0.0.0.0"
    MONGO_PORT: int = 27017
    MONGO_USERNAME: str = "user"
    MONGO_PASSWORD: str = "pass"
    MONGO_DB: str = "test"
    MONGO_PRODUCTS_COLLECTION: str = "test"

    # Minio-related settings
    MINIO_HOST: str = None
    MINIO_PORT: str = None
    MINIO_BUCKET_NAME_PRODUCTS: str = None
    MINIO_BUCKET_NAME_ASTER: str = None
    MINIO_BUCKET_NAME_DEM: str = None
    MINIO_BUCKET_MODELS: str = None
    MINIO_BUCKET_CLASSIFICATIONS: str = None
    MINIO_DATA_FOLDER_NAME: str = None
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "minio"

    # Temporal directory
    TMP_DIR: str = "/tmp"

    # Dates used for each season
    SPRING_START: str = "2001-12-31"
    SPRING_END: str = "2001-12-31"
    SUMMER_START: str = "2001-12-31"
    SUMMER_END: str = "2001-12-31"
    AUTUMN_START: str = "2001-12-31"
    AUTUMN_END: str = "2001-12-31"

    # Model folders in MinIO
    LAND_COVER_MODEL_FOLDER: str = ""

    class Config:
        env_file = ".env"
        file_path = Path(env_file)
        if not file_path.is_file():
            print("⚠️ `.env` not found in current directory")
            print("⚙️ Loading settings from environment")
        else:
            print(f"⚙️ Loading settings from dotenv @ {file_path.absolute()}")


settings = _Settings()
