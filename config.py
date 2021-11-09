from pathlib import Path

from pydantic import AnyUrl, BaseModel, BaseSettings, SecretStr

class _Settings(BaseSettings):
    # mongo settings
    MONGO_HOST: str = "0.0.0.0"
    MONGO_PORT: int = 27017
    MONGO_USERNAME: str = "user"
    MONGO_PASSWORD: str = "pass"
    MONGO_DB: str = "test"
    MONGO_COLLECTION: str = "test"

    # DFS
    MINIO_HOST: str = None
    MINIO_PORT: int = 9000
    MINIO_BUCKET_NAME: str = None
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "minio"

    # Sentinel download API
    SENTINEL_HOST: str = "https://scihub.copernicus.eu/dhus"
    SENTINEL_USERNAME: str = ''
    SENTINEL_PASSWORD: str = ''

    TMP_DIR: str = "/tmp"

    class Config:
        env_file = ".env"
        file_path = Path(env_file)
        if not file_path.is_file():
            print("⚠️ `.env` not found in current directory")
            print("⚙️ Loading settings from environment")
        else:
            print(f"⚙️ Loading settings from dotenv @ {file_path.absolute()}")


settings = _Settings()