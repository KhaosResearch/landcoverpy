FROM python:3.10-slim

LABEL org.opencontainers.image.source=https://github.com/KhaosResearch/landcoverpy/
LABEL org.opencontainers.image.description="Landcoverpy demo deployment"
LABEL org.opencontainers.image.licenses=MIT

RUN apt-get update && apt-get install -y libexpat1

WORKDIR /app

COPY requirements_ds_download.txt .
COPY requirements_landcoverpy.txt .

# Take into account that ds-download is a private package. You are not expected to build this image.
# However, the image built using this Dockerfile is public in the LandCoverPy repository. 
# Can be pulled using `docker pull ghcr.io/khaosresearch/demo-landcoverpy:latest`.
# If you want to build this image, you would need to provide your own repository for downloading sentinel-2 images.
# In the future, we will make the ds-download package public.
ENV EXTRA_PIP_PACKAGES="--index-url=http://<private-repository>/simple --trusted-host <repository-ip> ds-download==0.2.4 landcoverpy"

RUN pip install --no-cache-dir -r requirements_ds_download.txt && \
    pip install --no-cache-dir -r requirements_landcoverpy.txt && \
    pip install $EXTRA_PIP_PACKAGES

COPY configure_minio_mongo.py .
COPY download_products.py .
COPY landcover_workflow.py .
COPY main.py .

# Fixed venv for LandCoverPy, for the docker-compose to work
ENV MINIO_PORT="9000"
ENV MINIO_BUCKET_NAME_PRODUCTS=s2-products
ENV MINIO_BUCKET_NAME_COMPOSITES=s2-composites
ENV MINIO_BUCKET_NAME_ASTER=aster-slope-aspect
ENV MINIO_BUCKET_NAME_DEM=aster-dem
ENV MINIO_BUCKET_DATASETS=datasets
ENV MINIO_BUCKET_MODELS=ml-models
ENV MINIO_BUCKET_GEOJSONS=geojsons
ENV MINIO_BUCKET_TILE_METADATA=tile-metadata
ENV MINIO_BUCKET_CLASSIFICATIONS=classification-maps
ENV MONGO_DB=sentinel2-metadata
ENV MONGO_PRODUCTS_COLLECTION=products
ENV MONGO_COMPOSITES_COLLECTION=composites
ENV MONGO_PORT="27017"
ENV TMP_DIR=/app/tmp/
ENV DB_FILE=/app/data/dataset.csv
ENV LC_PROPERTY="category"
ENV SL_PROPERTY="subcategory"
ENV LC_LABELS_FILE=/app/data/lc_labels.json
ENV SL_LABELS_FILE=/app/data/sl_labels.json
ENV SEASONS_FILE=/app/data/seasons.json

#Fixed venv for ds-downlad, for the docker-compose to work
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/data/gcloud-user.json"
ENV GOOGLE_CLOUD_BUCKET_NAME="gcp-public-data-sentinel-2"
ENV MINIO_BUCKET_NAME=s2-products
ENV MONGO_DATABASE_NAME=sentinel2-metadata
ENV MONGO_COLLECTION_NAME=products
ENV MONGO_COMPOSITE_COLLECTION_NAME=composites
ENV TMP_DIR=/app/tmp/
