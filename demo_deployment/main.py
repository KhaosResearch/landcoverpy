from configure_minio_mongo import create_minio_buckets, create_mongo_collections
from download_products import download_products
from landcover_workflow import landcover_workflow

print("Creating Minio buckets and MongoDB collections...")
create_minio_buckets()
create_mongo_collections()

print("Downloading products and generating composites...")
download_products()

print("Running landcover workflow...")
landcover_workflow()



