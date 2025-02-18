import os
from minio import Minio
import pymongo

def create_minio_buckets():
    minio_client = Minio(
        f"{os.getenv('MINIO_HOST')}:{os.getenv('MINIO_PORT')}",
        access_key=os.getenv('MINIO_ACCESS_KEY'),
        secret_key=os.getenv('MINIO_SECRET_KEY'),
        secure=False
    )
    
    buckets = [
        "s2-products", "s2-composites", "aster-slope-aspect", "aster-dem",
        "datasets", "ml-models", "classification-maps"
    ]
    
    for bucket in buckets:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
            print(f"Created bucket: {bucket}")
        else:
            print(f"Bucket {bucket} already exists.")

def create_mongo_collections():
    client = pymongo.MongoClient(
        f"mongodb://{os.getenv('MONGO_USERNAME')}:{os.getenv('MONGO_PASSWORD')}@{os.getenv('MONGO_HOST')}:{os.getenv('MONGO_PORT')}/")
    
    db = client[os.getenv('MONGO_DB')]
    collections = ["products", "composites"]
    
    for collection in collections:
        if collection not in db.list_collection_names():
            db.create_collection(collection)
            print(f"Created collection: {collection}")
        else:
            print(f"Collection {collection} already exists.")
