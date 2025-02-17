from pymongo import MongoClient
from pymongo.collection import Collection

from landcoverpy.config import settings


class MongoConnection():
    "Simple MongoBD class including some useful methods"
    def __init__(self, host=settings.MONGO_HOST, port=settings.MONGO_PORT, username=settings.MONGO_USERNAME, password=settings.MONGO_PASSWORD):
        self.mongo_client = MongoClient(
                host = f"mongodb://{host}:{port}/",
                username = username,
                password = password
            )

    def get_collection_object(self) -> Collection:
        return self.mongo_client[settings.MONGO_DB][settings.MONGO_PRODUCTS_COLLECTION]
    
    def get_composite_collection_object(self) -> Collection:
        return self.mongo_client[settings.MONGO_DB][settings.MONGO_COMPOSITES_COLLECTION]