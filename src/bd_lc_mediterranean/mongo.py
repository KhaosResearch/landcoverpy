from pymongo import MongoClient
from pymongo.collection import Collection

from bd_lc_mediterranean.config import settings


class MongoConnection():
    "Simple MongoBD class including some useful methods"
    def __init__(self, host=settings.MONGO_HOST, port=settings.MONGO_PORT, username=settings.MONGO_USERNAME, password=settings.MONGO_PASSWORD, database=settings.MONGO_DB, collection=settings.MONGO_PRODUCTS_COLLECTION):
        self.mongo_client = MongoClient(
                host = f"mongodb://{host}:{port}/",
                username = username,
                password = password
            )
        self.db = database
        self.col = collection

    def set_database(self, database: str):
        self.db = database

    def set_collection(self, collection: str):
        self.col = collection

    def get_collection_object(self) -> Collection:
        return self.mongo_client[self.db][self.col]