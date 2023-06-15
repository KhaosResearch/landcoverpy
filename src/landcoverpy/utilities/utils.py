from datetime import datetime

from pymongo.collection import Collection
from pymongo.cursor import Cursor

from landcoverpy.config import settings

def get_season_dict():
    spring_start = datetime.strptime(settings.SPRING_START, '%Y-%m-%d')
    spring_end = datetime.strptime(settings.SPRING_END, '%Y-%m-%d')
    summer_start = datetime.strptime(settings.SUMMER_START, '%Y-%m-%d')
    summer_end = datetime.strptime(settings.SUMMER_END, '%Y-%m-%d')
    autumn_start = datetime.strptime(settings.AUTUMN_START, '%Y-%m-%d')
    autumn_end = datetime.strptime(settings.AUTUMN_END, '%Y-%m-%d')
    
    seasons =   {
        "spring" : (spring_start, spring_end),
        "summer" : (summer_start, summer_end),
        "autumn" : (autumn_start, autumn_end)
    }

    return seasons

def get_products_by_tile_and_date(
    tile: str,
    mongo_collection: Collection,
    start_date: datetime,
    end_date: datetime,
) -> Cursor:
    """
    Query to mongo for obtaining products filtered by tile, date and cloud percentage
    """
    product_metadata_cursor = mongo_collection.aggregate(
        [
            {
                "$project": {
                    "_id": 1,
                    "indexes": {
                        "$filter": {
                            "input": "$indexes",
                            "as": "index",
                            "cond": {
                                "$and": [
                                    {"$eq": ["$$index.mask", None]},
                                    {"$eq": ["$$index.name", "cloud-mask"]},
                                    {"$lt": ["$$index.value", 100]},
                                ]
                            },
                        }
                    },
                    "id": 1,
                    "title": 1,
                    "size": 1,
                    "date": 1,
                    "creationDate": 1,
                    "ingestionDate": 1,
                    "objectName": 1,
                }
            },
            {
                "$match": {
                    "indexes.0": {"$exists": True},
                    "title": {"$regex": f"_T{tile}_"},
                    "date": {
                        "$gte": start_date,
                        "$lte": end_date,
                    },
                }
            },
            {
                "$sort": {"date":-1}
            },
        ]
    )

    return product_metadata_cursor