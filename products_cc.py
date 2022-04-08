import datetime
import json
from pathlib import Path
from typing import Collection
import pandas as pd
import os
from pymongo import MongoClient
from sentinelsat.sentinel import SentinelAPI, geojson_to_wkt, read_geojson
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Initialize Sentinel client
sentinel_api = SentinelAPI(
    user = os.environ.get("DHUS_USERNAME"),
    password = os.environ.get("DHUS_PASSWORD"),
    api_url = os.environ.get("DHUS_HOST"),
    show_progressbars=False,
)

def connect_mongo_products_collection():
    '''
    Return a collection from Mongo DB
    '''
    mongo_client = MongoClient(
    f"mongodb://" + os.environ.get("MONGO_HOST") + ":" + os.environ.get("MONGO_PORT") + "/",
    username=os.environ.get("MONGO_USERNAME"),
    password=os.environ.get("MONGO_PASSWORD"),
)
    mongo_db = mongo_client[os.environ.get("MONGO_DATABASE_NAME")]
    mongo_col = mongo_db[os.environ.get("MONGO_COLLECTION_NAME")]

    return mongo_col


def get_products_by_tile_and_date(tiles: dict, mongo_collection: Collection,  seasons: dict):
    '''
        Query to mongo for obtaining products of the given dates and tiles and create a dictionary with all available products in mongo with its cloud percentage.
        
        Parameters:
            tiles (dict): Dictionary containing tiles names.
            mongo_collection (Colection): Data for mongo access.
            seasons (dict): Dictionary of date range (start and end date) for each query.
    
        Output:
            tiles (dict): Dictionary order by tile with all available products in mongo and with its cloud percentage.

    '''
    missing_tiles = []
    missing_indices = []
    # Obtain title and cloud cover percentage of every product available for each tile and each season in the given seasons list
    for tile in tiles:
        for season in seasons:
            start_date, end_date = seasons[season]
    
            product_metadata_cursor = mongo_collection.aggregate([
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
                '$match': {
                    'title': {
                        '$regex': f'_{tile}_'
                    },
                    'date': {
                        '$gte': start_date,
                        '$lt': end_date,
                    }
                }
            }])
            products = list(product_metadata_cursor)

            # Check if exist products for the given tile
            if len(products) == 0:
                missing_tiles.append(tile)
            
            else:
                dict_products = {}
                # Generate a dictionary of the products set with title as key and cloud cover as value
                for product in products:
                    title = product['title']
                    # Check if the product has indices (some products do not have indices calculate due to an error in the download process)
                    if len(product['indexes']) == 0: 
                        missing_indices.append(product['id'])                                        
                    else:
                        dict_products[title] = product['indexes'][0]['value']
                            
                # Add the dictionary of products to the original tiles dictionary by tile and season
                tiles[tile][season] = dict_products
    # Save dictionary to a JSON file
    with open('dict_products.json', 'w') as f:
        json.dump(tiles, f)

    print("missing tiles", missing_tiles)
    print("missing indices", missing_indices)
    return tiles


def get_tiles(sentinel_api: SentinelAPI, countries: list, order_by_country = False):
    '''
        Get set of tiles for a given area.

        Parameters:
            sentinel_api (SentinelAPI): Sentinel api access data.
            countries (list): Set of countries to get tiles of.
            order_by_country (bool): If True the output dictionary is order by country, otherwise, the dictionary will have only tile names.
    
    '''

    tiles_country = {}  

    for country in countries:
        # Read geojson of the given country
        geojson = read_geojson(Path("geojson/" + str.lower(country) + ".geojson"))
        footprint = geojson_to_wkt(geojson)

        start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
        end_date = datetime.strptime("2021-01-20", "%Y-%m-%d")

        products = sentinel_api.query(
            area=footprint,
            filename="S2*",
            producttype="S2MSI2A",
            platformname="Sentinel-2",
            cloudcoverpercentage=(0, 100),
            date=(start_date, end_date),
            order_by="cloudcoverpercentage, +beginposition",
            limit=None,
        )

        products_df = sentinel_api.to_dataframe(products)
        ids = products_df.index


        # Get list of unique tiles in the given area.
        tiles = pd.Series(products_df["title"]).str[38:44]
        tiles = pd.unique(tiles)

        # Add tiles to the dictionary
        for tile in tiles:
            tiles_country[tile] = {}
        
        if order_by_country:
            tiles_country = {country: tiles_country}
    #Save tiles dictionary to a JSON file
    with open('tiles.json', 'w') as f:
        json.dump(tiles_country, f)
    print(tiles_country)



if __name__ == "__main__":
    countries = ['albania', 'algeria', 'bulgaria', 'canarias', 'croatia', 'egypt', 'france', 'gibraltar', 'greece', 'italy', 'lybia', 'malta', 'montenegro', 'morocco', 'slovenia', 'spain', 'syria', 'tunisia', 'turkey']
    mong = connect_mongo_products_collection()
    #tiles = get_tiles(sentinel_api, countries)
    with open('tiles.json') as json_file:
        tiles = json.load(json_file)
    
    seasons = {'Spring' : (datetime.strptime("2021-03-01", "%Y-%m-%d"), datetime.strptime("2021-04-01", "%Y-%m-%d")),
               'Summer' : (datetime.strptime("2021-06-01", "%Y-%m-%d"), datetime.strptime("2021-07-01", "%Y-%m-%d")),
               'Autumn' : (datetime.strptime("2021-11-01", "%Y-%m-%d"), datetime.strptime("2021-12-01", "%Y-%m-%d"))}
    res = get_products_by_tile_and_date(mongo_collection=mong, tiles=tiles, seasons=seasons)