from datetime import datetime

from sentinelsat.sentinel import SentinelAPI

from bd_lc_mediterranean.config import settings


def _get_sentinel():
    """
    Initialize Sentinel client
    """
    sentinel_api = SentinelAPI(
        user=settings.SENTINEL_USERNAME,
        password=settings.SENTINEL_PASSWORD,
        api_url=settings.SENTINEL_HOST,
        show_progressbars=False,
    )
    return sentinel_api

def _sentinel_date_to_datetime(date: str):
    """
    Parse a string date (YYYYMMDDTHHMMSS) to a sentinel datetime
    """
    date_datetime = datetime(
        int(date[0:4]),
        int(date[4:6]),
        int(date[6:8]),
        int(date[9:11]),
        int(date[11:13]),
        int(date[13:15]),
    )
    return date_datetime