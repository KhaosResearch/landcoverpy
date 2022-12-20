from datetime import datetime

from sentinelsat.sentinel import SentinelAPI

from landcoverpy.config import settings


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
    date_datetime = datetime.strptime(date, '%Y%m%dT%H%M%S')
    return date_datetime