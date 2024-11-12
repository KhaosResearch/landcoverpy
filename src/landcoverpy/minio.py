import signal
import time
from typing import Callable

from minio import Minio
from minio.error import MinioException
from urllib3.exceptions import HTTPError

from landcoverpy.config import settings
from landcoverpy.exceptions import RuntimeMinioException

class MinioConnection(Minio):
    "A class including handled MinIO methods"
    def __init__(self, host=settings.MINIO_HOST, port=settings.MINIO_PORT, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY):
        super().__init__(
                    endpoint = f"{host}:{port}",
                    access_key = access_key,
                    secret_key = secret_key,
                    secure = False
                )

    def fget_object(self, *args, **kwargs):
        "Handled version of the fget_object Minio's method"
        super().fget_object(*args, **kwargs)
        
    def fput_object(self, *args, **kwargs):
        "Handled version of the fput_object Minio's method"
        super().fput_object(*args, **kwargs)