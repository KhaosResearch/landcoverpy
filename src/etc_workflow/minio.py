import signal
import time
from typing import Callable
from urllib3.exceptions import HTTPError

from minio import Minio
from minio.error import MinioException

from etc_workflow.config import settings
from etc_workflow.exceptions import RuntimeMinioException

# Signal used for simulating a time-out in minio connections.
def _timeout_handler(signum, frame):
    raise RuntimeMinioException


# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, _timeout_handler)

def retry(n_retries: int, delay: int):
    """
        Adds a timeout for minio connections of 250s.
        If connection fails for timeout or any other reason, it will retry a maximum of `n_retries` times with a delay of `delay` seconds.
    """
    def _safe_minio_execute(func: Callable):
        def _wrapper_safe_minio_execute(*args, **kwargs):
            attempt = 0
            while attempt < n_retries:
                signal.alarm(
                    250
                )
                try:
                    f =  func(*args, **kwargs)
                except (RuntimeMinioException, MinioException, HTTPError) as e:
                    print(f"MinIO-related error. Retrying in one minute. Trace:\n{e}")
                    signal.alarm(0)
                    time.sleep(delay)
                    attempt += 1
            signal.alarm(0)
            return f
        return _wrapper_safe_minio_execute
    return _safe_minio_execute
            

class MinioConnection(Minio):
    "A class including handled MinIO methods"
    def __init__(self, host=settings.MINIO_HOST, port=settings.MINIO_PORT, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY):
        super().__init__(
                    endpoint = f"{host}:{port}",
                    access_key = access_key,
                    secret_key = secret_key,
                    secure = False
                )

    @retry(n_retries=100, delay=60)
    def fget_object(self, *args, **kwargs):
        "Handled version of the fget_object Minio's method"
        super().fget_object(*args, **kwargs)
        
    @retry(n_retries=100, delay=60)
    def fput_object(self, *args, **kwargs):
        "Handled version of the fput_object Minio's method"
        super().fput_object(*args, **kwargs)