"""Bounded retries for transient storage errors."""

import errno
from time import sleep


TRANSIENT_ERRORS = {errno.EAGAIN, errno.EIO, errno.ECONNRESET, errno.ESTALE, errno.ETIMEDOUT}


def retry_read(operation, *args):
    for attempt in range(6):
        try:
            return operation(*args)
        except OSError as error:
            if attempt == 5 or not (isinstance(error, TimeoutError) or error.errno in TRANSIENT_ERRORS):
                raise
            delay = 2**attempt
            print(f"Storage read failed ({error}); retry {attempt + 1}/5 in {delay}s", flush=True)
            sleep(delay)
