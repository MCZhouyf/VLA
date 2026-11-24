"""Bounded retries for transient storage errors."""

import errno
import hashlib
import json
from time import sleep


TRANSIENT_ERRORS = {errno.EAGAIN, errno.EIO, errno.ECONNRESET, errno.ESTALE, errno.ETIMEDOUT}


def trajectory_fingerprint(images, metadata):
    digest = hashlib.sha256()
    for member in images:
        digest.update(member.CRC.to_bytes(4, "little"))
    digest.update(json.dumps(metadata["action"]).encode())
    if "executed_action" in metadata:
        digest.update(json.dumps(metadata["executed_action"]).encode())
        digest.update(json.dumps(metadata["perturbation_applied"]).encode())
    return digest.hexdigest()


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
