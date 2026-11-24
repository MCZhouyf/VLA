"""Copy and verify frozen T1 source archives on local storage."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from t1_storage import retry_read, trajectory_fingerprint


def checksum(path, destination=None):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(2**20):
            digest.update(chunk)
            if destination is not None:
                destination.write(chunk)
    return digest.hexdigest()


def verify(path, expected):
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("episode.json"))
        images = (member for member in archive.infolist() if member.filename.endswith(".jpg"))
        fingerprint = trajectory_fingerprint(images, metadata)
        bad_member = archive.testzip()
    if fingerprint != expected or bad_member is not None:
        raise ValueError(f"Source fingerprint or ZIP CRC mismatch for {path}")


def stage(source, destination, expected):
    if destination.exists():
        digest = retry_read(checksum, source)
        if checksum(destination) != digest:
            raise ValueError(f"Copied archive SHA-256 differs from source {destination}")
        verify(destination, expected)
        return dict(name=destination.name, bytes=destination.stat().st_size, sha256=digest)

    return retry_read(copy_new, source, destination, expected)


def copy_new(source, destination, expected):
    with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.") as temp:
        staged = Path(temp.name)
        digest = checksum(source, temp)
        temp.flush()
        if checksum(staged) != digest:
            raise ValueError(f"Copied archive SHA-256 differs from source {staged}")
        verify(staged, expected)
        # Hard-link the verified file atomically before the temporary name is removed.
        try:
            destination.hardlink_to(staged)
        except FileExistsError:
            if checksum(destination) != digest:
                raise ValueError(f"Concurrent archive copy differs from source {destination}")
            verify(destination, expected)
    return dict(name=destination.name, bytes=destination.stat().st_size, sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("Workers must be positive")
    rows = {}
    dataset = args.dataset.read_bytes()
    for line in dataset.splitlines():
        row = json.loads(line)
        source = Path(row["path"])
        entry = source, row["fingerprint"]
        if rows.setdefault(source.name, entry) != entry:
            raise ValueError(f"Conflicting archive basename: {source.name}")
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        copies = [pool.submit(stage, source, args.output / name, expected)
                  for name, (source, expected) in rows.items()]
        records = [job.result() for job in copies]
    size = sum(record["bytes"] for record in records)
    with NamedTemporaryFile(dir=args.manifest.parent, mode="w", delete=False) as temp:
        for record in records:
            temp.write(json.dumps(record) + "\n")
    Path(temp.name).replace(args.manifest)
    print(json.dumps(dict(archives=len(rows), bytes=size,
                          dataset_sha256=hashlib.sha256(dataset).hexdigest(),
                          path=str(args.output.resolve())), indent=2))


if __name__ == "__main__":
    main()
