"""Download and verify the three available pinned Phase 0 checkpoints."""

import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

from huggingface_hub import HfApi, snapshot_download


REVISIONS = {
    "CraftJarvis/JarvisVLA-Qwen2-VL-7B": "27e83fb89ddf9fe2a3340b9db0a85ba9c3b9c971",
    "Qwen/Qwen2-VL-2B-Instruct": "1860c46f12b7449d0f5b426a7474ab715342bc65",
    "CraftJarvis/MineStudio_STEVE-1.official": "34e18c2fd00058deb371b6c34b183eb52de8166c",
}
FIELDS = ("repo", "revision", "state", "files", "bytes", "error")


def fetch_checkpoint(work, repo, revision):
    print(f"Starting {repo} at {revision}", flush=True)
    snapshot = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if snapshot.sha != revision or not snapshot.siblings:
        raise RuntimeError("Expected a nonempty manifest for the pinned revision")

    target = work / "models" / repo.split("/", 1)[1]
    snapshot_download(repo_id=repo, revision=revision, local_dir=target, max_workers=2)
    total_bytes = 0
    for item in snapshot.siblings:
        path = target / item.rfilename
        size = path.stat().st_size
        if item.size is not None and size != item.size:
            raise RuntimeError(f"Size mismatch for {item.rfilename}")
        if item.lfs is not None:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for block in iter(lambda: file.read(4 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != item.lfs.sha256:
                raise RuntimeError(f"SHA-256 mismatch for {item.rfilename}")
        total_bytes += size
    return len(snapshot.siblings), total_bytes


def write_manifest(path, rows):
    with NamedTemporaryFile(dir=path.parent, prefix=".phase0_models_", mode="w",
                            newline="", delete=False) as stream:
        staged = Path(stream.name)
        try:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(rows)
            stream.flush()
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


def main():
    work = Path(os.environ["WORK"])
    manifest = work / "results" / "phase0_model_downloads.csv"
    if manifest.is_file():
        with manifest.open(newline="") as stream:
            verified = {item["repo"]: item["revision"] for item in csv.DictReader(stream)
                        if item["state"] == "verified"}
        if verified == REVISIONS:
            print(f"The pinned checkpoints are already recorded as verified in {manifest}", flush=True)
            return 0

    rows = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        downloads = {repo: executor.submit(fetch_checkpoint, work, repo, revision)
                     for repo, revision in REVISIONS.items()}
        for repo, revision in REVISIONS.items():
            try:
                count, size = downloads[repo].result()
            except Exception as error:
                message = f"{type(error).__name__}: {error}".splitlines()[0][:200]
                print(f"{repo} failed: {message}", flush=True)
                rows.append((repo, revision, "hf_mirror_failed", "", "", message))
            else:
                print(f"Verified {repo}, {count} files, {size} bytes", flush=True)
                rows.append((repo, revision, "verified", count, size, ""))
    write_manifest(manifest, rows)
    return int(any(row[2] != "verified" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
