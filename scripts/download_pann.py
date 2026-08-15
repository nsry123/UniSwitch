"""Download and verify the optional official PANN CNN14 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

CHECKPOINT_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
CHECKPOINT_SHA256 = "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"
DEFAULT_DESTINATION = Path("models/Cnn14_mAP=0.431.pth")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(destination: Path, force: bool = False) -> None:
    """Download to a temporary file, verify it, then atomically install it."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        if sha256_file(destination) == CHECKPOINT_SHA256:
            print(f"Checkpoint already verified: {destination}")
            return
        raise RuntimeError(f"Refusing to overwrite unverified file: {destination}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".download",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            print(f"Downloading {CHECKPOINT_URL}")
            with urlopen(CHECKPOINT_URL) as response:  # noqa: S310 - fixed trusted URL
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)

        actual_digest = sha256_file(temporary_path)
        if actual_digest != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Checkpoint checksum mismatch: expected {CHECKPOINT_SHA256}, "
                f"received {actual_digest}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"Verified checkpoint installed at {destination}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download_checkpoint(arguments.destination, force=arguments.force)
