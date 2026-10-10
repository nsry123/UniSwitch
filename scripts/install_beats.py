"""Install a manually obtained official BEATs checkpoint with digest verification."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

BEATS_CHECKPOINT_PATH = Path("models/BEATs_iter3_plus_AS2M.pt")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_digest(value: str) -> str:
    """Normalize a trusted SHA-256 digest or raise a CLI-friendly error."""
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("sha256 must be exactly 64 hexadecimal characters")
    return normalized


def install_checkpoint(
    source: Path,
    expected_sha256: str,
    destination: Path = BEATS_CHECKPOINT_PATH,
    force: bool = False,
) -> None:
    """Verify a source checkpoint, then atomically install it and its digest sidecar."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    expected = validate_digest(expected_sha256)
    if not source.is_file():
        raise FileNotFoundError(f"BEATs checkpoint not found: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint checksum mismatch: expected {expected}, received {actual}"
        )
    if destination.exists() and not force and source != destination:
        raise RuntimeError(f"Refusing to overwrite existing file: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    if source != destination:
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".install",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                with source.open("rb") as source_file:
                    shutil.copyfileobj(source_file, temporary_file, length=1024 * 1024)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    sidecar = destination.with_suffix(".sha256")
    sidecar_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{sidecar.name}.",
            suffix=".install",
            delete=False,
        ) as temporary_file:
            sidecar_temporary = Path(temporary_file.name)
            temporary_file.write(f"{expected}  {destination.name}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(sidecar_temporary, sidecar)
        sidecar_temporary = None
    finally:
        if sidecar_temporary is not None:
            sidecar_temporary.unlink(missing_ok=True)
    print(f"Verified BEATs checkpoint installed at {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", type=Path, default=BEATS_CHECKPOINT_PATH)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    install_checkpoint(
        arguments.source,
        arguments.sha256,
        destination=arguments.destination,
        force=arguments.force,
    )
