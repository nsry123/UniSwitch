import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.install_beats import install_checkpoint


def test_install_beats_is_runnable_by_script_path() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/install_beats.py", "--help"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--sha256" in result.stdout


def test_install_beats_verifies_and_writes_sidecar(tmp_path) -> None:
    source = tmp_path / "downloaded.pt"
    destination = tmp_path / "models" / "beats.pt"
    source.write_bytes(b"official-checkpoint")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    install_checkpoint(source, digest, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.with_suffix(".sha256").read_text().split()[0] == digest


def test_install_beats_rejects_wrong_digest(tmp_path) -> None:
    source = tmp_path / "downloaded.pt"
    source.write_bytes(b"checkpoint")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        install_checkpoint(source, "0" * 64, tmp_path / "beats.pt")
