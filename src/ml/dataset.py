"""Dataset manifest parsing and leakage validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SPLITS = {"train", "validation", "test"}
REQUIRED_COLUMNS = {
    "path",
    "label",
    "subject",
    "session",
    "device",
    "sample_rate",
    "split",
}


@dataclass(frozen=True)
class AudioRecord:
    """One labeled recording and its leakage-group metadata."""

    path: Path
    label: str
    subject: str
    session: str
    device: str
    sample_rate: int
    split: str

    @property
    def group(self) -> tuple[str, str, str]:
        return (self.subject, self.session, self.device)


def load_manifest(path: str | Path, require_files: bool = True) -> list[AudioRecord]:
    """Load a CSV manifest and reject malformed or group-leaking splits."""
    manifest_path = Path(path).expanduser().resolve()
    records: list[AudioRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            values = {name: (row.get(name) or "").strip() for name in REQUIRED_COLUMNS}
            empty = sorted(name for name, value in values.items() if not value)
            if empty:
                raise ValueError(f"manifest row {row_number} has empty fields: {empty}")
            split = values["split"].lower()
            if split not in ALLOWED_SPLITS:
                raise ValueError(
                    f"manifest row {row_number} has invalid split {split!r}; "
                    f"expected one of {sorted(ALLOWED_SPLITS)}"
                )
            try:
                sample_rate = int(values["sample_rate"])
            except ValueError as error:
                raise ValueError(
                    f"manifest row {row_number} sample_rate is not an integer"
                ) from error
            if sample_rate <= 0:
                raise ValueError(f"manifest row {row_number} sample_rate must be positive")

            audio_path = Path(values["path"])
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            audio_path = audio_path.resolve()
            if require_files and not audio_path.is_file():
                raise FileNotFoundError(f"manifest row {row_number} audio not found: {audio_path}")
            records.append(
                AudioRecord(
                    path=audio_path,
                    label=values["label"],
                    subject=values["subject"],
                    session=values["session"],
                    device=values["device"],
                    sample_rate=sample_rate,
                    split=split,
                )
            )

    if not records:
        raise ValueError("manifest contains no records")
    duplicate_paths = {
        record.path for record in records if sum(r.path == record.path for r in records) > 1
    }
    if duplicate_paths:
        raise ValueError(f"manifest contains duplicate paths: {sorted(map(str, duplicate_paths))}")

    group_splits: dict[tuple[str, str, str], set[str]] = {}
    for record in records:
        group_splits.setdefault(record.group, set()).add(record.split)
    leaking_groups = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaking_groups:
        raise ValueError(f"subject/session/device groups leak across splits: {leaking_groups}")

    present_splits = {record.split for record in records}
    missing_splits = ALLOWED_SPLITS - present_splits
    if missing_splits:
        raise ValueError(f"manifest has no records for splits: {sorted(missing_splits)}")
    return records
