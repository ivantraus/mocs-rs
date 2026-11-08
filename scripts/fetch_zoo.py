#!/usr/bin/env python3
"""Fetch the public UCI Zoo archive and convert it to a provenance-marked ARFF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import pathlib
import urllib.request
import zipfile


URL = "https://archive.ics.uci.edu/static/public/111/zoo.zip"
SHA256 = "d49013253628277182bb172f5ab1bf3befad090b54bbb7e52733c9c9534e240a"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/111/zoo"
ATTRIBUTES = (
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
    "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail",
    "domestic", "catsize",
)
NOMINAL = {
    "hair": "{0,1}", "feathers": "{0,1}", "eggs": "{0,1}", "milk": "{0,1}",
    "airborne": "{0,1}", "aquatic": "{0,1}", "predator": "{0,1}", "toothed": "{0,1}",
    "backbone": "{0,1}", "breathes": "{0,1}", "venomous": "{0,1}", "fins": "{0,1}",
    "legs": "{0,2,4,5,6,8}", "tail": "{0,1}", "domestic": "{0,1}", "catsize": "{0,1}",
}
MAX_ARCHIVE_BYTES = 1_000_000


def convert_zoo(data: bytes) -> tuple[str, int]:
    """Convert zoo.data bytes; read the member without extracting the archive."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        matches = [info for info in archive.infolist() if info.filename == "zoo.data"]
        if len(matches) != 1:
            raise ValueError("archive must contain exactly one zoo.data member")
        info = matches[0]
        if info.file_size > MAX_ARCHIVE_BYTES or info.is_dir():
            raise ValueError("zoo.data member has an invalid size or type")
        raw = archive.read(info)
    rows: list[list[str]] = []
    for number, row in enumerate(csv.reader(io.StringIO(raw.decode("utf-8"))), 1):
        if not row:
            continue
        if len(row) != 18:
            raise ValueError(f"zoo.data row {number} has {len(row)} fields; expected 18")
        fields = [field.strip() for field in row[1:]]  # remove animal_name identifier
        if any(not value for value in fields):
            raise ValueError(f"zoo.data row {number} contains an empty field")
        if any(fields[i] not in ("0", "1") for i in range(12)) or fields[12] not in NOMINAL["legs"][1:-1].split(","):
            raise ValueError(f"zoo.data row {number} contains an invalid feature value")
        if any(fields[i] not in ("0", "1") for i in range(13, 16)):
            raise ValueError(f"zoo.data row {number} contains an invalid binary feature")
        if fields[16] not in tuple(str(i) for i in range(1, 8)):
            raise ValueError(f"zoo.data row {number} contains an invalid class")
        rows.append(fields)
    if len(rows) != 101:
        raise ValueError(f"expected 101 Zoo rows, found {len(rows)}")

    lines = ["% UCI Zoo: animal_name identifier removed; original row order retained.", "@relation zoo"]
    lines.extend(f"@attribute {name} {NOMINAL[name]}" for name in ATTRIBUTES)
    lines.append("@attribute type {1,2,3,4,5,6,7}")
    lines.append("@data")
    lines.extend(",".join(row) for row in rows)
    return "\n".join(lines) + "\n", len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent / "results/data")
    args = parser.parse_args()
    request = urllib.request.Request(URL, headers={"User-Agent": "mocs-rs-reproducible-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("download exceeds the 1 MB safety limit")
    digest = hashlib.sha256(archive_bytes).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Zoo archive SHA-256 mismatch: expected {SHA256}, got {digest}")
    arff, row_count = convert_zoo(archive_bytes)
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arff_path = output_dir / "zoo.arff"
    attribution_path = output_dir / "zoo-attribution.txt"
    arff_path.write_text(arff, encoding="utf-8")
    attribution_path.write_text(
        "UCI Machine Learning Repository: Zoo\n"
        f"Dataset page: {UCI_PAGE}\n"
        f"Downloaded archive: {URL}\n"
        f"Archive SHA-256: {digest}\n"
        f"Converted rows: {row_count}\n"
        "Conversion: removed animal_name identifier, retained all 16 features and the type target, "
        "retained source row order, emitted ARFF. This file is the public UCI source dataset and "
        "does not claim to reproduce any historical train/test folds.\n"
        "Creator: Richard Forsyth. Citation: Forsyth, R. (1990). Zoo [Dataset]. UCI Machine "
        "Learning Repository. https://doi.org/10.24432/C5R59V\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0).\n",
        encoding="utf-8",
    )
    print(f"wrote {arff_path} and {attribution_path} ({row_count} rows; SHA-256 {digest})")


if __name__ == "__main__":
    main()
