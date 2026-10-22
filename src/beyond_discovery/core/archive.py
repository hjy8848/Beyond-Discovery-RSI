"""Exclusive run ownership, immutable content objects and fsynced hash chain.

Hash chaining detects corruption; it is not protection against the machine's
owner replacing the entire archive. Agents receive values, never archive paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from .contracts import canonical, digest, plain


def create_json(path, value):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Archive:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.objects = self.path / "objects"
        self.objects.mkdir(mode=0o700)
        self.journal = self.path / "journal.jsonl"
        fd = os.open(str(self.journal), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        self.head = None
        self.sequence = 0

    def append(self, kind, value):
        data = plain(value)
        object_hash = digest(data)
        object_path = self.objects / (object_hash + ".json")
        if not object_path.exists():
            create_json(object_path, data)
        elif json.loads(object_path.read_text()) != data:
            raise ValueError("immutable object collision")
        row = {"sequence": self.sequence, "parent_hash": self.head,
               "kind": kind, "object_hash": object_hash}
        row["content_hash"] = digest(row)
        with self.journal.open("a") as handle:
            handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.head = row["content_hash"]
        self.sequence += 1
        return object_hash

    def marker(self, name, value):
        create_json(self.path / name, value)


def read_verified(path):
    root = Path(path)
    head = None
    records = []
    for sequence, line in enumerate((root / "journal.jsonl").read_text().splitlines()):
        row = json.loads(line)
        expected = row.pop("content_hash")
        if digest(row) != expected or row["parent_hash"] != head or row["sequence"] != sequence:
            raise ValueError("journal hash chain mismatch")
        data = json.loads((root / "objects" / (row["object_hash"] + ".json")).read_text())
        if digest(data) != row["object_hash"]:
            raise ValueError("archive object hash mismatch")
        records.append((row["kind"], data))
        head = expected
    return records
