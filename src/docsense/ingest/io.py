"""Persistence for ingestion records.

All record collections (chunks, manifest rows) go through the `RecordStore`
abstraction so that swapping JSONL for Parquet later is a new class plus a
config value (`record_format`) rather than a refactor. Nested single documents
(`PaperDocument`, `IngestReport`) bypass `RecordStore` and are written as plain
pretty-printed JSON.
"""

import json
from pathlib import Path
from typing import Any, Protocol

from docsense.ingest.schema import Chunk, IngestReport, ManifestRecord, PaperDocument


class RecordStore(Protocol):
    """Append-oriented store for flat dict records (chunks, manifest rows).

    The only persistence interface callers may use for record collections.
    """

    extension: str

    def append(self, records: Any, path: Path) -> None:
        """Append `records` (an iterable of flat dicts) to `path`."""
        ...

    def read_all(self, path: Path) -> list[dict[str, Any]]:
        """Read every record from `path`; `[]` if `path` does not exist."""
        ...

    def count(self, path: Path) -> int:
        """Number of records stored at `path`."""
        ...


class JsonlRecordStore:
    """RecordStore over newline-delimited JSON.

    Creates parent directories on append. `read_all` on a missing path returns `[]`.
    """

    extension: str = "jsonl"

    def append(self, records: Any, path: Path) -> None:
        """Append each record in `records` as one JSON line to `path`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for record in records:
                f.write(json.dumps(record))
                f.write("\n")

    def read_all(self, path: Path) -> list[dict[str, Any]]:
        """Read every record from `path`; `[]` if `path` does not exist."""
        if not path.exists():
            return []
        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def count(self, path: Path) -> int:
        """Number of records stored at `path`."""
        return len(self.read_all(path))


def get_record_store(fmt: str = "jsonl") -> RecordStore:
    """Factory keyed by config `record_format`.

    Args:
        fmt: backend name, e.g. "jsonl".

    Raises:
        ValueError: if `fmt` is unknown.
    """
    if fmt == "jsonl":
        return JsonlRecordStore()
    raise ValueError(f"Unknown record_format: {fmt!r}")


def write_chunks(chunks: list[Chunk], path: Path, store: RecordStore) -> None:
    """Append `chunks` via `store` -- the corpus-wide file Phase 2 reads."""
    store.append((chunk.model_dump(mode="json") for chunk in chunks), path)


def read_chunks(path: Path, store: RecordStore) -> list[Chunk]:
    """Read and validate every Chunk from `path`."""
    return [Chunk.model_validate(record) for record in store.read_all(path)]


def append_manifest(record: ManifestRecord, path: Path, store: RecordStore) -> None:
    """Append one ManifestRecord. Append-only keeps fetch runs resumable."""
    store.append([record.model_dump(mode="json")], path)


def read_manifest(path: Path, store: RecordStore) -> list[ManifestRecord]:
    """Read and validate every ManifestRecord from `path`; `[]` if `path` is absent."""
    return [ManifestRecord.model_validate(record) for record in store.read_all(path)]


def write_paper(doc: PaperDocument, out_dir: Path) -> Path:
    """Write `out_dir/papers/<arxiv_id>.json` (model_dump_json, indent=2). Returns the path."""
    path = out_dir / "papers" / f"{doc.arxiv_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.model_dump_json(indent=2))
    return path


def read_paper(path: Path) -> PaperDocument:
    """Read and validate a PaperDocument from `path`."""
    return PaperDocument.model_validate_json(path.read_text())


def write_ingest_report(report: IngestReport, out_dir: Path) -> Path:
    """Write `out_dir/ingest_report.json` (model_dump_json, indent=2). Returns the path."""
    path = out_dir / "ingest_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2))
    return path
