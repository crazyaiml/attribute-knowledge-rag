from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from akrag.models import AttributeDocument


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_xlsx(path: str | Path) -> list[dict[str, Any]]:
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError("Install openpyxl: pip install akrag[excel]") from exc

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:]]


def read_ndjson(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_ndjson(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_file(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return read_csv(p)
    if suffix in (".xlsx", ".xls"):
        return read_xlsx(p)
    if suffix in (".ndjson", ".jsonl"):
        return read_ndjson(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def write_ndjson(documents: list[AttributeDocument], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(doc.model_dump_json(exclude={"vector"}) + "\n")


def documents_to_ndjson_str(documents: list[AttributeDocument]) -> str:
    lines = [doc.model_dump_json(exclude={"vector"}) for doc in documents]
    return "\n".join(lines)
