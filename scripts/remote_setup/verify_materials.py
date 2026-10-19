"""Verify copied SimpRetro materials against their paired provenance receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    entries = {}
    for name in ("emolecules.txt", "stock.sqlite", "fingerprint_base.pkl"):
        path = args.directory / name
        receipt = json.loads(path.with_name(name + ".json").read_text(encoding="utf-8"))
        actual = digest(path)
        expected_size = receipt["bytes"]
        ok = actual == receipt["sha256"] and path.stat().st_size == expected_size
        entries[name] = {
            "ok": ok,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "receipt_schema": receipt["schema_version"],
        }
        if not ok:
            raise SystemExit(f"Material integrity check failed: {name}")
    index_receipt = json.loads((args.directory / "stock.sqlite.json").read_text(encoding="utf-8"))
    linked = index_receipt["inventory_sha256"] == entries["emolecules.txt"]["sha256"]
    if not linked:
        raise SystemExit("Stock index is not bound to the copied inventory")
    db_uri = (args.directory / "stock.sqlite").resolve().as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(db_uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        ethanol = connection.execute("SELECT 1 FROM stock WHERE smiles=?", ("CCO",)).fetchone()
    result = {
        "ok": True,
        "materials": entries,
        "inventory_index_hash_link": linked,
        "read_only_membership_probe": {"CCO": ethanol is not None},
        "stock_identity": "SimpRetro official eMolecules; not the paper's Enamine stock",
        "route_search_performed": False,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
