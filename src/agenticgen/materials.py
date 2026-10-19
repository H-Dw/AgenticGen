"""Prepare provenance-bound local assets; never infer an inventory from RAG examples."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import tarfile
import time
from pathlib import Path

import httpx

from agenticgen.io import read_json, sha256, utc_now, write_json

SIMP_RETRO_STOCK_PAGE = "https://drive.google.com/file/d/1x33LmAizIdA5Dgw7IJp7_k7gRdNVv5cT/view"
SIMP_RETRO_STOCK_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1x33LmAizIdA5Dgw7IJp7_k7gRdNVv5cT&export=download&confirm=t"
)


def download_official_stock(output: Path) -> dict:
    """Download only the public stock linked by the pinned SimpRetro README."""
    if output.exists():
        metadata = read_json(output.with_suffix(output.suffix + ".json"))
        if sha256(output) != metadata["sha256"]:
            raise ValueError("Existing stock differs from its download receipt")
        return metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    digest, size, last = hashlib.sha256(), 0, time.monotonic()
    with httpx.stream("GET", SIMP_RETRO_STOCK_URL, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        if "application/octet-stream" not in response.headers.get("content-type", ""):
            raise ValueError("Stock download returned a login/HTML page, not the inventory")
        expected = int(response.headers["content-length"])
        with partial.open("wb") as stream:
            for block in response.iter_bytes(chunk_size=1024 * 1024):
                stream.write(block)
                digest.update(block)
                size += len(block)
                if time.monotonic() - last > 15:
                    print(f"Stock download: {size}/{expected} bytes", flush=True)
                    last = time.monotonic()
    if size != expected:
        raise ValueError("Incomplete stock download; partial file retained")
    metadata = {
        "schema_version": "agenticgen.stock-download.v1",
        "downloaded_at": utc_now(),
        "source": "Official SimpRetro README: in-stock molecules download",
        "source_page": SIMP_RETRO_STOCK_PAGE,
        "resolved_url": str(response.url),
        "stock_identity": "SimpRetro official emolecules.txt; NOT SynCraft paper Enamine stock",
        "bytes": size,
        "sha256": digest.hexdigest(),
    }
    partial.replace(output)
    write_json(output.with_suffix(output.suffix + ".json"), metadata)
    return metadata


def fetch_reference_corpus(directory: Path) -> dict:
    """Fetch the public paper release and verify its internal SHA256 manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata_url = "https://api.figshare.com/v2/articles/33201870"
    response = httpx.get(metadata_url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    article = response.json()
    released = next(f for f in article["files"] if f["name"] == "syncraft_data_release_v1.tar.gz")
    archive = directory / released["name"]
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".partial")
        with httpx.stream(
            "GET", released["download_url"], follow_redirects=True, timeout=120
        ) as download:
            download.raise_for_status()
            with partial.open("wb") as stream:
                for block in download.iter_bytes(1024 * 1024):
                    stream.write(block)
        if partial.stat().st_size != released["size"]:
            raise ValueError("Incomplete Figshare archive")
        partial.replace(archive)
    if archive.stat().st_size != released["size"]:
        raise ValueError("Figshare archive size differs from release metadata")
    if released.get("computed_md5"):
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "md5").hexdigest() != released["computed_md5"]:
                raise ValueError("Figshare archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as tar:
        manifest = tar.extractfile("./MANIFEST.sha256").read().decode("utf-8")
        checksums = dict(
            line.split(maxsplit=1)[::-1] for line in manifest.splitlines() if line.strip()
        )
        raw = tar.extractfile("./cliff_pair_corpus/cliff_pairs.json").read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != checksums["./cliff_pair_corpus/cliff_pairs.json"]:
            raise ValueError("Reference corpus differs from publisher release manifest")
        # Read only named members; never extract arbitrary archive paths.
        (directory / "cliff_pairs.json").write_bytes(raw)
        (directory / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
        (directory / "LICENSE").write_bytes(tar.extractfile("./LICENSE").read())
    corpus = json.loads(raw)
    receipt = {
        "schema_version": "agenticgen.reference-release.v1",
        "source": metadata_url,
        "download_url": released["download_url"],
        "license": article["license"],
        "archive_sha256": sha256(archive),
        "corpus_sha256": digest,
        "records": len(corpus),
        "fields": sorted(corpus[0]),
        "route_certificates_present": False,
    }
    write_json(directory / "receipt.json", receipt)
    return receipt


def prepare_template_cache(root: Path, output: Path, image: str) -> dict:
    templates = root / "SimpRetro_templates.json"
    if output.exists():
        metadata = read_json(output.with_suffix(output.suffix + ".json"))
        if metadata["templates_sha256"] != sha256(templates) or metadata["sha256"] != sha256(
            output
        ):
            raise ValueError("Existing fingerprint cache differs from its inputs/receipt")
        return metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = Path(__file__).parent / "workers"
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            "1",
            "--mount",
            f"type=bind,source={root.resolve()},target=/simpretro,readonly",
            "--mount",
            f"type=bind,source={workers.resolve()},target=/workers,readonly",
            "--mount",
            f"type=bind,source={output.parent.resolve()},target=/job",
            "--env",
            "PYTHONPATH=/simpretro",
            image,
            "python",
            "/workers/prepare_retro_cache.py",
            "/simpretro/SimpRetro_templates.json",
            "/job/" + output.name,
        ],
        check=True,
        timeout=1800,
    )
    return read_json(output.with_suffix(output.suffix + ".json"))


def index_stock(source: Path, output: Path) -> dict:
    """Disk-backed exact membership, matching upstream canonicalize_inventory=false.

    Preserve strings and stereochemistry. No canonicalization, salt stripping,
    similarity-based membership, or generated targets are added to the stock.
    """
    if output.exists():
        metadata = read_json(output.with_suffix(output.suffix + ".json"))
        if sha256(source) != metadata["inventory_sha256"] or sha256(output) != metadata["sha256"]:
            raise ValueError("Stock index does not match its inventory/receipt")
        return metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("An incomplete index exists; inspect it before retrying")
    db = sqlite3.connect(partial)
    count, blank, batch, last = 0, 0, [], time.monotonic()
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA cache_size=-65536")
        db.execute("CREATE TABLE stock(smiles TEXT PRIMARY KEY) WITHOUT ROWID")
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                smiles = line.strip()
                if not smiles:
                    blank += 1
                    continue
                if any(c.isspace() for c in smiles):
                    raise ValueError("Inventory must contain exactly one SMILES per line")
                count += 1
                batch.append((smiles,))
                if len(batch) == 50000:
                    db.executemany("INSERT OR IGNORE INTO stock VALUES(?)", batch)
                    db.commit()
                    batch.clear()
                    if time.monotonic() - last > 15:
                        print(f"Stock indexed: {count} lines", flush=True)
                        last = time.monotonic()
        db.executemany("INSERT OR IGNORE INTO stock VALUES(?)", batch)
        db.commit()
        unique = db.execute("SELECT COUNT(*) FROM stock").fetchone()[0]
        if not unique:
            raise ValueError("Empty stock cannot be used for search")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Stock index integrity check failed")
    finally:
        db.close()
    metadata = {
        "schema_version": "agenticgen.stock-index.v1",
        "created_at": utc_now(),
        "inventory_sha256": sha256(source),
        "sha256": sha256(partial),
        "input_nonempty_lines": count,
        "blank_lines": blank,
        "unique_smiles": unique,
        "bytes": partial.stat().st_size,
        "canonicalize_inventory": False,
        "membership": "exact strings, same as upstream search default and SimpRetro ASScore",
    }
    partial.replace(output)
    write_json(output.with_suffix(output.suffix + ".json"), metadata)
    return metadata
