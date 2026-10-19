import ast
import hashlib
import io
import json
import sqlite3
import tarfile
from types import SimpleNamespace

import httpx
import pytest

from agenticgen.materials import download_official_stock, fetch_reference_corpus, index_stock
from agenticgen.workers.stock_index import DiskMolInventory, SQLiteStock


def test_stock_download_rejects_html_and_records_binary_hash(tmp_path, monkeypatch):
    def mocked(content, kind):
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=content,
                    headers={"content-type": kind, "content-length": str(len(content))},
                )
            )
        )
        monkeypatch.setattr(httpx, "stream", lambda method, url, **kw: client.stream(method, url))

    path = tmp_path / "stock.smi"
    mocked(b"<html>Login</html>", "text/html")
    with pytest.raises(ValueError, match="login/HTML"):
        download_official_stock(path)
    assert not path.exists()
    mocked(b"CCO\nCC\n", "application/octet-stream")
    receipt = download_official_stock(path)
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "NOT" in receipt["stock_identity"]
    assert download_official_stock(path) == receipt


def test_index_preserves_upstream_exact_membership_and_stereo(tmp_path):
    raw = tmp_path / "stock.smi"
    raw.write_text("CCO\nOCC\nCCO\nC[C@H](O)Cl\n\n", encoding="utf-8")
    index = tmp_path / "stock.sqlite"
    receipt = index_stock(raw, index)
    assert receipt["input_nonempty_lines"] == 4 and receipt["unique_smiles"] == 3
    stock = SQLiteStock(index)
    expected = {s.strip() for s in raw.read_text().splitlines() if s.strip()}
    for smiles in ["CCO", "OCC", "COC", "C[C@H](O)Cl", "C[C@@H](O)Cl"]:
        assert (smiles in stock) == (smiles in expected)
    inventory = DiskMolInventory(index)
    mol = SimpleNamespace(smiles="CCO", identifier=None, metadata={})
    inventory.fill_metadata(mol)
    assert mol.metadata["is_purchasable"]
    mol.identifier = "other-id"
    assert not inventory.is_purchasable(mol)
    with pytest.raises(sqlite3.OperationalError):
        stock.db.execute("DELETE FROM stock")
    assert index_stock(raw, index) == receipt
    raw.write_text("COC\n")
    with pytest.raises(ValueError, match="does not match"):
        index_stock(raw, index)


def test_memory_adapter_rewrites_only_pinned_stock_sites(repo, monkeypatch):
    monkeypatch.syspath_prepend(str(repo / "src/agenticgen/workers"))
    from agenticgen.workers.retro_search import replace_stock_loading

    root = repo / "external/SimpRetro/syntheseus"
    for filename, name, constructor in [
        ("cli/search.py", "run_from_config", False),
        ("reaction_prediction/inference/simpretro.py", "__init__", True),
    ]:
        text = (root / filename).read_text()
        tree = ast.parse(text)
        function = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
        )
        rewritten = replace_stock_loading(
            ast.get_source_segment(text, function), constructor=constructor
        )
        compiled = ast.unparse(rewritten)
        assert ("SQLiteStock" if constructor else "DiskMolInventory") in compiled
        if not constructor:
            assert "alg.run_from_mol" in compiled and "MolSetMCTS" in compiled
            assert "inventory_smiles =" not in compiled
    with pytest.raises(ValueError, match="AST changed"):
        replace_stock_loading("def wrong(self):\n    return 1", constructor=True)


def test_reference_release_checks_manifest_and_does_not_extract_other_paths(tmp_path, monkeypatch):
    raw = json.dumps([{"source": "CCO", "edits": [["STOP"]], "reasoning": "Fixture"}]).encode()
    archive = tmp_path / "syncraft_data_release_v1.tar.gz"

    def prepare_archive(expected):
        with tarfile.open(archive, "w:gz") as tar:
            members = {
                "./cliff_pair_corpus/cliff_pairs.json": raw,
                "./MANIFEST.sha256": (
                    expected + "  ./cliff_pair_corpus/cliff_pairs.json\n"
                ).encode(),
                "./LICENSE": b"Engineering fixture",
                "../must_not_extract": b"Do not extract arbitrary archive members",
            }
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        reply = {
            "files": [
                {
                    "name": archive.name,
                    "size": archive.stat().st_size,
                    "download_url": "https://example.invalid/not-requested",
                }
            ],
            "license": {"name": "fixture"},
        }
        monkeypatch.setattr(
            httpx,
            "get",
            lambda *a, **kw: httpx.Response(
                200, json=reply, request=httpx.Request("GET", "https://example.invalid/metadata")
            ),
        )

    prepare_archive(hashlib.sha256(raw).hexdigest())
    receipt = fetch_reference_corpus(tmp_path)
    assert receipt["records"] == 1
    assert not (tmp_path.parent / "must_not_extract").exists()
    assert not receipt["route_certificates_present"]
    prepare_archive("0" * 64)
    with pytest.raises(ValueError, match="manifest"):
        fetch_reference_corpus(tmp_path)
