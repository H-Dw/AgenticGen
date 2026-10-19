"""Read-only stock membership shared by search, ASScore and route verification."""

import sqlite3
from functools import lru_cache
from pathlib import Path


class SQLiteStock:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("PRAGMA query_only=ON")
        self._contains = lru_cache(maxsize=4096)(self._query)

    def __contains__(self, smiles):
        return self._contains(smiles)

    def _query(self, smiles):
        return (
            self.db.execute("SELECT 1 FROM stock WHERE smiles=?", (smiles,)).fetchone() is not None
        )


class DiskMolInventory:
    """Equivalent to upstream SmilesListInventory(..., canonicalize=False) membership."""

    def __init__(self, path):
        self.stock = SQLiteStock(path)

    def is_purchasable(self, mol):
        return mol.identifier is None and mol.smiles in self.stock

    def fill_metadata(self, mol):
        mol.metadata["is_purchasable"] = self.is_purchasable(mol)
