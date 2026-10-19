"""Transactional jobs, immutable inputs, execution history, and a process lock."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from pathlib import Path

from agenticgen.io import object_hash, utc_now


class JobLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    task_id TEXT PRIMARY KEY, stage TEXT NOT NULL, case_id TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                    workdir TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0, started_at TEXT, completed_at TEXT,
                    result_path TEXT, reason TEXT, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,
                    time TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def add(self, task_id: str, stage: str, case_id: str, payload: dict, workdir: Path) -> None:
        digest = object_hash(payload)
        with self.connect() as db:
            old = db.execute("SELECT * FROM jobs WHERE task_id=?", (task_id,)).fetchone()
            if old:
                if old["payload_sha256"] != digest or old["workdir"] != str(workdir):
                    raise ValueError(f"Frozen task changed: {task_id}")
                return
            db.execute(
                "INSERT INTO jobs(task_id,stage,case_id,payload,payload_sha256,workdir,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (task_id, stage, case_id, json.dumps(payload), digest, str(workdir), utc_now()),
            )

    def get(self, task_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result

    def list(self, *, stage: str | None = None, status: str | None = None) -> list[dict]:
        where, args = [], []
        for key, value in (("stage", stage), ("status", status)):
            if value is not None:
                where.append(f"{key}=?")
                args.append(value)
        query = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "")
        with self.connect() as db:
            rows = [dict(r) for r in db.execute(query + " ORDER BY rowid", args)]
        for row in rows:
            row["payload"] = json.loads(row["payload"])
        return rows

    def claim(self, task_id: str) -> dict | None:
        now = utc_now()
        with self.connect() as db:
            changed = db.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1,started_at=?,"
                "completed_at=NULL,reason=NULL,updated_at=? WHERE task_id=? AND status='pending'",
                (now, now, task_id),
            ).rowcount
            if not changed:
                return None
            db.execute(
                "INSERT INTO events(task_id,time,kind,detail) VALUES(?,?,?,?)",
                (task_id, now, "claimed", "{}"),
            )
        return self.get(task_id)

    def finish(self, task_id: str, result_path: Path, *, skipped: bool = False) -> None:
        if not result_path.is_file():
            raise ValueError("A durable result is required before committing a job")
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,result_path=?,completed_at=?,updated_at=? "
                "WHERE task_id=? AND status='running'",
                (
                    "skipped" if skipped else "completed",
                    str(result_path),
                    utc_now(),
                    utc_now(),
                    task_id,
                ),
            )

    def fail(self, task_id: str, reason: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed',reason=?,completed_at=?,updated_at=? "
                "WHERE task_id=?",
                (reason, utc_now(), utc_now(), task_id),
            )
            db.execute(
                "INSERT INTO events(task_id,time,kind,detail) VALUES(?,?,?,?)",
                (task_id, utc_now(), "failed", json.dumps({"reason": reason})),
            )

    def requeue(self, task_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("Recovery requires an explicit reason")
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='pending',updated_at=?,reason=? WHERE task_id=?",
                (utc_now(), reason, task_id),
            )
            db.execute(
                "INSERT INTO events(task_id,time,kind,detail) VALUES(?,?,?,?)",
                (task_id, utc_now(), "requeued", json.dumps({"reason": reason})),
            )

    def supersede(self, task_id: str, reason: str) -> None:
        """Keep old receipts while removing a stale dependent from the active plan."""
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='superseded',reason=?,updated_at=? WHERE task_id=?",
                (reason, utc_now(), task_id),
            )
            db.execute(
                "INSERT INTO events(task_id,time,kind,detail) VALUES(?,?,?,?)",
                (task_id, utc_now(), "superseded", json.dumps({"reason": reason})),
            )

    def counts(self) -> dict:
        with self.connect() as db:
            rows = db.execute("SELECT stage,status,count(*) n FROM jobs GROUP BY stage,status")
            result = {}
            for row in rows:
                result.setdefault(row["stage"], {})[row["status"]] = row["n"]
        return result


@contextlib.contextmanager
def campaign_lock(path: Path):
    """The OS releases this lock after crashes; never infer liveness from stale PID text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if stream.tell() == 0:
        stream.write(b" ")
        stream.flush()
    stream.seek(0)
    try:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another controller holds the campaign lock") from error
        yield
    finally:
        stream.close()
