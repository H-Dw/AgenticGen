"""One serial GPU worker reuses immutable weights across independent microbatches."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import uuid
from pathlib import Path


class FlowrSession:
    def __init__(self, python: Path, env: dict, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"gpu_worker_{uuid.uuid4().hex[:12]}.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(python), "-u", "-m", "agenticgen.workers.flowr_campaign", "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            env=env,
            bufsize=1,
        )
        self.messages: queue.Queue = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.messages.put(json.loads(line))
            except ValueError:
                # Native informational output is preserved, never mistaken for a reply.
                self.log.write(line)
                self.log.flush()
        self.messages.put({"status": "worker_exited", "returncode": self.process.poll()})

    def run(self, task_path: Path, timeout: float) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError(f"GPU worker exited; see {self.log_path}")
        request_id = uuid.uuid4().hex
        self.process.stdin.write(
            json.dumps({"task_path": str(task_path), "request_id": request_id}) + "\n"
        )
        self.process.stdin.flush()
        try:
            response = self.messages.get(timeout=timeout)
        except queue.Empty as error:
            self.close()
            raise TimeoutError(f"GPU worker timed out; see {self.log_path}") from error
        if response.get("request_id") != request_id or response.get("status") != "completed":
            raise RuntimeError(f"GPU task failed: {response}; see {self.log_path}")
        return response

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=1)
        self.log.close()
