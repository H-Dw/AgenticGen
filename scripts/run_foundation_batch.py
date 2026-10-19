"""Run independent cases with bounded concurrency and full-cohort RAG exclusions.

Each subprocess uses the ordinary AgenticGen runner. Case subsets share the
frozen input/config but never share a mutable provider or chemistry runtime.
Collecting existing receipts does not repeat any model or evaluator call.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agenticgen.base_config import FoundationConfig, load_foundation
from agenticgen.base_reporting import foundation_summary
from agenticgen.dataset import load_dataset
from agenticgen.io import read_json, sha256, utc_now, write_json
from agenticgen.reporting import usage_from_receipts


def collect(output: Path) -> dict:
    manifest = read_json(output / "manifest.json")
    results = []
    child_manifests = []
    for case_id in manifest["case_ids"]:
        folder = output / "runs" / case_id
        result = folder / "cases" / case_id / "result.json"
        if result.is_file():
            results.append(read_json(result))
        child = folder / "manifest.json"
        if child.is_file():
            child_manifests.append({"case_id": case_id, "sha256": sha256(child)})
    if not results:
        return {"completed_cases": 0, "expected_cases": len(manifest["case_ids"])}
    template = read_json(output / "runs" / results[0]["case_id"] / "manifest.json")
    settings = FoundationConfig.model_validate(template["effective_config"])
    summary = foundation_summary(results, settings, output)
    usage = [usage_from_receipts(output / "runs" / c) for c in manifest["case_ids"]]
    summary["usage"] = {
        key: sum(u[key] for u in usage)
        for key in (
            "successful_api_calls",
            "http_attempts_in_success_receipts",
            "reported_prompt_tokens",
            "reported_completion_tokens",
        )
    }
    summary["usage"]["unreported_usage"] = "Not inferred without provider usage receipts"
    summary.update(
        batch_run=True,
        expected_case_count=len(manifest["case_ids"]),
        completed_case_count=len(results),
        incomplete_case_ids=[
            c for c in manifest["case_ids"] if c not in {r["case_id"] for r in results}
        ],
        child_manifests=child_manifests,
    )
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-no-api-errors", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.collect_only:
        print(collect(output))
        return 0
    if args.config is None:
        parser.error("--config is required for a new run")
    config_path = args.config.resolve()
    config = load_foundation(config_path)
    dataset_path = config.resolve(config.value.dataset)
    dataset = load_dataset(dataset_path)
    proposed_manifest = {
        "schema_version": "agenticgen.foundation-batch.v1",
        "status": "running",
        "started_at": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "dataset_sha256": sha256(dataset_path),
        "case_ids": [c.case_id for c in dataset.cases],
        "workers": args.workers,
        "rag_exclusions": "Every child builds its agent against the full frozen cohort",
        "evaluation_cache_scope": "Within each case; no cross-case evaluator cache",
        "jobs": [],
    }
    if args.resume:
        manifest = read_json(output / "manifest.json")
        for key in ("config_sha256", "dataset_sha256", "case_ids"):
            if manifest[key] != proposed_manifest[key]:
                raise ValueError("Resume configuration/cohort differs from frozen batch")
        pending = []
        manifest.setdefault("attempt_history", [])
        for case_id in manifest["case_ids"]:
            folder = output / "runs" / case_id
            if not (folder / "manifest.json").exists():
                pending.append(case_id)
                continue
            child = read_json(folder / "manifest.json")
            if child["status"] == "running":
                raise ValueError("An existing case is still running; do not duplicate it")
            if child["status"] == "completed":
                continue
            receipts = list(folder.glob("cases/*/calls/*.receipt.json"))
            if not args.retry_no_api_errors or receipts:
                raise ValueError("Failed case has API receipts or explicit retry was not requested")
            target = output / "attempts" / (case_id + ".before_interface_repair")
            # Both targets are fixed children of this batch; no recursive removal.
            if not folder.resolve().is_relative_to(output) or not target.resolve().is_relative_to(
                output
            ):
                raise ValueError("Attempt archive escaped batch directory")
            target.parent.mkdir(exist_ok=True)
            previous_hash = sha256(folder / "manifest.json")
            folder.rename(target)
            manifest["attempt_history"].append(
                {
                    "case_id": case_id,
                    "path": str(target.relative_to(output)),
                    "manifest_sha256": previous_hash,
                    "api_receipts": 0,
                }
            )
            pending.append(case_id)
        manifest.update(status="running", resumed_at=utc_now(), workers=args.workers)
        manifest["jobs"] = [
            {"case_id": c, "exit_code": 0, "preserved_completed_run": True}
            for c in manifest["case_ids"]
            if c not in pending
        ]
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "logs").mkdir()
        manifest = proposed_manifest
        pending = manifest["case_ids"]
    write_json(output / "manifest.json", manifest)

    def execute(case_id):
        with (output / "logs" / f"{case_id}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agenticgen",
                    "run",
                    "--config",
                    str(config_path),
                    "--output",
                    str(output / "runs" / case_id),
                    "--case-id",
                    case_id,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {"case_id": case_id, "exit_code": process.returncode, "ended_at": utc_now()}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, case_id): case_id for case_id in pending}
        for future in as_completed(futures):
            job = future.result()
            manifest["jobs"].append(job)
            write_json(output / "manifest.json", manifest)
            summary = collect(output)
            print({"finished": len(manifest["jobs"]), "total": len(futures), **job}, flush=True)
    summary = collect(output)
    manifest.update(
        status="completed" if all(j["exit_code"] == 0 for j in manifest["jobs"]) else "partial",
        ended_at=utc_now(),
        summary_sha256=sha256(output / "summary.json"),
    )
    write_json(output / "manifest.json", manifest)
    print(summary, flush=True)
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
