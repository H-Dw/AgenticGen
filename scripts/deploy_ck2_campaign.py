"""Deploy a source-only campaign snapshot and explicitly launch selected phases.

Requires Paramiko on the controlling machine. Passwords are read only from
AGENTICGEN_SSH_PASSWORD or getpass; --ssh-key can reuse an existing trusted key.
This helper does not install packages or change the pinned upstream checkout.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import gzip
import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

SOURCE_DIRECTORIES = ("src", "scripts", "examples", "docs")
SOURCE_FILES = ("pyproject.toml", "README.md", ".gitmodules")
SUBMODULES = ("FLOWR.ROOT", "SynCraft-Core", "SimpRetro")
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "outputs",
    "checkpoints",
    "node_modules",
    "environment",
    "secrets",
    "credentials",
}
SECRET_NAMES = {
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "credentials.toml",
    "api_keys.json",
    "token.json",
    "tokens.json",
    "known_hosts",
    "authorized_keys",
}
DEFAULT_ROOT = "/workspace/AgenticGen_experiments/ck2_original_pose_20261020_high"
DEFAULT_PYTHON = "/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def included(path: Path, repo: Path) -> bool:
    parts = path.relative_to(repo).parts
    return not (
        any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in parts)
        or path.name in SECRET_NAMES
        or path.name.startswith(".env")
        or path.suffix.lower() in {".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx"}
    )


def source_files(repo: Path) -> list[Path]:
    """Allow only the requested project trees, never following local symlinks."""
    selected = [repo / name for name in SOURCE_FILES if (repo / name).is_file()]
    for name in SOURCE_DIRECTORIES:
        root = repo / name
        if root.exists():
            selected.extend(path for path in root.rglob("*") if path.is_file())
    result = []
    for path in sorted(set(selected)):
        if not included(path, repo):
            continue
        if path.is_symlink() or any(
            parent.is_symlink() for parent in path.parents if parent != repo
        ):
            raise ValueError(f"Source snapshot cannot include symlinks: {path.relative_to(repo)}")
        if not path.resolve().is_relative_to(repo.resolve()):
            raise ValueError("Source file escapes the repository")
        result.append(path)
    if not (repo / "src/agenticgen/campaign.py").is_file():
        raise ValueError("Campaign source module is missing")
    return result


def build_snapshot(repo: Path, archive: Path) -> dict:
    entries = {}
    contents = {}
    for path in source_files(repo):
        name, data = path.relative_to(repo).as_posix(), path.read_bytes()
        contents[name] = data
        entries[name] = {"sha256": digest(data), "bytes": len(data)}
    revisions = {
        name: git_output(repo / "external" / name, "rev-parse", "HEAD") for name in SUBMODULES
    }
    identity = {"files": entries, "external_revisions": revisions}
    package_files = {
        name.removeprefix("src/agenticgen/"): entry["sha256"]
        for name, entry in entries.items()
        if name.startswith("src/agenticgen/") and name.endswith(".py")
    }
    manifest = {
        "schema_version": "agenticgen.campaign-source-snapshot.v1",
        "snapshot_sha256": digest(canonical_json(identity)),
        "created_at": utc_now(),
        "local_head": git_output(repo, "rev-parse", "HEAD"),
        "agenticgen_code_sha256": digest(
            json.dumps(package_files, sort_keys=True, separators=(",", ":")).encode()
        ),
        **identity,
    }
    contents["snapshot.manifest.json"] = canonical_json(manifest)
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz,
    ):
        with tarfile.open(fileobj=gz, mode="w") as bundle:
            for name, data in sorted(contents.items()):
                item = tarfile.TarInfo(name)
                item.size, item.mtime, item.mode = len(data), 0, 0o644
                bundle.addfile(item, io.BytesIO(data))
    manifest["archive_sha256"] = digest(archive.read_bytes())
    return manifest


def render_template(value: object, substitutions: dict[str, str]) -> object:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key.lower() in {
                "password",
                "api_key",
                "access_token",
                "secret_key",
                "authorization",
            }:
                if item not in (None, ""):
                    raise ValueError(f"Literal credential field is prohibited: {key}")
            result[key] = render_template(item, substitutions)
        return result
    if isinstance(value, list):
        return [render_template(item, substitutions) for item in value]
    if isinstance(value, str):
        for key, replacement in substitutions.items():
            value = value.replace(f"@{key}@", replacement)
        unresolved = re.findall(r"@[A-Z][A-Z0-9_]*@", value)
        if unresolved:
            raise ValueError(f"Unknown deployment placeholders: {', '.join(unresolved)}")
    return value


def remote_paths(args) -> dict[str, str]:
    root = args.remote_root.rstrip("/")
    path = PurePosixPath(root)
    if not path.is_absolute() or ".." in path.parts or root == args.upstream.rstrip("/"):
        raise ValueError("Use a distinct absolute experiment root without '..'")
    code = f"{root}/code"
    return {
        "RUN_ROOT": root,
        "CODE_ROOT": code,
        "INPUT_ROOT": f"{root}/inputs",
        "OUTPUT_ROOT": args.campaign_output or f"{root}/results",
        "ENVIRONMENT_ROOT": f"{root}/environment",
        "PYTHON": args.python,
        "FLOWR_ROOT": f"{code}/external/FLOWR.ROOT",
        "SYNCRAFT_ROOT": f"{code}/external/SynCraft-Core",
        "SIMP_RETRO_ROOT": f"{code}/external/SimpRetro",
        "CHECKPOINT": args.checkpoint,
        "VINA_BINARY": args.vina,
        "PROXY": args.proxy,
    }


class Remote:
    def __init__(self, args):
        try:
            import paramiko
        except ImportError as error:
            raise RuntimeError("Install Paramiko in the controlling Python environment") from error
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        if args.known_hosts:
            self.client.load_host_keys(str(args.known_hosts))
        self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        password = None
        if not args.ssh_key:
            password = os.environ.get("AGENTICGEN_SSH_PASSWORD")
            if not password:
                if not sys.stdin.isatty():
                    raise RuntimeError("Set AGENTICGEN_SSH_PASSWORD or use --ssh-key")
                password = getpass.getpass("SSH password (not saved): ")
        try:
            self.client.connect(
                args.host,
                port=args.port,
                username=args.user,
                password=password,
                key_filename=str(args.ssh_key) if args.ssh_key else None,
                look_for_keys=False,
                allow_agent=False,
                timeout=30,
            )
        except Exception as error:
            self.client.close()
            raise RuntimeError(f"SSH connection failed ({type(error).__name__})") from None

    def close(self):
        self.client.close()

    def python(self, source: str, payload: dict | None = None) -> dict:
        # JSON travels on stdin, so shell quoting cannot reinterpret paths or values.
        import base64

        code = base64.b64encode(source.encode()).decode()
        command = "python3 -c " + shlex.quote(f"import base64;exec(base64.b64decode('{code}'))")
        stdin, stdout, stderr = self.client.exec_command(command, timeout=60)
        stdin.write(json.dumps(payload or {}))
        stdin.channel.shutdown_write()
        output, error_output = stdout.read().decode(), stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        if code:
            # Remote helpers contain no credentials; do not echo arbitrary remote output.
            raise RuntimeError(f"Remote deployment helper exited {code}: {error_output[-1200:]}")
        return json.loads(output)

    def upload(self, source: Path, target: str):
        with self.client.open_sftp() as sftp:
            parent = str(PurePosixPath(target).parent)
            accumulated = "/"
            for component in PurePosixPath(parent).parts[1:]:
                accumulated = posixpath.join(accumulated, component)
                try:
                    sftp.stat(accumulated)
                except OSError:
                    sftp.mkdir(accumulated)
            pending = target + ".partial"
            sftp.put(str(source), pending)
            sftp.posix_rename(pending, target)


INVENTORY_SOURCE = r"""
import json, pathlib, subprocess, sys
p=json.load(sys.stdin); root=pathlib.Path(p["upstream"])
def git(path,*args):
    return subprocess.check_output(["git","-C",str(path),*args],text=True).strip()
modules={}
for name in p["submodules"]:
    source=root/"external"/name
    modules[name]={"path":str(source),"revision":git(source,"rev-parse","HEAD"),
                   "tracked_clean":not bool(git(source,"status","--porcelain","--untracked-files=no"))}
print(json.dumps({"upstream":str(root),"revision":git(root,"rev-parse","HEAD"),
                  "clean":not bool(git(root,"status","--porcelain")),"submodules":modules,
                  "python_present":pathlib.Path(p["python"]).is_file(),
                  "checkpoint_present":pathlib.Path(p["checkpoint"]).is_file()}))
"""


INSTALL_SOURCE = r"""
import hashlib,json,os,pathlib,sys,tarfile,time
p=json.load(sys.stdin); root=pathlib.Path(p["root"]); target=root/"code"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def active():
    for path in (root/"environment/jobs").glob("*.launch.json"):
        job=json.loads(path.read_text()); pid=job["pid"]
        state=path.with_name(path.name.replace(".launch.json",".status.json"))
        if state.exists() and json.loads(state.read_text()).get("status") in {"completed","failed"}: continue
        try: os.kill(pid,0)
        except ProcessLookupError: continue
        return True
    return False
if active(): raise RuntimeError("A campaign controller is active; source and configs are immutable")
if sha(pathlib.Path(p["archive"])) != p["archive_sha256"]: raise RuntimeError("Archive hash differs")
config_data={name:json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n" for name,value in p["configs"].items()}
for name,data in config_data.items():
    existing=root/name
    if existing.exists() and existing.read_text()!=data and (pathlib.Path(p["output"])/"manifest.json").exists():
        raise RuntimeError("A prepared campaign cannot change its frozen configuration")
if target.is_symlink(): raise RuntimeError("The code destination cannot be a symbolic link")
if target.exists():
    old=json.loads((target/"snapshot.manifest.json").read_text())
    if old["snapshot_sha256"] != p["snapshot_sha256"] and not p["replace_code"]:
        raise RuntimeError("Code already exists with another snapshot; explicit --replace-code is required")
staging=root/("code.staging."+p["snapshot_sha256"][:16])
if staging.exists(): raise RuntimeError("Staging directory already exists; inspect the interrupted deployment")
staging.mkdir(parents=True)
with tarfile.open(p["archive"],"r:gz") as bundle:
    for item in bundle.getmembers():
        member=pathlib.PurePosixPath(item.name)
        if not item.isfile() or member.is_absolute() or ".." in member.parts: raise RuntimeError("Unsafe archive entry")
        destination=staging.joinpath(*member.parts); destination.parent.mkdir(parents=True,exist_ok=True)
        with bundle.extractfile(item) as source: destination.write_bytes(source.read())
manifest=json.loads((staging/"snapshot.manifest.json").read_text())
if manifest["snapshot_sha256"] != p["snapshot_sha256"]: raise RuntimeError("Snapshot identity differs")
for name,entry in manifest["files"].items():
    if sha(staging/name) != entry["sha256"]: raise RuntimeError("Source file hash differs: "+name)
(staging/"external").mkdir()
for name,source in p["external_paths"].items(): (staging/"external"/name).symlink_to(source,target_is_directory=True)
previous=None
if target.exists():
    previous=root/("code.previous."+str(time.time_ns())); target.rename(previous)
staging.rename(target)
for name,data in config_data.items():
    path=root/name; path.write_text(data)
receipt={"schema_version":"agenticgen.campaign-deployment.v1","snapshot_sha256":p["snapshot_sha256"],
         "archive_sha256":p["archive_sha256"],"code":str(target),"previous_code":str(previous) if previous else None,
         "agenticgen_code_sha256":manifest.get("agenticgen_code_sha256"),
         "external_paths":p["external_paths"],"external_revisions":manifest["external_revisions"],
         "configuration_hashes":{name:sha(root/name) for name in config_data},"file_count":len(manifest["files"]),
         "formal_jobs_launched":0}
(root/"environment/deployment.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
"""


RELEASE_SOURCE = r"""
import hashlib,json,pathlib,sys,tarfile
p=json.load(sys.stdin); root=pathlib.Path(p["root"])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if sha(pathlib.Path(p["archive"])) != p["archive_sha256"]: raise RuntimeError("Archive hash differs")
releases=root/"releases"; releases.mkdir(exist_ok=True)
target=releases/("preflight_"+p["snapshot_sha256"][:16])
receipts=root/"environment/releases"; receipts.mkdir(exist_ok=True)
receipt_path=receipts/(p["snapshot_sha256"]+".json")
if target.exists():
    existing=json.loads((target/"snapshot.manifest.json").read_text())
    if existing["snapshot_sha256"] != p["snapshot_sha256"]: raise RuntimeError("Release identity differs")
    for name,entry in existing["files"].items():
        if sha(target/name) != entry["sha256"]: raise RuntimeError("Existing immutable release changed")
    if not receipt_path.is_file(): raise RuntimeError("Existing release has no receipt; inspect it")
    print(receipt_path.read_text()); sys.exit(0)
staging=target.with_name(target.name+".staging")
if staging.exists(): raise RuntimeError("Release staging already exists; inspect interrupted deployment")
staging.mkdir()
with tarfile.open(p["archive"],"r:gz") as bundle:
    for item in bundle.getmembers():
        member=pathlib.PurePosixPath(item.name)
        if not item.isfile() or member.is_absolute() or ".." in member.parts: raise RuntimeError("Unsafe archive entry")
        destination=staging.joinpath(*member.parts); destination.parent.mkdir(parents=True,exist_ok=True)
        with bundle.extractfile(item) as source: destination.write_bytes(source.read())
manifest=json.loads((staging/"snapshot.manifest.json").read_text())
if manifest["snapshot_sha256"] != p["snapshot_sha256"]: raise RuntimeError("Snapshot identity differs")
for name,entry in manifest["files"].items():
    if sha(staging/name) != entry["sha256"]: raise RuntimeError("Source hash differs: "+name)
(staging/"external").mkdir()
for name,source in p["external_paths"].items(): (staging/"external"/name).symlink_to(source,target_is_directory=True)
staging.rename(target)
receipt={"schema_version":"agenticgen.campaign-release.v1","snapshot_sha256":p["snapshot_sha256"],
         "agenticgen_code_sha256":manifest.get("agenticgen_code_sha256"),"archive":p["archive"],
         "archive_sha256":p["archive_sha256"],"code":str(target),"external_paths":p["external_paths"],
         "external_revisions":manifest["external_revisions"],"file_count":len(manifest["files"]),
         "active_code_unchanged":True,"formal_jobs_launched":0}
receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
"""


READ_RELEASE_SOURCE = r"""
import json,pathlib,sys
p=json.load(sys.stdin); root=pathlib.Path(p["root"])
receipt=json.loads((root/"environment/releases"/(p["release"]+".json")).read_text())
if receipt["snapshot_sha256"] != p["release"]: raise RuntimeError("Release identity differs")
receipt["configs"]={name:json.loads((root/name).read_text()) for name in ("foundation.high.json","campaign.json")}
print(json.dumps(receipt))
"""


JOB_RUNNER = r"""
import datetime,json,os,pathlib,subprocess,sys,traceback
job_path=pathlib.Path(sys.argv[1]); job=json.loads(job_path.read_text())
state_path=job_path.with_name(job_path.name.replace(".job.json",".status.json"))
state={"job_id":job["job_id"],"pid":os.getpid(),"status":"running","phases":[],
       "started_at":datetime.datetime.now(datetime.UTC).isoformat(),"completion_calls_not_inferred":True}
def save(): state_path.write_text(json.dumps(state,indent=2)+"\n")
save(); code=1
try:
    if not os.environ.get("DEEPSEEK_API_KEY"): raise RuntimeError("DEEPSEEK_API_KEY is absent in the interactive shell")
    for command in job["commands"]:
        state["current_phase"]=command["phase"]; save()
        code=subprocess.call(command["argv"],cwd=job["code"])
        state["phases"].append({"phase":command["phase"],"exit_code":code}); save()
        if code: break
    state["status"]="completed" if code==0 else "failed"
except BaseException as error:
    state["status"]="failed"; state["error_type"]=type(error).__name__
    traceback.print_exc()
finally:
    state["finished_at"]=datetime.datetime.now(datetime.UTC).isoformat(); state["exit_code"]=code; save()
sys.exit(code)
"""


LAUNCH_SOURCE = r"""
import json,os,pathlib,subprocess,sys,time
p=json.load(sys.stdin); root=pathlib.Path(p["root"]); jobs=root/"environment/jobs"; jobs.mkdir(exist_ok=True)
for relative in ("code/snapshot.manifest.json","campaign.json","foundation.high.json"):
    if not (root/relative).is_file(): raise RuntimeError("Deployment is incomplete: "+relative)
for prior in jobs.glob("*.launch.json"):
    receipt=json.loads(prior.read_text()); status=prior.with_name(prior.name.replace(".launch.json",".status.json"))
    if status.exists() and json.loads(status.read_text()).get("status") in {"completed","failed"}: continue
    try: os.kill(receipt["pid"],0)
    except ProcessLookupError: continue
    raise RuntimeError("Another campaign controller is active")
job_id=p["phase"]+"-"+str(time.time_ns()); base=jobs/job_id
job={"job_id":job_id,"code":str(root/"code"),"commands":p["commands"]}
job_path=pathlib.Path(str(base)+".job.json"); job_path.write_text(json.dumps(job,indent=2)+"\n")
runner=root/"environment/run_campaign_job.py"; runner.write_text(p["runner"])
log_path=pathlib.Path(str(base)+".log")
with log_path.open("xb") as log:
    process=subprocess.Popen(["bash","-ic",p["shell"]+" "+__import__("shlex").quote(str(job_path))],
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={"schema_version":"agenticgen.campaign-launch.v1","job_id":job_id,"phase":p["phase"],
         "pid":process.pid,"log":str(log_path),"job":str(job_path),"commands":p["commands"],
         "source_snapshot_sha256":json.loads((root/"code/snapshot.manifest.json").read_text())["snapshot_sha256"]}
pathlib.Path(str(base)+".launch.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
"""


STATUS_SOURCE = r"""
import json,os,pathlib,sys
p=json.load(sys.stdin); root=pathlib.Path(p["root"]); jobs=[]
for path in sorted((root/"environment/jobs").glob("*.launch.json")):
    job=json.loads(path.read_text()); status=path.with_name(path.name.replace(".launch.json",".status.json"))
    if status.exists(): job["state"]=json.loads(status.read_text())
    try: os.kill(job["pid"],0); job["process_present"]=True
    except ProcessLookupError: job["process_present"]=False
    jobs.append(job)
result={"jobs":jobs}
for name in ("status.json","manifest.json"):
    path=pathlib.Path(p["output"])/name
    if path.exists(): result[name]=json.loads(path.read_text())
print(json.dumps(result))
"""


def inventory(remote: Remote, args) -> dict:
    return remote.python(
        INVENTORY_SOURCE,
        {
            "upstream": args.upstream,
            "python": args.python,
            "checkpoint": args.checkpoint,
            "submodules": SUBMODULES,
        },
    )


def deploy(remote: Remote, args, paths: dict[str, str]) -> dict:
    for name in ("foundation_template", "campaign_template"):
        if getattr(args, name) is None:
            raise ValueError("Deployment requires both JSON configuration templates")
    configs = {}
    for name, source in (
        ("foundation.high.json", args.foundation_template),
        ("campaign.json", args.campaign_template),
    ):
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Configuration templates must be JSON objects")
        configs[name] = render_template(value, paths)
    payload = upload_snapshot(remote, args, paths)
    return remote.python(
        INSTALL_SOURCE,
        {
            **payload,
            "configs": configs,
            "replace_code": args.replace_code,
            "output": paths["OUTPUT_ROOT"],
        },
    )


def upload_snapshot(remote: Remote, args, paths: dict[str, str]) -> dict:
    checked = inventory(remote, args)
    if not checked["python_present"] or not checked["checkpoint_present"]:
        raise ValueError("The existing interpreter or checkpoint is unavailable")
    with tempfile.TemporaryDirectory(prefix="agenticgen-source-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        manifest = build_snapshot(args.repo.resolve(), archive)
        external = {}
        for name, expected in manifest["external_revisions"].items():
            actual = checked["submodules"][name]
            if actual["revision"] != expected or not actual["tracked_clean"]:
                raise ValueError(f"Remote upstream differs from pinned source: {name}")
            external[name] = actual["path"]
        remote_archive = (
            f"{paths['ENVIRONMENT_ROOT']}/source-{manifest['snapshot_sha256'][:16]}"
            f"-{manifest['archive_sha256'][:12]}.tar.gz"
        )
        remote.upload(archive, remote_archive)
        return {
            "root": paths["RUN_ROOT"],
            "archive": remote_archive,
            "archive_sha256": manifest["archive_sha256"],
            "snapshot_sha256": manifest["snapshot_sha256"],
            "external_paths": external,
        }


def promote(remote: Remote, args, paths: dict[str, str]) -> dict:
    if not args.release or not re.fullmatch(r"[a-f0-9]{64}", args.release):
        raise ValueError("Promotion requires the full --release snapshot SHA-256")
    release = remote.python(
        READ_RELEASE_SOURCE, {"root": paths["RUN_ROOT"], "release": args.release}
    )
    return remote.python(
        INSTALL_SOURCE,
        {
            "root": paths["RUN_ROOT"],
            "archive": release["archive"],
            "archive_sha256": release["archive_sha256"],
            "snapshot_sha256": release["snapshot_sha256"],
            "external_paths": release["external_paths"],
            "configs": release["configs"],
            "replace_code": True,
            "output": paths["OUTPUT_ROOT"],
        },
    )


def launch_payload(args, paths: dict[str, str]) -> dict:
    phases = ("baseline", "design") if args.phase == "baseline-design" else (args.phase,)
    commands = [
        {
            "phase": phase,
            "argv": [
                args.python,
                "-m",
                "agenticgen.campaign",
                "run",
                "--config",
                f"{paths['RUN_ROOT']}/campaign.json",
                "--output",
                paths["OUTPUT_ROOT"],
                "--phase",
                phase,
            ],
        }
        for phase in phases
    ]
    exports = {
        "PYTHONPATH": f"{paths['CODE_ROOT']}/src",
        "PYTHONUNBUFFERED": "1",
        "AGENTICGEN_PYTHON": args.python,
        "FLOWR_PYTHON": args.python,
        "FLOWR_CHECKPOINT": args.checkpoint,
        "VINA_BINARY": args.vina,
        "HTTP_PROXY": args.proxy,
        "HTTPS_PROXY": args.proxy,
        "OMP_NUM_THREADS": "1",
    }
    shell = (
        "source /opt/miniforge3/etc/profile.d/conda.sh && "
        f"conda activate {shlex.quote(args.conda_env)} && "
        + " && ".join(f"export {key}={shlex.quote(value)}" for key, value in exports.items())
        + f" && exec {shlex.quote(args.python)} "
        + shlex.quote(f"{paths['ENVIRONMENT_ROOT']}/run_campaign_job.py")
    )
    return {
        "root": paths["RUN_ROOT"],
        "phase": args.phase,
        "commands": commands,
        "shell": shell,
        "runner": JOB_RUNNER,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "command", choices=("inventory", "deploy", "release", "promote", "launch", "status")
    )
    root.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    root.add_argument("--host", default="ksai.scnet.cn")
    root.add_argument("--port", type=int, default=10399)
    root.add_argument("--user", default="root")
    root.add_argument("--ssh-key", type=Path)
    root.add_argument("--known-hosts", type=Path)
    root.add_argument("--remote-root", default=DEFAULT_ROOT)
    root.add_argument("--upstream", default="/workspace/AgenticGen")
    root.add_argument("--python", default=DEFAULT_PYTHON)
    root.add_argument("--conda-env", default="molsteer-flowr-dtk")
    root.add_argument("--checkpoint", default="/workspace/materials/flowr_root_v2.ckpt")
    root.add_argument("--vina", default="/usr/bin/vina")
    root.add_argument("--proxy", default="http://127.0.0.1:17897")
    root.add_argument("--campaign-output")
    root.add_argument("--foundation-template", type=Path)
    root.add_argument("--campaign-template", type=Path)
    root.add_argument("--replace-code", action="store_true")
    root.add_argument("--release", help="Full immutable release snapshot SHA-256 for promotion")
    root.add_argument(
        "--phase",
        choices=(
            "baseline-design",
            "baseline",
            "design",
            "preflight",
            "generation",
            "evaluate",
            "all",
        ),
    )
    root.add_argument("--receipt", type=Path, help="Optional local nonsecret JSON receipt")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "launch" and not args.phase:
        raise ValueError("Launch requires an explicit --phase")
    paths = remote_paths(args)
    with contextlib.closing(Remote(args)) as remote:
        if args.command == "inventory":
            value = inventory(remote, args)
        elif args.command == "deploy":
            value = deploy(remote, args, paths)
        elif args.command == "release":
            value = remote.python(RELEASE_SOURCE, upload_snapshot(remote, args, paths))
        elif args.command == "promote":
            value = promote(remote, args, paths)
        elif args.command == "launch":
            value = remote.python(LAUNCH_SOURCE, launch_payload(args, paths))
        else:
            value = remote.python(
                STATUS_SOURCE, {"root": paths["RUN_ROOT"], "output": paths["OUTPUT_ROOT"]}
            )
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(canonical_json(value))
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f"Deployment failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
