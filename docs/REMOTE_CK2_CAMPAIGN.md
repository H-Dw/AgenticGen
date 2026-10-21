# Remote CK2 original-pose campaign

The deployment helper creates a source snapshot in
`/workspace/AgenticGen_experiments/ck2_original_pose_20261020_high/code`.
It copies the current local `src/`, `scripts/`, `examples/`, `docs/`,
`pyproject.toml`, `README.md`, and `.gitmodules`; it excludes caches, outputs,
checkpoints, local environments, and credential files. Each included file has a
SHA-256 entry in `code/snapshot.manifest.json`. Local uncommitted source is
included, so the manifest identifies the tested bytes independently of Git HEAD.

The existing `/workspace/AgenticGen` checkout supplies three unchanged upstream
snapshots through `code/external/` symbolic links. Before deployment their
tracked files must be clean and their revisions must match the local submodules:

| Module | Verified revision |
| --- | --- |
| FLOWR.ROOT | `739a33c2aab74d074dd24605b9a2e34dc809ae9c` |
| SynCraft-Core | `557703e110b26a354afed2955ded5217837f6692` |
| SimpRetro | `5abbdd12d23bec3b0c524a3f5e59d160beac00f3` |

Frozen molecular and receptor inputs live separately in `inputs/`. The helper
does not upload or change these files. Campaign outcomes live in `results/`.
Deployment records and controller logs live in `environment/`.

## Authentication and runtime

Run the helper with a local Python that has Paramiko. It reads the SSH password
from `AGENTICGEN_SSH_PASSWORD`, or prompts with `getpass` in an interactive
terminal. Passwords are absent from arguments, uploaded files, and receipts.
`--ssh-key PATH` instead reuses an already authorized private key. SSH host keys
must already be trusted in the local known-hosts file; `--known-hosts PATH`
selects an additional trusted file.

Remote jobs start with `bash -ic`, which loads the existing root shell API-key
configuration without printing it, followed by activation of
`molsteer-flowr-dtk`. The interpreter is
`/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python`. Activation is necessary for
the existing DTK shared libraries. The helper installs no system or Python
packages and does not change Torch, DTK, or upstream source.

The current network path uses a temporary SSH reverse tunnel from remote
`127.0.0.1:17897` to the controller's existing `127.0.0.1:7897` proxy. The helper
exports `HTTP_PROXY` and `HTTPS_PROXY` for the job; it does not start, stop, or
reconfigure either proxy. Keep this tunnel alive until the LLM jobs finish.
The checkpoint is the existing
`/workspace/materials/flowr_root_v2.ckpt`, with SHA-256
`f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0`.

## Deploy and run

From the project root, using a Python with Paramiko:

```powershell
python scripts/deploy_ck2_campaign.py inventory

python scripts/deploy_ck2_campaign.py deploy `
  --foundation-template examples/ck2_original_pose_high/foundation.high.template.json `
  --campaign-template examples/ck2_original_pose_high/campaign.template.json `
  --receipt outputs/ck2_original_pose_20261020_high/environment/deployment.json

python scripts/deploy_ck2_campaign.py launch --phase baseline
python scripts/deploy_ck2_campaign.py status
```

The template renderer substitutes `@RUN_ROOT@`, `@CODE_ROOT@`, `@INPUT_ROOT@`,
`@OUTPUT_ROOT@`, `@ENVIRONMENT_ROOT@`, `@PYTHON@`, `@FLOWR_ROOT@`,
`@SYNCRAFT_ROOT@`, `@SIMP_RETRO_ROOT@`, `@CHECKPOINT@`, `@VINA_BINARY@`, and
`@PROXY@`. It preserves all scientific parameter values and `${ENV_VAR}`
references. It rejects unknown deployment placeholders and literal credential
fields. Rendered files are `foundation.high.json` and `campaign.json` at the run
root. Configurations become immutable when the campaign creates its results
manifest.

`deploy` starts no experiment. `launch` requires an explicit phase. `baseline`
measures the 20 original poses without proposing molecular edits. Launch
`design` after baseline completion and the intended authorization. The other
supported phases are `preflight`, `generation`, `evaluate`, and `all`.
`baseline-design` runs these two phases sequentially when both have been
authorized. A launch uses the standard campaign ledger and does not bypass its
resume, locking, or frozen-input checks.

Each detached controller has immutable `*.job.json` and `*.launch.json` records,
an updating `*.status.json`, and a `*.log` under `environment/jobs/`.
The launch receipt means that a controller was started. A completed controller
process can still contain failed campaign jobs; inspect `results/status.json`,
the ledger counts, and per-job failure records to judge experiment completion.

To deploy revised code after a phase has exited, repeat `deploy` with
`--replace-code`. The previous code is retained as `code.previous.TIMESTAMP`;
`inputs/` and `results/` are untouched. Deployment is rejected while a recorded
controller is active. The checkpoint and upstream snapshots are shared and
should remain fixed for the experiment.

When an independent preflight needs revised code while the design controller is
still running, create an immutable sibling release instead:

```powershell
python scripts/deploy_ck2_campaign.py release `
  --receipt outputs/ck2_original_pose_20261020_high/environment/preflight-release.json
```

This writes `releases/preflight_SNAPSHOT_PREFIX/` and a release receipt containing
the full snapshot SHA-256, package code fingerprint, source archive hash, and
absolute release path. It does not change the active `code/`, configuration,
inputs, results, or controller. An independent preflight process may import
`RELEASE_PATH/src`, use the existing immutable campaign configuration/results,
and call `Campaign.preflight()` without acquiring the design controller lock.
Keep its execution records separate and preserve failed checks.

After the design controller has exited and that release passes preflight,
promote the exact recorded source archive:

```powershell
python scripts/deploy_ck2_campaign.py promote --release FULL_SNAPSHOT_SHA256
```

Promotion does not rebuild a snapshot from the potentially changed local
working tree. It verifies and expands the retained archive, preserving identical
source bytes and the package code fingerprint. It refuses promotion while a
recorded controller is active and retains the previous `code/` directory.
This command starts no generation. The experiment owner decides when production
is ready to launch.

## Readiness scope recorded on 2026-10-20

The DTK interpreter, Vina 1.2.5, PLIP 3.0.0, Open Babel 3.1.0, Meeko 0.7.1,
and ligand preparation checks passed. An authenticated read-only DeepSeek model
catalog request returned HTTP 200 and listed `deepseek-flash`; no proposal was
made by this check. The legacy independent CPU SimpRetro worker is absent on
this server instance, so this readiness check does not certify retrosynthesis.
This campaign's configured retrospective evaluator is disabled.

An initial one-source affinity smoke test confirmed the native API can score an
unchanged input pose. It is informational. The campaign uses its current
singleton, fixed-seed affinity scoring implementation, and its baseline results
are the comparison values for the frozen experiment.

The first baseline controller completed successfully at `2026-10-19T18:49:14Z`
(2026-10-20 in Asia/Shanghai): all 20 source-geometry jobs and the 20-pose source
affinity job completed. Its deployed source snapshot was
`1e32277267440ae184e8df16ea9e825278afde6bfe9f413474388c39b1ddb959`.
The frame audit for `single_b003_final_025` found zero coordinate change between
the frozen candidate and its mapped source SDF. The score-only pose, complex,
and PLIP ligand coordinates differed from that source by at most 0.000755 Å,
consistent with PDB/PDBQT coordinate rounding. No coordinate alignment or
redocking was used. Machine-readable evidence is under
`environment/source-geometry-frame-audit.json`; original case artifacts remain
under `results/cases/single_b003_final_025/source_geometry/`.

## Recovery after the SSH outage on 2026-10-20

SSH checks failed between 00:56 and 02:26 UTC. When access returned at
02:56 UTC, the old production controller and GPU worker were absent and the
controller lock was free. The new container's PID 1 started at approximately
02:51:37 UTC. The available logs contain neither the old controller's exit code
nor an exception explaining the interruption; they do not establish its trigger.

The ledger contained 979 completed generation batches and one interrupted batch:
`generation:single_b017_final_021:gradient_ce_graph_pose_w0p25:batch_004`.
That batch had already saved its final tensors and all four particle records.
Recovery verified the frozen task and input hashes, original execution and code
fingerprints, final tensor fingerprint, particle indices, and saved SDFs. It
retained all four original particles and committed the durable generation result
without resampling. The original `runtime.json` remains unchanged; the new
`result.json` explicitly records its recovery provenance. Its missing affinity
receipt is handled by the existing affinity recovery queue during evaluation.

The ledger backup, original controller status, validation hashes, and restoration
receipt are retained under
`environment/recovery/interruption_20261020T030259Z/`. Production code,
checkpoint, frozen inputs, prompts, seeds, experiment grid, and guidance window
were unchanged. Previously completed design jobs remain completed and are reused.

The replacement controller is `all-1792465385963952567`, launched at
03:03:07 UTC with the same source snapshot
`71a893cdbab3ae7021baa7ad5b68839150eb219a617d7062ec252a9e5585058c`.
Monitor the latest `all-*.launch.json` together with its matching status and log;
the interrupted controller's historical status is not current process liveness.

## Recovery after the host reboot on 2026-10-21

SSH was unavailable in the checks from 00:26 through 06:26 UTC. Access returned
at approximately 07:06 UTC. Host uptime and boot ID establish a reboot at
01:10:38 UTC (09:10:38 Asia/Shanghai); the available evidence does not establish
what triggered that reboot or the previous controller's exit code. The old
controller and worker were absent and the operating-system lock was free.

The surviving ledger initially contained 4,669 completed generation batches and
one running row, while some saved execution directories were newer than their
ledger states. Recovery verified and committed the existing complete result for
`single_b007_final_028 / gradient_brier_graph_w0p25 / batch_019`. An initial
restart then exited with `FileExistsError` when it reached the already existing
`batch_020/execution_001` directory. This restart did not sample that batch.

A full scan of all remaining ledger rows found three additional complete
executions (`batch_020` through `batch_022`) and one incomplete execution
(`batch_023`). For the complete executions, recovery checked frozen input and
payload hashes, execution/code fingerprints, final tensor fingerprints, particle
indices and SDFs, and the final-pose affinity receipts before restoring their
ledger entries. Existing results, tensors, poses, and scores were retained byte
for byte. The original completed results were not regenerated.

For `batch_023`, the surviving runtime receipt reported generation completion,
but its referenced `raw_final.pt`, particle records, SDFs, and final result were
absent. The runtime flag alone was insufficient evidence of a durable result.
Recovery preserved `execution_001`, registered it as an infrastructure
interruption, and queued the identical frozen seed (`86940000`) in
`execution_002`. This is distinct from an invalid molecular decoding result and
does not add attempts beyond the planned scientific batch.

The original ledger/status/log backups, artifact hashes, and recovery receipts
are retained under both of these remote directories:

- `environment/recovery/interruption_20261021T071307Z/`
- `environment/recovery/reconciliation_20261021T072241Z/`

Reconciliation brought the ledger to 4,673 completed and 3,327 pending generation
batches. The replacement controller is `all-1792567365741013153`, launched at
07:22:47 UTC with the same source snapshot
`71a893cdbab3ae7021baa7ad5b68839150eb219a617d7062ec252a9e5585058c`.
Production Python code, checkpoint, frozen inputs, prompts, grid, and guidance
window were unchanged. All 20 completed design jobs are reused, with no repeated
LLM requests. The historical failed launch and both recovery records remain
available for audit.

At 07:24:31 UTC, the replacement controller was live with no failed ledger jobs.
Both the resumed `batch_023/execution_002` and the next, previously unstarted
`batch_024/execution_001` had completed: all eight particles decoded and obtained
native final-pose pIC50 scores. The controller had advanced to
`gradient_brier_graph_w1p0 / batch_000`. This verifies productive continuation,
beyond merely checking that a launch receipt exists.
