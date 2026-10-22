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

## Evaluation-only identity repair on 2026-10-22

All 8,000 generation batches completed at 02:18:50 UTC (10:18:50 Asia/Shanghai),
and the previously missing affinity batch completed through the standard recovery
queue. The controller then entered the CPU evaluation phase. At 04:38:56 UTC,
1,472 evaluation batches had completed and eight had failed while the four-worker
evaluation queue continued making progress.

The eight failures shared the same pre-Vina error:
`Input pose connectivity or specified stereochemistry differs`. Read-only probes
found identical mapped atom/bond identities, but a SMILES traversal-root chiral
atom with an implicit H could serialize from the SDF into the opposite parsed
stereo representation. A small synthetic root-stereocenter example reproduced
this behavior; it was not evidence of a Vina crash or grounds to alter a generated
pose. Some chemically symmetric centers additionally carried map-induced stereo.

`pose_graph_smiles` now normalizes an independent graph copy through explicit H
atoms before producing the mapped request, and PLIP metadata checks use the same
normalization. The supplied SDF and heavy-atom coordinates are untouched. The
original strict `load_input_pose` atom-map, element, charge, isotope, bond, and
specified-stereo checks remain unchanged. Tests confirm that opposite real
stereochemistry and changed mapped connectivity still fail.

Validation passed 224 local tests and Ruff. On the remote RDKit 2026.03.6
environment, all eight original failing inputs passed the unchanged strict check
with corrected serialization, a true opposite stereoisomer was rejected, and a
real previously failing input completed Vina score-only plus PLIP with an
unchanged input SDF hash. Validation receipts are retained in
`environment/evaluation_identity_v2_037a13119148/validation.json`.

The healthy production controller and its source snapshot/configuration remain
unchanged. An immutable evaluation-only overlay and separate maintenance process
were installed under:

- Code and file hashes: `environment/evaluation_identity_v2_037a13119148/`.
- Maintenance launch, log, status, and ledger backup:
  `environment/jobs/evaluation_identity_v2_037a13119148/`.
- Original main controller: `all-1792567365741013153`.
- Maintenance PID at launch: `26339` (use status plus liveness, not this PID alone).

The maintenance driver is `scripts/recover_ck2_evaluations.py`. It holds its own
OS lock, works only on failed evaluation jobs with the verified serialization
error, and re-evaluates the same four saved poses in a new execution directory.
It retains the original failure, logs, results, and protocol provenance. It
neither launches generation/LLM calls nor claims the healthy pending queue. It
checks for newly occurring instances of this same error while the old controller
finishes its queue; other errors or a second failure require investigation.

Every repaired result identifies the overlay files and geometry protocol hash.
The driver finalizes the manifest only after all 8,000 evaluation jobs complete,
the main controller exits successfully, and the global controller lock is free.
The heartbeat inspector also reads this maintenance process's status and errors.
Keep the user's current 150-minute inspection interval; the maintenance driver's
internal queue check is separate from the Codex heartbeat schedule.

The first corrected batch (`single_b004_final_027 / gradient_ce_graph_pose_w0p25 /
batch_015`) completed in `execution_002`. All four geometry results succeeded.
A read-only comparison at 04:57:22 UTC confirmed unchanged actual SDF hashes and
affinity parent receipts; all three particles that already had successful Vina
and PLIP measurements in the old execution retained identical scores and contact
sets. The local receipt is
`outputs/ck2_original_pose_20261020_high/environment/evaluation-recovery-audit-20261022T045722Z.json`.
This verifies an actual completed repair; remaining same-error jobs continue
through the maintenance queue without restarting the healthy controller.

### 2026-10-22 15:16 UTC: non-tetrahedral request serialization recovery

At 15:10 UTC the main controller was healthy with 5,506/8,000 evaluation
batches completed. The v2 maintenance process had recovered 80 batches, then
stopped at 13:20 UTC because a Meeko metadata identity exception was outside its
verified scope. Four failed evaluation batches remained; their original logs and
execution directories were retained.

The new case (`single_b007_final_028 / fragment_graph_pose / batch_024`) has a
non-tetrahedral sulfur center. Meeko retains an explicit H in its SMILES metadata.
The corrected v2 metadata comparison already recognized the same source graph,
but reusing that explicit-H representation as a request introduced an unmapped H
vertex and failed the unchanged heavy-atom identity check. The request serializer
now leaves non-tetrahedral hydrogens implicit while still normalizing ordinary
and tetrahedral centers. Metadata comparisons retain the explicit-H graph copy;
the strict geometry worker remains unchanged. No stereochemical constraint was
relaxed, and no coordinates, generated graph, affinity result, or grid changed.

An immutable replacement overlay is
`environment/evaluation_identity_v3_c6167b4ce685/`, with launch/status/log/ledger
backup under `environment/jobs/evaluation_identity_v3_c6167b4ce685/`. It was
launched at 15:16 UTC (initial PID 14373) only after confirming no maintenance
process remained alive. The old v2 overlay, failure status, logs, and all 80
repaired results are preserved. The new watcher recognizes only the two verified
identity exception signatures and retains the one-repair limit per evaluation.

Validation passed all 12 old/new real input identity probes, rejected genuinely
opposite specified stereo, and completed actual Vina score-only plus PLIP on the
new triggering pose (Vina -7.312 kcal/mol); its input SDF hash was unchanged.
Receipts are `evaluation-identity-20261022T151534Z-validation.json` and
`evaluation-identity-20261022T151616Z-launch.json` in the local experiment's
`environment/` directory. The full local suite passed 226 tests, including the
new non-tetrahedral request and root-stereo regressions; Ruff passed.

At 15:17:30 UTC the read-only recovery audit confirmed 83 completed repaired
batches, including the non-tetrahedral triggering batch. Every actual SDF hash
and affinity parent receipt was unchanged; all previously successful Vina scores
and PLIP contact sets remained identical. The audit receipt is
`evaluation-recovery-audit-20261022T151730Z.json`. The main controller had reached
5,584 completed evaluations, with no failed backlog and five running jobs
(four main evaluations plus the remaining maintenance evaluation). The v2
historical failed status is superseded by the healthy v3 maintenance process;
both status histories remain visible to the heartbeat inspector.

At 15:18:19 UTC all four newly blocked batches had completed recovery; the ledger
showed 5,593/8,000 evaluations completed, four main evaluations running, and no
failed jobs. The v3 watcher returned to watching for future matching errors.
