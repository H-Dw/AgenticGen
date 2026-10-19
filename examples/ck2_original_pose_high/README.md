# CK2 original-pose high-reasoning campaign

This profile uses the frozen 20 distinct molecular graphs in
`examples/ck2_affinity_top20/cases.json`, retaining the original Steer pose of
each graph's highest historical pIC50 particle. It does not apply a new SA,
PoseBusters, strain, or docking filter. The source selection remains fixed.

`foundation.high.template.json` retains the existing SynCraft retrieval,
interaction-aware prompt, host guidance, parser, and permissive reconstruction.
It enables DeepSeek `high` reasoning. It uses the actual original Steer pose for
PLIP conditions and Vina `score_only`, without embedding or redocking.

Only a parsable, changed graph proceeds to FLOWR. An unparsable proposal permits
five additional proposals with the same prompt, with all attempts recorded.
STOP and unchanged outputs are separate outcomes. Network retries are recorded
separately and do not consume chemical proposals. No SA improvement or interaction
retention gate selects which parsable changed products are sampled.

The templates need runtime paths. Use the source-snapshot deployment helper in
[the remote guide](../../docs/REMOTE_CK2_CAMPAIGN.md). It renders configurations
without changing scientific parameters or putting credentials in files.

There are 16 arms per usable edited graph: three hard inpainting conditions,
one unguided control, and twelve soft guidance conditions (CE/Brier × graph or
graph plus original-interaction coordinates × weights 0.25/1/4). Each applicable
arm attempts 100 particles in 25 microbatches of four, with paired seeds. A
persistent worker loads the checkpoint once, while resetting all RNGs per task.

The guidance interval remains **0.1 ≤ t < 0.9** for 100 linear steps (80 updates).
Read [the protocol](../../docs/CK2_ORIGINAL_POSE_CAMPAIGN.md) for region definitions,
gradient operators, applicability rules, evaluation, and restart semantics.

All final tensors precede decoding. Failed decoding and duplicate structures
remain in the attempted denominator; there is no hidden replenishment. Final
structures use common fixed-seed, singleton native affinity-head rescoring and
Vina `score_only`; strain relief is measured on a copy. SA is an auxiliary proxy.
This campaign does not run a retrosynthesis search or establish synthesis-route
reachability. The existing SimpRetro adapter remains a separate module.

The campaign emits a durable SQLite ledger, immutable proposal/execution
receipts, per-particle structures and scores, and coverage-first CSV reports.
Preflight particles are stored separately and do not enter the 100-particle arm
budgets. A passing real-checkpoint preflight is required before production.
