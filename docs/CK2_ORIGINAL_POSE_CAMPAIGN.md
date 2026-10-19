# CK2 original-pose campaign protocol

## Frozen inputs and decisions

The 20 candidates are the historical pIC50-ranked unique graphs already selected
in `examples/ck2_affinity_top20`. Neither ranking nor a geometry/SA filter changes
after inspecting responses. Original heavy-atom coordinates are mapped by complete
graph isomorphism without fitting. Ambiguous correspondences are enumerated and
the deterministic choice is recorded. PDB/PDBQT intermediate serialization may
round coordinates; the input SDF is never overwritten by an evaluator.

Native SynCraft interaction-aware retrieval, prompts and reconstruction are
reused. Existing host guidance remains present. Interaction descriptions are
computed from the original Steer complex. They are soft prompt guidance; the
native editor may replace/delete an interacting atom. Such changes are recorded,
not silently vetoed by an extra graph audit. Unparsable outputs allow an initial
proposal plus five additional proposals, with the same prompt and no fabricated
error-feedback demonstrations. STOP and unchanged are distinct from parse failure.

DeepSeek uses `deepseek-flash`, enabled thinking and `reasoning_effort=high`,
maximum output 65,536 tokens, 1,200-second timeout and three transport attempts.
The high-mode request omits ineffective temperature. Only final content is parsed;
reasoning content is not written to disk. API credentials remain in the environment.

## Conditions and generation arms

I contains surviving original interaction atom maps. E contains actual edited,
added and deletion-boundary maps. Q is their deduplicated union. Original contacts
that were deleted or changed are explicitly listed. A graph-only placeholder can
be constructed without conformer embedding; its free coordinates are not prior
or self-conditioning inputs. The source ligand defines the same pocket for all
arms. All arms use the edited product's slot count; the campaign does not add a
variable-size proposal mechanism.

| Arm | Atom/charge condition | Bond condition | Coordinate condition |
| --- | --- | --- | --- |
| graph | Entire edited graph | Entire edited graph | None |
| fragment_graph | Q | Q-induced pairs and E-incident pairs, including absent bonds | None |
| fragment_graph_pose | Q | Same local graph | Original world coordinates of I only |
| unguided | None | None | None |
| gradient | Soft Q labels | Soft local graph labels | Optional soft original I anchors |

E coordinates are free except for overlap with I. Atom, edge and coordinate
channels are independent, including clocks, fixed embeddings and detached native
self-conditioning. An empty coordinate condition is reported as not applicable
for coordinate arms; it is not relabeled as a new control. Zero generation degrees
of freedom are also reported explicitly. Whole-graph inpainting can refine pose
but cannot repair its already fixed chemical graph.

Checkpoint-native categorical encoding supplies labels, including its Kekule
representation. RDKit aromatic bond class 4 is not assumed to be a model label.
Native FLOWR source and v2 checkpoint are pinned by revision/SHA-256.

For each usable graph, the campaign declares 3 hard arms + 1 shared unguided
control + 2 losses × 2 coordinate settings × 3 weights = 16 arms. Each applicable
arm makes exactly 100 attempts, in 25 microbatches of four. Seeds are paired across
arms, independently derived across batches/cases. Maximum production budget is
32,000 attempted particles; excluded design outcomes and inapplicable arms reduce
the actual budget and are visible in reports. No resampling, deduplication or
best-of-extra sampling fills failed slots.

## Gradient implementation

The two graph losses are cross-entropy and Brier loss over native atom, charge,
present-bond and absent-bond categorical targets. Nonempty components are
normalized by their counts and averaged; undirected pairs count once. The optional
coordinate term is radial pseudo-Huber displacement from original I coordinates,
with exploratory scale 0.5 Å. Rewards are negatives of these losses.

The current categorical state reaches the model through native argmax, so this
implementation does not claim a derivative through discrete current labels.
It differentiates reward with respect to endpoint logits, normalizes the gradient
by a common per-particle RMS, modifies predicted probabilities and invokes the
native categorical sampler. Coordinates use an actual model endpoint-to-current
VJP, normalized to the native step RMS with a 0.025 Å per-atom injection cap.
Model parameters are frozen and native self-conditioning is detached. This is an
approximate intervention, not an exact conditional-rate sampler.

The predeclared strength grid is 0.25, 1, 4. All conditions keep **0.1 ≤ t < 0.9**
with 100 linear integration steps, giving 80 guided updates. No window search or
adaptive strength search is performed. Affinity, Vina, SA and PLIP are terminal
evaluators rather than hidden reward terms. See
[the reward mechanism](REGIONAL_GRADIENT_GUIDANCE.md) and each arm's saved
`reward_design_ir.json` for mathematical details and approximation limits.

## Real-model preflight and final evaluation

Production requires a code/protocol-bound passing receipt. Separate preflight
batches check identical unguided/zero-weight output tensors and RNG/prior states,
active CE/Brier gradients, finite differences at an actual noisy t=0.4 state,
the 80-update clock, actual injection/cap, and hard terminal category/pose
conditions. An identical SDF is rescored in different orders/task seeds to check
affinity invariance. Preflight has its own directory and is excluded from arm
denominators. Failure never silently substitutes replay or surrogate gradients.

The worker saves raw final tensors and a full physical-particle list before any
RDKit decoding. Every attempted particle receives a result row. Unreadable
particles, duplicates, disconnected graphs, absent scores and evaluation errors
remain distinguishable. There is no largest-component pruning during FLOWR
decoding. Native SynCraft's earlier graph reconstruction behavior is retained.

Final pIC50 is recomputed on each actual decoded pose with zero coordinate noise,
t=0.9999 and native self-conditioning enabled. Its first pass uses a random prior;
therefore every pose uses a fixed seed 42 and a singleton native collate/predict,
independent of generation seed, list order or microbatch membership. All RNGs and
model flags are restored. Historical selection pIC50 remains provenance, not the
new baseline. Vina uses `score_only` on the actual pose without redocking. PLIP,
PoseBusters, same-graph MMFF relief on a copy, SA and source similarity are recorded.
Absolute force-field energies across different graphs are not compared as strain.

Soft-arm atom maps are not hard identities. A separate tensor-slot lineage
connects reference and generated slots. Exact original-contact comparisons require
matching actual element and charge; substitutions are reported independently.
Reports also include residue/type-only recall and interaction coordinate drift.
Atom maps and coordinates in the scored generated SDF are not rewritten for this
comparison. None of these scores constitutes experimental affinity or synthesis
verification. Retrosynthesis is disabled in this profile.

## Execution, persistence and recovery

`python -m agenticgen.campaign` offers `prepare`, `run`, `status`, `requeue` and
`report`. Phases are `baseline`, `design`, `preflight`, `generation`, `evaluate`
and `all`. Both generation-phase and direct generation-job entry require preflight.
The source-only snapshot deployment and runtime requirements are documented in
[the remote guide](REMOTE_CK2_CAMPAIGN.md).

A SQLite WAL ledger claims tasks atomically. An OS-held controller lock prevents
duplicate controllers. Every retry has a fresh execution directory and code/input
receipt. Completed API content is reused after controller failure, avoiding
duplicate calls. Sampling is committed before affinity scoring; head-only failures
can recover without repeating generation. Evaluation children bind the exact parent
execution and affinity receipt hashes. A re-generated parent supersedes old
children without deleting their artifacts.

The 30-minute thread heartbeat inspects liveness, ledger and failure receipts.
Healthy unchanged runs stay quiet. Infrastructure/software errors may be diagnosed,
fixed and resumed under the existing authorization. Scientific design failures,
STOP, bad decoded structures and duplicates are outcomes, not reasons for hidden
extra proposals or changed parameters. Code changes preserve failed versions and
repeat affected verification before continuing. Reports separate planned,
observed, decoded and scored coverage, and keep per-case/per-arm comparisons.
