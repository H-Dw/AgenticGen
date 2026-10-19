# One actual runtime diagnostic

This is `single_b004_final_027`, the first frozen candidate with both original
interaction anchors and a partial graph mask, used for preflight coverage. It is
not a selected optimization winner. For both generated arms, particle index 0 was
chosen before examining scores. All four particles from each diagnostic batch
remain in the full run artifacts.

- `source.sdf`: the mapped original Steer pose, without fitting or redocking.
- `design.json`: actual DeepSeek high proposal, native SynCraft reconstruction
  and I/E/Q masks. Its chemical explanation is unverified model content.
- `graph_particle_0000.sdf`: whole-graph inpainting diagnostic; Vina **+32.966**
  kcal/mol and PoseBusters failure. This is a poor pose despite a readable graph.
- `ce_particle_0000.sdf`: soft CE graph-only diagnostic; Vina **−9.146** kcal/mol
  and PoseBusters pass. This does not establish superiority across the cohort.
- `measurements.json`: measured results and SHA-256 of the actual SDFs. Scores
  use Vina `score_only`; no redocking or coordinate replacement was performed.

These diagnostics precede production and are excluded from the 100-attempt arm
budgets. Generated-pose pIC50 is not included in this small export; the production
worker independently rescores every eligible actual final pose. Passing software
and gradient preflight does not imply that a generated molecule satisfies the
chemical, interaction or affinity objectives.
