---
name: agenticgen-syncraft-foundation
description: Run AgenticGen's configured SynCraft foundation for CK2 small-molecule computational evaluation, optionally with frozen generation-history evidence.
---

# AgenticGen SynCraft foundation

Use the project-local CLI or `agenticgen.api.optimize(config, output)`. Read `docs/SYNCRAFT_FOUNDATION.md` and the exact chosen configuration before execution. Resolve paths relative to the configuration file. Credentials come from the host environment and must not be written to files or logs.

Choose the requested mechanism explicitly:

- `standard.paper.json`: fixed pass-K, graph audit, SimpRetro and similarity selection.
- `interaction.live.json`: original docking/PLIP context, fixed pass-K, product redocking and SA/binding selection.
- `standard.inference.json`: proposal diagnostics only; no final optimization winner.
- `standard.history.json`: the same standard foundation with frozen, graph-bound Flow-history evidence.

Run `agenticgen preflight --config CONFIG` before a new setup. Missing inventories, receptor inputs, runtimes or credentials require configuration; do not switch profiles to manufacture success. Keep generation-history evidence disabled for the baseline. Any model, pass-K, template, inventory or selection change belongs in the run configuration and manifest.

Run `agenticgen run --config CONFIG --output NEW_DIRECTORY`. Do not alter saved proposals after evaluation. Return the summary and result paths, graph/route/binding failure statuses, selected candidate if one exists, and actual executed scope. Distinguish search-reported routes from checked graph/inventory/template routes, and both from experimental feasibility. All binding and geometry results are computational predictions.

Use local CK2 inputs. Do not substitute a different target or search unrelated biological-risk knowledge bases. New source/product poses must bind to their exact molecular graph; original affinity and strain cannot be copied to an edited molecule.
