# Regional graph and pose guidance / 区域图与位姿引导

## Experimental claim and evidence / 实验命题与证据

The hypothesis is that a fixed, edited local graph and surviving contacts from an
original Steer pose can serve as useful soft conditions during native molecular
generation. Lower surrogate loss is **not** evidence of better synthesis,
affinity, physical stability, or retained PLIP contacts. Those are independent
terminal measurements. The implementation is an explicitly new approximation;
it does not reproduce exact Discrete Guidance or TFG-Flow.

本实验验证“局部图与原始互作锚点是否能帮助生成保留编辑意图的结构”。图的边缘
概率与坐标匹配只是可微代理，不能证明联合分子图有效、立体化学正确、互作成立或
可合成性改善。未经运行的真实模型梯度、最终质量与实验效果均保持未知。

| Evidence | Original role | Transfer to this program | Limit |
|---|---|---|---|
| [DiGress §3.2, Eq.3](https://arxiv.org/html/2209.14734v3) | Cross-entropy for clean node and edge classification during training | Local categorical target loss | Training loss does not establish inference-time guidance quality |
| FLOWR pinned `flowr/models/losses.py`, `compute_type_loss`, `compute_bond_loss`, `compute_charge_loss` | Checkpoint-native classification losses | Bind classes to the actual vocabulary and product preprocessing | Regional balance and inference controller are new adaptations |
| [Discrete Guidance, §3 and TAG implementation](https://arxiv.org/html/2406.01572v3) | Adjust CTMC rates with conditional-probability ratios; approximate ratios with derivatives | Evidence that discrete control needs an explicit discrete operator | Our output-logit intervention is neither the exact rate transform nor TAG |
| [TFG-Flow §3.2–3.4](https://arxiv.org/html/2501.14216v2) | Separate discrete Monte-Carlo rate estimation and continuous endpoint guidance | Keep categorical and coordinate mechanisms distinct; use a real endpoint VJP | We do not implement its Monte-Carlo estimator or theoretical sampler |
| EvoMolSteer `generation/scalar_guidance.py::live_pullback` | Real generator endpoint reward pulled back to current coordinates | Reuse detached old self-conditioning and true coordinate VJP semantics | Native-step RMS dose and output-logit operator are new here |
| Squared categorical probability error | Explicit mathematical comparator | Brier loss to contrast CE sensitivity | No claim that this precise molecular guidance variant was previously validated |

FLOWR revision: `739a33c2aab74d074dd24605b9a2e34dc809ae9c`.
EvoMolSteer reference revision: `1834d89541348c0a882d79a696bec31c3e147d48`.
The external checkouts are not edited by this extension. The literature retrieval
uses only these primary mathematical/ML papers; no biological knowledge base is
used for reward construction.

## RewardDesignIR / 目标、区域与候选架构

Every run binds the region condition file by its canonical JSON SHA256. The
campaign manifest and task input receipts bind source/product structures,
receptor and checkpoint; the worker records its arguments and tensor hashes.
`controller.reward_design_ir()` serializes the execution-specific decision record;
`controller.provenance()` serializes the smaller active-program contract.

- **I**: surviving atoms involved in original Steer contacts. Deleted contacts are
  recorded and cannot be restored by inventing a coordinate anchor.
- **E**: actual edited/new atoms and deletion-boundary atoms, determined after
  native SynCraft graph reconstruction.
- **Q = I ∪ E**: deduplicated categorical support. Repeated contact observations or
  overlapping I/E membership do not multiply an atom's weight.
- **Pairs**: unique undirected Q-internal pairs and E-incident boundary rows, from
  the explicit condition mask, including applicable no-bond targets. Self pairs
  are excluded from the reward even if retained in native hard-condition masks.
- **Coordinate anchors**: I only, in the original receptor-world Angstrom frame.
  Newly placed E coordinates may be arbitrary initialization and are not targets.
- **Monitoring only**: pIC50, Vina score-only, SA, structural strain, validity,
  stereochemistry, graph retention and PLIP contact preservation.

The selected architecture is a hybrid of soft output-logit intervention and
continuous real-model VJP. Hard graph/fragment clamping remains a separate
inpainting experiment. Exact discrete conditional-rate guidance was considered
but is not implemented under this sampler contract. Affinity-gradient guidance is
excluded to keep the affinity head an independent terminal measurement.

条件输入不是已诊断的物理错误；它是用户指定的保留假设。我们没有伪造 StatePacket
或未提供的诊断。坐标尺度0.5 Å、引导权重以及0.025 Å限幅均来自已批准的探索方案，
不是经过CK2校准的物理阈值。最终条件失败只加标签、保留结果，不据此补采样或隐瞒
无效产物。

## Loss and sensitivities / 损失与导数

For endpoint logits z, let p = softmax(z) and y be the checkpoint-native encoding
of the edited product. The four component groups are atom type, charge, present
bond, and no-bond. Each group is averaged over its unique supported slots or
pairs. The graph loss is the mean of the nonempty component losses; empty bond
groups are absent, rather than assigned a measured zero.

\[
\ell_{CE}(p,y)=-\log p_y,\qquad
\ell_{Brier}(p,y)=\sum_c(p_c-\mathbb{1}[c=y])^2.
\]

Both have their simplex optimum at the target one-hot distribution. CE has an
unbounded penalty as target probability approaches zero; Brier is bounded.
Unnormalized reward derivatives for one classification item are

\[
\nabla_z(-\ell_{CE})=e_y-p,\qquad
\nabla_z(-\ell_{Brier})=2J_{softmax}(e_y-p).
\]

The component-count and item-count divisors remain part of the actual gradient.
Bond logits are first symmetrized as `(z_ij + z_ji)/2`; each upper-triangle pair
then contributes once. Its raw-logit derivative is split equally between the two
orientations, so the intervention preserves undirected symmetry.

The v2 checkpoint can remove aromaticity. Chemical RDKit bond class 4 is therefore
**not** blindly used as the model target. The worker supplies the real native
encoded product posterior, including its Kekule single/double labels. Atom order,
identities and charges are checked against the condition artifact.

Real-checkpoint preflight also established that native absent-edge adjacency rows
can be all-zero vectors; native training decodes them as class 0 by argmax. Target
binding accepts this representation without rewriting the native tensors. Node
and charge targets remain normalized; nonzero supported edge rows remain finite,
nonnegative and normalized. Diagonals and unrelated padded rows are not objectives.

For optional coordinates, let d_i = endpoint_world_i - original_anchor_i and
s = 0.5 Angstrom:

\[
L_{pose}=\frac1{|I|}\sum_{i\in I}
\left[\sqrt{1+\|d_i\|^2/s^2}-1\right],\qquad
R=-L_{graph}-\mathbb{1}_{pose}L_{pose}.
\]

\[
\nabla_{d_i}(-L_{pose})=
-\frac{d_i}{|I|s^2\sqrt{1+\|d_i\|^2/s^2}}.
\]

The pose loss is zero only at the anchors, with zero gradient there. It does not
impose a hard tolerance or a guarantee of directional interactions. The graph and
pose optima are jointly attainable in the surrogate representation, but the
generator's feasible dynamics may make their gradients conflict. The soft sum
does not enforce noncompensable per-contact preservation; final checks report
such failures independently.

## Actual control path / 实际控制链

Native FLOWR applies argmax to current atom/charge/bond states before generator
embedding. We never claim a derivative through those decisions. Instead:

1. During the fixed window, clone current coordinates x_t with `requires_grad`,
   freeze model weights and detach previous self-conditioning. Reuse the real
   generator call and its pocket encoding. Retain its last self-conditioning
   recycle output graph.
2. Evaluate R on endpoint logits and, when enabled, endpoint world coordinates.
   Compute both endpoint-logit gradients and the true VJP `∂R/∂x_t` in one
   autograd call. A disconnected or nonfinite derivative fails explicitly.
3. Normalize categorical gradients with one per-particle RMS over directly
   supported unique logit entries. Apply `eta * dt` after normalization, softmax
   the modified logits, and pass probabilities to the unchanged native CTMC.
4. Execute exactly one native integration step. Scale the current-coordinate VJP
   by that particle's actual native coordinate-step RMS, apply eta, and cap each
   atom's injected displacement at 0.025 Angstrom. Padding remains immobile.
5. Leave the next self-conditioning prediction native and detached. It is not an
   additional hard graph clamp. Subsequent generator Jacobians can spread a local
   endpoint signal across other current-state coordinates.

The clock is fixed: t=0 noise, t=1 endpoint, 100 linear steps, active at
`0.1 <= t < 0.9` (80 steps). It includes a final native suffix and native terminal
prediction. There is no window search, SMC, affinity reward or hard inpainting in
these gradient groups. Rounding the recorded float32 clock to six decimals binds
the exact registered boundary despite accumulated floating-point error.

The grid is CE/Brier × graph/graph-plus-pose × eta {0.25, 1, 4}, plus one common
zero-weight native baseline. Weight zero installs **no wrapper** and changes no
parameter, prediction, tensor, or random-number use. The chosen nonzero weights
control intervention dose and do not repair a misspecified surrogate.

对已经归一化的梯度再施加时窗与外部权重，避免时间权重被归一化抵消。分类通道
日志中的KL是“修改后与原始预测终态类别分布”的KL，不是假称完整轨迹或实际CTMC
条件分布的KL。坐标日志记录真实原生步长、注入量、原始VJP及限幅情况。

## Falsifiers, validation and runtime artifacts

The strongest shortcut is matching supported categorical marginals and a few
anchors while producing an invalid connected graph, incorrect stereochemistry,
severe strain, or a donor/acceptor geometry that no longer forms the old contact.
Another is satisfying the reward during control but losing it in the native
suffix. Independent final graph/contact/quality metrics and fixed attempted-sample
denominators reveal these outcomes; low reward alone never counts as success.

The CPU tests cover CE/Brier finite differences and sign, unique-pair counting,
mask support, per-particle batch invariance, radial pseudo-Huber zero/gradient,
native-step scaling, padding and clipping, method/parameter restoration, zero
weight bitwise/RNG parity, and the exact 80-step clock. A toy nonlinear real
forward exercises the same endpoint-to-current autograd path.

`probe_model_reward` additionally checks an actual loaded checkpoint before a
production group: current-coordinate VJP finite difference and all three
output-logit derivative finite differences, using a real noisy state and native
conditioning. It restores RNG and parameter flags. Numerical failures are not
silently replaced with identity gradients or detached predictions. These local
checks do not validate the scientific benefit of the reward.

Worker outputs include `guidance_trace.json`, active reward provenance, the
RewardDesignIR, and a separate real-model probe result when requested. Every step
records losses, raw logit and coordinate gradient RMS, per-head endpoint KL,
native RMS, actual injected RMS/max displacement, and clipping fractions. Native
zero-guidance sampling and terminal evaluation remain separately attributable.

English derivation summary: this program treats the edited product as a local
categorical reference and the original Steer contacts as optional world-frame
coordinate anchors. It compares two target-consistency loss curvatures, rather
than assuming a training loss automatically constitutes valid conditional
sampling. The categorical path acts on differentiable endpoint logits before the
native discrete transition. The coordinate path differentiates the same reward
through the actual frozen generator, conditional on the current state and
detached old self-conditioning. The registered dose, clock and cap define a
bounded intervention, while native completion and independent final evaluators
test whether the local surrogate survives and remains chemically useful.
