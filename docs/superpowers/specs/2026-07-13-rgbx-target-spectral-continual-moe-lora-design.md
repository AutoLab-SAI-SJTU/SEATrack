# RGB-X Target-Spectral Continual MoE-LoRA Design

> Status: approved design, pending user review of this written specification
> Date: 2026-07-13
> Repository: `SEATrack-ProbAlign-VRE`
> Evidence snapshot: SAME arXiv:2602.01990v2; Prism commit `7154be2a72a4f8e694c4361b7c6e05bb51bf5cc4`; SEATrack commit `793b70f1d7226ceec67bda47bf2c00bb014ac4e7`; literature frozen on 2026-07-13

## 1. Decision summary

The approved research direction is **Reliability-Gated Target-Spectral Continual MoE-LoRA for RGB-X tracking**.

The method uses the eigenspectrum of bounded, target-specific second-moment memories in the existing HMoE router-input space. It does not use image Fourier frequency or sensor wavelength spectra. Historical RGB-X target geometry directly modulates the next frame's HMoE logits. After the routing mechanism is validated, the same history controls causal online updates of the router and, later, selected HMoE LoRA experts.

The approved progression is:

1. spectral-memory-only forward routing;
2. router-only online adaptation;
3. selected HMoE LoRA expert adaptation after predeclared gates pass.

The standard tracking result is sequence-reset OPE. A cross-sequence continual stream is a secondary deployment stress test, not a replacement for standard OPE.

## 2. Research question and claim boundary

### 2.1 Research question

In unlabeled RGB-X single-object tracking, can reliable bounded history estimate target-specific shared, private, dynamic, and background covariance subspaces that improve expert routing while limiting router drift and pseudo-label pollution during online MoE-LoRA adaptation?

### 2.2 Defensible contribution

The contribution to test is:

> A causal RGB-X tracking framework in which reliability-admitted target history forms bounded common/private/dynamic/background input-covariance eigenspaces that jointly control current HMoE routing, memory admission, and safe online updates of the router and selected low-rank experts.

This is a conditional claim until the experiments pass. The paper must not claim any of the following in isolation:

- the first covariance/eigenspace MoE router;
- the first online tracker;
- the first historical-memory tracker;
- the first RGB-X MoE or MoE-LoRA tracker;
- the first SVD/orthogonality method for LoRA;
- the first RGB-T test-time adaptation method;
- sparse inference or expert-skipping speedups.

### 2.3 Closest collisions

- [SAME](https://arxiv.org/abs/2602.01990) stabilizes MoE-LoRA continual instruction tuning with accumulated input geometry, expert preconditioning, and activation/freezing.
- [EMoE](https://arxiv.org/abs/2601.12137) routes current tokens using energy in an input-covariance-aligned eigenbasis.
- [ERMoE](https://arxiv.org/abs/2511.10971) uses expert eigenbases for stable routing.
- [SPMTrack](https://openaccess.thecvf.com/content/CVPR2025/html/Cai_SPMTrack_Spatio-Temporal_Parameter-Efficient_Fine-Tuning_with_Mixture_of_Experts_for_Scalable_CVPR_2025_paper.html) combines historical frames with routed low-rank experts.
- [DTPTrack](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.html) uses reliability-gated bounded temporal history.
- [PURA](https://openaccess.thecvf.com/content/CVPR2025/html/Shao_PURA_Parameter_Update-Recovery_Test-Time_Adaption_for_RGB-T_Tracking_CVPR_2025_paper.html) performs RGB-T test-time parameter/statistic updates and SVD-based recovery.
- [GOLA](https://arxiv.org/abs/2512.05359) applies SVD and orthogonality to RGB-T LoRA weight ranks.
- [SpecTrack](https://arxiv.org/abs/2607.05988) uses spectral prompts for multispectral/hyperspectral wavelength data.
- [XTrack](https://arxiv.org/abs/2405.17773), [FlexTrack](https://arxiv.org/abs/2507.05899), [OneTrackerV2](https://arxiv.org/abs/2605.03716), and SEATrack itself already cover RGB-X MoE routing, multimodal unification, heterogeneous/missing modalities, or low-rank experts.

The proposed distinction is not any single ingredient. It is the combination of causal target history, RGB-X common/private geometry, tracking target/background separation, bounded covariance memories, forward routing, and reliability-gated online MoE-LoRA updates.

Any “style” adaptation baseline must have a written inclusion and implementation protocol before its results are inspected. Baselines are not added or removed in response to favorable or unfavorable outcomes.

## 3. Existing-system facts and constraints

### 3.1 HMoE geometry

The current `HMoE` is in `lib/models/layers/attn.py`. For input `x=[B,N,768]` with two slots:

$$
H=\operatorname{reshape}(\operatorname{linear1}(\operatorname{norm}(x)))
\in\mathbb R^{B\times 2N\times384}.
$$

The router is:

$$
W_g=\texttt{gate\_thi}\in\mathbb R^{384\times ES},\qquad S=2,
$$

with attention experts $E=4$ and FFN experts $E=8$. Legacy logits are:

$$
L=HW_g.
$$

The same logits feed:

- `Dispatch = softmax(L / D_temp, dim=token-subtoken)`;
- `Combine = softmax(slot-summed L / C_temp, dim=expert)`.

All experts are evaluated. The method changes routing and update admission, not dense execution.

### 3.2 RGB-X and template/search ordering

In `lib/models/layers/attn_blocks.py`, each active block invokes each HMoE twice:

```text
HMoE(concat(RGB-template, X-template))
HMoE(concat(RGB-search, X-search))
```

Within either call, tokens have known order:

- first RGB template/search tokens;
- then X template/search tokens.

Consequently, $N=2L_z$ or $N=2L_x$ in the geometry above, and Dispatch normalization never pools template and search tokens together.

Attention HMoE is called before candidate elimination. FFN HMoE is called after candidate elimination. The current repository's nontrivial pruning path is shape-inconsistent because `candidate_elimination` expects attention tensors while its caller passes token tensors; current keep ratios of 1 hide that branch. Therefore the first spectral pilot requires `keep_ratio_search=1`. Before any pruning experiment, CE must be repaired to pass actual `[B,H,L,L]` attention and must pass a nontrivial-keep-rate test. After repair, RGB/X observer pairs use the intersection of global patch indices without reordering or filtering legacy HMoE inputs.

Template and search must not be pooled by raw token count. Search has four times as many spatial tokens and mostly background. The fixed template supplies immutable anchor moments; dynamic memories are updated from response-weighted search tokens.

### 3.3 Current inference limitations

`lib/test/tracker/seatrack.py` currently:

- stores only the initial template and latest bounding box;
- runs the whole frame forward under `torch.no_grad()`;
- discards historical search crops, features, confidence, and router distributions;
- has no online optimizer, spectral memory, replay, teacher, snapshot, or rollback;
- returns a fused response only.

Per-modality attention evidence can be obtained without two additional backbone forwards. `compute_gra_stats()` in `lib/models/layers/attn_blocks.py` already computes RGB and X template-to-search distributions locally, but the current model retains only aggregated scalar summaries. Instrumentation must expose the normalized evidence distributions per `(block, modality)` with global search indices. They are not modality-specific tracker boxes or head predictions.

If candidate elimination is active, every modality response is scattered back to the common full search grid through its global patch indices before cross-modal agreement or JS divergence is computed. Missing locations are masked rather than treated as zero-probability evidence.

The fused distribution is defined on the same grid as:

$$
P^F=\operatorname{normalize}
\left(
\texttt{output\_window}\odot\texttt{score\_map}
\right).
$$

Cross-modal comparison uses common valid support and renormalizes after masking. HMoE instrumentation must separately expose detached per-token `Combine`, keyed by `(block, attn_or_ffn, template_or_search)`, before any scalar aggregation.

### 3.4 LoRA constraints

- HMoE expert `lora_a[e]` has input-oriented shape `[384,4]`; an input-space preconditioner left-multiplies its gradient.
- The experts are slices of stacked parameters. Zeroing a gradient slice does not freeze AdamW momentum or decoupled weight decay. Unselected parameter slices and optimizer state must remain bitwise unchanged.
- qkv `MergedLinear` LoRA is merged during evaluation and may bypass its LoRA branch. It is excluded from the first design.
- Shared `linear1` and `linear2` stay frozen online. Updating `linear1` would invalidate all stored spectral coordinates.
- The shared rank-4 path remains a representational ceiling; spectral routing does not increase its rank.

### 3.5 Prism lessons that must not be copied blindly

The audited Prism implementation is not treated as authoritative code for this repository. Relevant deviations include centered/current-batch sketch replacement instead of the paper's cumulative uncentered moment, random QR directions instead of a sorted PCA basis, incomplete curvature coverage, and expert mixing semantics that can introduce cross-expert terms. SEATrack uses its own tensor geometry and a mathematically valid per-expert low-rank path.

In particular, for:

$$
G_g=\nabla_{W_g}\mathcal L\in\mathbb R^{384\times ES},
$$

an input-space transform is:

$$
\widetilde G_g=SG_g,
$$

not $G_gS$.

## 4. Architecture

### 4.1 Component boundaries

The design has five isolated components:

```text
PairedSpectralObserver
    captures aligned pre-router RGB/X features and target/background factors

BoundedEigenspaceBank
    stores and updates low-rank spectral states

HistoryConditionedRouter
    converts past projectors into bounded residuals on current HMoE logits

ReliabilityEstimator
    predicts memory safety, update utility, and modality asymmetry

ContinualUpdateController
    executes predict-then-update, replay, gradient shaping, transactions, rollback, and reset
```

Proposed interfaces:

```text
PairedSpectralObserver.capture(key, r_rgb, r_x, global_indices, frame_meta)
ReliabilityEstimator.score(frame_output, modality_responses, anchor_state)
    -> q_mem, q_upd, r_rgb, r_x, diagnostics
BoundedEigenspaceBank.prepare(key, factors, admission)
BoundedEigenspaceBank.route_context(key, mode, causal_prior)
    -> bounded_operator
ContinualUpdateController.after_prediction(frame_transaction)
```

`key` identifies `(block, attn_or_ffn)`. The bank distinguishes immutable template anchor factors from dynamic search memories.

### 4.2 Paired RGB-X observations

Let paired raw legacy router-slot features be:

$$
r_{t,i,s}^{R},r_{t,i,s}^{X}\in\mathbb R^{384}.
$$

Define observer-only RMS-normalized copies:

$$
\widetilde r_{t,i,s}^{m}=
\frac{r_{t,i,s}^{m}}
{\operatorname{RMS}(r_{t,i,s}^{m})+\epsilon}.
$$

RMS normalization never enters the zero-strength legacy routing path.

The shared backbone and router define a common coordinate system, so the first implementation uses no learned alignment map:

$$
c_{t,i,s}=\frac{\widetilde r_{t,i,s}^{R}+\widetilde r_{t,i,s}^{X}}{\sqrt2},\qquad
d_{t,i,s}=\frac{\widetilde r_{t,i,s}^{R}-\widetilde r_{t,i,s}^{X}}{\sqrt2}.
$$

This assumption has a predeclared audit: matched RGB-X pairs must have stronger shared alignment than shuffled pairs. If that audit fails, the common/private branch fails its gate; a learned aligner is not silently added to rescue the first experiment.

After frame $t$ has been predicted, the fused response and predicted box form soft target weights $a_{t,i}$ and hard-background weights $b_{t,i}$. Background weights are restricted to high-response distractors outside a dilated predicted box.

Define:

$$
M_{id,t}=
\frac{\sum_{i,s}a_{t,i}c_{t,i,s}c_{t,i,s}^{\top}}
{\sum_{i,s}a_{t,i}+\epsilon},
$$

$$
M_{private,t}=
\frac{\sum_{i,s}a_{t,i}d_{t,i,s}d_{t,i,s}^{\top}}
{\sum_{i,s}a_{t,i}+\epsilon},
$$

$$
p_t=\frac{\sum_{i,s}a_{t,i}c_{t,i,s}}
{\sum_{i,s}a_{t,i}+\epsilon},\qquad
M_{dyn,t}=(p_t-p_{t^-})(p_t-p_{t^-})^{\top},
$$

$$
M_{bg,t}=
\frac{\sum_{i,s}b_{t,i}c_{t,i,s}c_{t,i,s}^{\top}}
{\sum_{i,s}b_{t,i}+\epsilon},
$$

where $t^-$ is the previous trusted frame.

The four labels are operational, not claims of identifiable causal latent variables:

- $C_{id}$: shared target identity history plus immutable initial anchor;
- $C_{private}$: paired modality difference history;
- $C_{dyn}$: trusted target appearance transitions;
- $C_{bg}$: hard-background history.

Because $d_td_t^\top$ loses modality sign, private-branch sign is supplied solely by the latest trusted reliability asymmetry. No unused signed-mean state is stored.

### 4.3 Bounded spectral memory

For each adaptive state $j\in\{id,private,dyn,bg\}$, store:

$$
\mathcal M_j=(U_j,\Lambda_j,n_j^{eff}),
$$

with fixed rank. Identity additionally owns an immutable initial-anchor factorization:

$$
\mathcal M_{init}=(U_{init},\Lambda_{init}),
$$

computed once from the labeled first frame and never passed to the streaming update. The implementation supports ranks 8, 16, and 32, uses 16 as the provisional engineering default, and locks the normative rank through M1 before gate confirmation. Both $\mathcal M_{init}$ and the adaptive identity state are individually capped at that rank.

Let the old normalized moment be $C\approx U\Lambda U^\top$ with effective mass $n$. For one logical frame, collect all admitted weighted vectors before decomposition:

$$
ZZ^\top=\sum_\ell w_\ell v_\ell v_\ell^\top,
\qquad m=\sum_\ell w_\ell,
\qquad q=A_t^{mem}q_t^{mem}.
$$

If $q=0$, bypass the update and keep $(U,\Lambda,n)$ bitwise unchanged. For $q>0$, apply history decay once per logical frame:

$$
n'=\beta n+qm.
$$

For $n'>0$, update by a thin SVD of:

$$
B=
\left[
\sqrt{\frac{\beta n}{n'}}U\Lambda^{1/2},
\sqrt{\frac q{n'}}Z
\right].
$$

Sort the squared singular values and truncate to the fixed rank. This is a cumulative uncentered second moment, not Prism's centered overwrite sketch. Never apply $\beta$ separately to microbatches from the same frame; the frame-level aggregation above is the unit tested for batch-partition invariance.

Adaptive-state projectors use eigenspace clusters rather than individual eigenvector signs:

$$
\Pi_j=U_j\operatorname{diag}
\left(
\frac{\lambda_{j,k}}
{\lambda_{j,k}+\epsilon\bar\lambda_j}
\right)U_j^\top.
$$

A state is inactive until it passes minimum effective-sample and eigengap checks. Each $U_j$ is orthonormal internally. The four subspaces are not forced pairwise orthogonal because identity, appearance change, and background can genuinely overlap. Their principal angles are measured, and the combined routing operator is norm-bounded.

For identity, $\Pi_{id}$ is evaluated from the weighted concatenation of the immutable $C_{init}$ factors and adaptive $C_{id}$ factors without overwriting either source. The initial-anchor contribution is active from initialization; only the adaptive identity contribution is subject to effective-sample admission.

The immutable initial-template component of $C_{id}$ never decays:

$$
C_{id}^{t}=\lambda_0C_{init}+(1-\lambda_0)C_{adaptive}^{t}.
$$

The two factors are never jointly truncated or written back into $\mathcal M_{init}$. A temporary eigendecomposition of their concatenated weighted factors may be used to evaluate $C_{id}$ or its projector, giving rank at most $2r$, but the immutable source factors remain unchanged. The state and runtime memory budgets count both ranks and all temporary factors.

No raw frame history is unbounded. Replay is separately capped at the initial template plus at most eight trusted search crops.

### 4.4 History enters forward routing

Frame $t$ may use only memories committed through $t-1$. Let $a^-_{t,i}$ be a causal target prior:

- the initial-box mask for template tokens;
- the previous target state mapped to the current search crop for search tokens.

Let $\bar a_{t-1}\in[-1,1]$ be the latest trusted RGB-versus-X reliability asymmetry. Define modality-specific bounded operators:

$$
A_{t-1}^{R,i}=
a^-_{t,i}\left(
\alpha_{id}\Pi_{id}
+\alpha_{dyn}\Pi_{dyn}
+\alpha_p\bar a_{t-1}\Pi_{private}
\right)
-(1-a^-_{t,i})\alpha_{bg}\Pi_{bg},
$$

$$
A_{t-1}^{X,i}=
a^-_{t,i}\left(
\alpha_{id}\Pi_{id}
+\alpha_{dyn}\Pi_{dyn}
-\alpha_p\bar a_{t-1}\Pi_{private}
\right)
-(1-a^-_{t,i})\alpha_{bg}\Pi_{bg}.
$$

Normalize each operator so that:

$$
\|A_{t-1}^{m,i}\|_2\le\kappa<1.
$$

The core coefficients are one global four-scalar vector shared by blocks 5 and 9, attention/FFN, and both modalities; no per-layer coefficients are fitted. Parameterize them as:

$$
(\alpha_{id},\alpha_{dyn},\alpha_p,\alpha_{bg})
=\alpha_{budget}\operatorname{softmax}(u),
\qquad u\in\mathbb R^4,
$$

with $\alpha_{budget}$ frozen in the registry. Fit $u$ only on prediction-centered causal rollout clips by minimizing the next-frame outer tracking loss, using GT solely in that offline outer loss. Select one coefficient checkpoint by the registered $J_{core}$ metric on calibration sequences, then freeze it before gate confirmation. The private sign is supplied only by $\bar a_{t-1}$ and the RGB/X branch definition; learned coefficients cannot reverse its meaning. Per-layer coefficients are a post-core ablation, not a rescue option.

For column-vector notation:

$$
\widehat r_{t,i}^{m}=
(I+\rho_{t-1}A_{t-1}^{m,i})r_{t,i}^{m},
$$

where $\rho_{t-1}$ is committed history confidence. In batched row-vector code, the operator is transposed on the right.

$$
\rho_{t-1}=\operatorname{clip}
\left(
\operatorname{EMA}_{j\le t-1,\,A_j^{mem}=1}q_j^{mem},
0,1
\right).
$$

The initial template sets $\rho_0=1$ only for its immutable anchor projector. Dynamic/private/background projectors remain inactive until their own effective-sample and eigengap gates pass.

The HMoE logits become:

$$
L_t=\widehat R_tW_g.
$$

The residual logit change is clipped to a configured bound before it reaches Dispatch and Combine. No current-frame response influences the frame's first prediction.

This design avoids a second router. History changes allocation through the existing $W_g$.

### 4.5 Dense expert semantics

Current HMoE remains dense. `Combine` is the primary expert-utilization measure because it directly mixes expert outputs. `Dispatch` diagnostics are secondary because token-count changes affect its normalization.

In Stage E, “selected expert” means an expert is eligible for parameter updates. It does not mean unselected experts are skipped in the forward pass.

## 5. Causal reliability and online objectives

### 5.1 Predict-then-update contract

For frame $t$:

$$
\widehat y_t=
f_{\theta_{t-1}}(z_0,x_t,\mathcal M_{t-1})
\rightarrow
\text{commit output}
\rightarrow
\text{score reliability}
\rightarrow
\text{prepare memory/update transaction}
\rightarrow
\theta_t.
$$

The update can affect only frame $t+1$ and later. The deployment tracker receives a sanitized causal frame record containing only the previous prediction and non-label frame metadata. Ground truth, visibility, attributes, and corruption identities or boundaries remain evaluator-side and are removed before `track()` is called. `track()` asserts that forbidden keys are absent, and a sentinel integration test fails on any post-initialization access to them.

The committed prediction runs without gradients. Only an accepted candidate frame is reprocessed for the update objective.

Every computation associated with frame $t$—reliability features, augmentation gates, teacher targets, target/background weights, and update losses—is derived from the committed no-gradient prediction and detached before optimization. Teacher and student re-forwards of frame $t$ use the pre-frame state $(\theta_{t-1},\mathcal M_{t-1})$. Candidate memory $\mathcal M_t$, replay writes, optimizer state, and $\theta_t$ become visible only at frame $t+1$. The EMA teacher is snapshotted before the candidate update. A forward/backward cycle may use only frame $t$ and a trusted past crop, never a future frame.

### 5.2 Dual reliability

A small, frozen, validation-calibrated estimator outputs:

$$
q_t^{mem}=P(\text{current pseudo-target is correct}),
$$

$$
q_t^{upd}=P(y_t^{upd}=1\mid o_{\le t}),
$$

and modality reliabilities:

$$
r_t^R,r_t^X\in[0,1],\qquad
a_t=\frac{r_t^R-r_t^X}{r_t^R+r_t^X+\epsilon}.
$$

Permitted observable inputs are:

- fused response peak, top-1/top-2 margin, PSR, and normalized entropy;
- late-layer RGB and X template-to-search attention distributions and confidence;
- cross-modal response agreement;
- current-to-initial and current-to-trusted prototype similarity;
- temporal motion and scale residuals;
- weak-view consistency;
- a low-frequency forward/backward cycle signal.

There is no assumed RGB-only/X-only box output in the core method.

Reliability evaluation is two-stage for efficiency. Cheap fused-response, modality-attention, motion, and anchor features produce a provisional $q_t^{mem}$ every frame. A frame that fails the cheap $q_t^{mem}$/anchor pre-gate is rejected immediately. Every frame that passes it must run weak-view consistency and the registered forward/backward cycle check before either memory or gradient admission is evaluated; if the check cannot run or exceeds its latency budget, the frame is rejected rather than inheriting or inventing a cycle value. The check is therefore low-frequency only in the amortized sense that it runs on pre-gated candidates, and $e_t^{cycle}$ is always defined for every proposed memory write.

Offline labels are:

$$
y_t^{mem}=\mathbf1[
\operatorname{IoU}(\widehat b_t,b_t^{gt})\ge\tau_{loc}
],
$$

$$
y_t^{upd}=\mathbf1[
L_{t+1}(\theta'_t)
<L_{t+1}(\theta_{t-1})-\delta
\;\land\;
D_{anchor/replay}(\theta'_t)\le\delta_A
].
$$

The candidate $\theta'_t$ is produced only by the test-time unsupervised objective. Candidate and control branches use the same frame-$t+1$ crop, augmentations, and RNG seed. Frame $t+1$ and its GT form only the offline label and outer loss; no frame-$t+1$ quantity is an estimator input. Thus $q_t^{upd}$ estimates one-step next-frame benefit under the declared anchor/replay audit, not long-horizon safety. Long-horizon safety is evaluated by committed-stream recovery and rollback metrics.

### 5.3 Memory and gradient admission

Memory admission is:

$$
A_t^{mem}=\mathbf1[
q_t^{mem}\ge\tau_M
\land s_t^{anchor}\ge s_M
\land e_t^{cycle}\le\delta_M
].
$$

Gradient admission is stricter:

$$
A_t^{grad}=A_t^{mem}\mathbf1[
q_t^{upd}\ge\tau_U
\land e_t^{aug}\le\delta_U
\land \neg\mathrm{cooldown}
].
$$

Thresholds are calibrated on validation false-admission risk, not chosen from test results.

State-write rules are:

- both modalities reliable: update all eligible states;
- one modality reliable: write none of the paired $C_{id}$, $C_{private}$, $C_{dyn}$, or $C_{bg}$ states in the first implementation; current routing may still use the last committed asymmetry to favor the trusted modality;
- both unreliable: update neither memory nor parameters;
- high fused confidence but failed anchor/cycle checks: treat as confident drift and reject.

### 5.4 Minimal online objective

The online loss has two families:

$$
L_{online}=L_{current}+\lambda_RL_{replay},
$$

$$
L_{current}=L_{eq}+\lambda_XL_{cm}.
$$

For $L_{eq}$, an EMA teacher predicts on a weak view. The student predicts on a distinct strong geometric/photometric view. The student output is mapped back to the weak-view coordinates:

$$
L_{eq}=
\operatorname{JS}
(W^{-1}P_\theta(A_sx_t),
\operatorname{sg}P_{\bar\theta}(A_wx_t))
+\lambda_bD_{box}
(W^{-1}b_\theta^s,
\operatorname{sg}b_{\bar\theta}^w).
$$

This is not same-model/same-input self-distillation.

For conditional cross-modal consistency:

$$
L_{cm}=
(1-|a_t|)\operatorname{JS}(P_t^R,P_t^X)
+|a_t|\operatorname{JS}
(P_t^F,\operatorname{sg}P_t^{m^*}),
$$

where $m^*$ is the more reliable modality. When one modality fails, the reliable modality supervises the fused response; the bad modality is not forced to agree with the good one.

Replay stores detached data only:

- the immutable first-frame replay unit $(z_0,x_0^{anchor},y_0^{GT})$;
- at most eight trusted units $(z_0,x_j,T_j,y_j^{safe},\pi_j^{safe})$;
- crop transform $T_j$ from crop to image coordinates;
- safe teacher response/box and target-token Combine distribution;
- reliability, modality state, and timestamp.

The first-frame labeled unit is the only replay unit containing true supervision. For later units, $y_j^{safe}$ is a detached safe-teacher target. Replay prediction is always $f_\theta(z_0,x_j)$; a template or search crop alone is not a valid tracking input. Teacher, optimizer, snapshot, and transfer storage are included in reported memory rather than hidden outside the replay budget.

When the eight-crop replay is full, replace the crop with the lowest product of admitted reliability and feature-space novelty. The immutable initial anchor is never an eviction candidate. The same deterministic eviction rule and byte budget apply to every replay baseline.

The replay loss is:

$$
L_{replay}=
D_{pred}(f_\theta(z_0,x_j),\operatorname{sg}y_j^{safe})
+\lambda_\pi
\operatorname{KL}
(\operatorname{sg}\pi_j^{safe}\|\pi_\theta(z_0,x_j)).
$$

The default objective does not include router entropy, expert load balance, parameter L2, or a separate cycle loss. They are not justified by current evidence. In the core method, cycle is a required candidate gate and rollback signal, not a loss; removing it is a separately registered ablation.

### 5.5 Spectral gradient shaping

For router inputs, define:

$$
C_{adapt}=\lambda_dC_{dyn}+|a_t|\lambda_pC_{private},
$$

$$
C_{protect}=C_{id}+\lambda_bC_{bg}.
$$

Let the positive-definite protection metric and its whitened adaptation operator be:

$$
P=C_{protect}+\epsilon I,
\qquad
K=P^{-1/2}C_{adapt}P^{-1/2}
=Q\operatorname{diag}(\gamma_i)Q^\top,
\qquad Q^\top Q=I.
$$

Construct the symmetric filter in whitened parameter coordinates:

$$
d_i=\operatorname{clip}
\left(
\frac{\gamma_i}{1+\gamma_i},s_{min}^{w},s_{max}^{w}
\right),
\qquad
S_t^{w}=s_{min}^{w}(I-QQ^\top)+Q\operatorname{diag}(d_i)Q^\top.
$$

The coordinate mapping is explicit. Conceptually reparameterize $V=P^{1/2}W_g$, so that $G_V=P^{-1/2}G_{gate}$. Filter in $V$ coordinates and map the step back to $W_g$:

$$
T_t=P^{-1/2}S_t^{w}P^{-1/2},
\qquad
S_t=\frac{T_t}
{\max(1,\|T_t\|_2/s_{raw})}.
$$

The registry enforces $0\le s_{min}^{w}\le s_{max}^{w}\le1$ and $0<s_{raw}\le1$. Thus $S_t$ is symmetric positive semidefinite, correctly includes whitening and inverse mapping, and satisfies $\|S_t\|_2\le s_{raw}$. This replaces a direct projector built from generalized eigenvectors, which are not generally Euclidean-orthonormal and would not guarantee the declared Euclidean norm bound. The implementation may use low-rank Woodbury/eigendecomposition identities, but a numerical test must show equivalence to the dense 384-dimensional reference on small synthetic cases.

Then:

$$
\widetilde G_{gate}=S_tG_{gate}.
$$

An expert does not receive the raw router input. Its admitted, detached input is:

$$
\xi_{t,e}=\operatorname{reshape}
\left(\operatorname{Dispatch}_t^\top H_t\right)_e
\in\mathbb R^{S\times384}.
$$

For expert-specific identity/private/dynamic/background factors, apply that same detached linear Dispatch aggregation separately to the corresponding category-weighted feature contributions before their frame-level moments are formed. This preserves the actual expert input coordinate system and the target/background accounting; raw $H_t$ covariance is not substituted for expert input covariance. For each selected expert, build $C_{adapt,e}$ and $C_{protect,e}$ from those admitted $\xi_{t,e}$ contributions, construct $S_{t,e}$ by the same whitening procedure, and apply:

$$
\widetilde G_{A_e}=S_{t,e}G_{A_e},
$$

because $A_e=[384,4]$. Reusing the router operator $S_t$ for an expert is permitted only as a named approximation ablation; it is not described as expert-input curvature. $B_e=[4,384]$ needs a separate rank-4 hidden covariance or a simple trust-region update. The first Stage E implementation uses a trust region for $B_e$ rather than pretending the 384-dimensional operator applies to it.

### 5.6 Parameter stages

#### Stage 0: spectral-memory only

All model parameters are frozen. This stage tests whether historical eigenspaces improve routing beyond matched confidence memory.

#### Stage R: router-only

Only selected HMoE `gate_thi` parameters update. Freeze:

- `D_temp` and `C_temp`;
- HMoE experts and biases;
- `linear1`, `linear2`, and norms;
- backbone, head, and qkv LoRA.

The pilot spectral-layer set is configurable and defaults to blocks 5 and 9, with attention and FFN HMoE included. Formal layer ablations cover block 5, block 9, blocks 5+9, and all HMoE layers.

Stage R uses a causal non-queueing scheduler. If the current frame passes $A_t^{grad}=1$ and at least five frame indices have elapsed since the last committed optimizer step, update immediately from the current frame; otherwise discard that frame's parameter-update proposal. Proposals are never queued, ranked against future frames, or replayed later. This minimum interval is fixed on calibration data and included in all matched online baselines.

The normative online optimizer is momentum-free, weight-decay-free SGD. The controller always owns an AMP scaler, disabled as a no-op when AMP is off, and calls `scaler.unscale_(online_optimizer)` exactly once before gradient projection, slice masking, or finite-value checks. Gradient clipping is disabled in the core spectral method, so the realized router step is $\Delta W_g=-\eta S_tG_g$. The gradient-clipping control is a separate baseline whose realized step explicitly includes its registered clipping operator. AdamW and any optimizer with decoupled weight decay or persistent momentum are forbidden for stacked online parameters in the core experiment. A later adaptive-optimizer ablation must call this operation gradient shaping, not update projection, and must audit the realized $\Delta\Theta$ rather than infer it from the shaped gradient.

#### Stage E: selected HMoE LoRA experts

Stage E is enabled only after Stage R passes held-out gates. For each active HMoE, score experts on the current committed forward as $u_e^{target}-\lambda_Au_e^{anchor}$, where both utilizations are detached target-token Combine means and $\lambda_A$ is frozen in the registry. Select the highest-scoring one of four attention experts and the highest-scoring two of eight FFN experts; exact ties choose the lower expert index. When five router steps have committed since the last expert step, update those current-frame experts immediately. Otherwise discard the expert proposal without queueing it. Router learning rate is reduced when Stage E starts. These budgets and rules are matched in every selected-expert baseline.

Unselected expert values and optimizer state must remain bitwise unchanged.

Stage E uses the same momentum-free, weight-decay-free SGD rule. Once Stage E is enabled, the eigenspace bank maintains bounded keys `(block, attn_or_ffn, expert)` for every expert in the active blocks, updating their admitted input factors whether or not that expert is selected on the current step. Their four factor families, immutable identity anchors, and all temporary tensors count toward the memory budget. After AMP unscale, the controller applies expert-specific $S_{t,e}$ only to selected $A_e$ slices. For a selected $B_e$, use the registered Frobenius trust region:

$$
\widetilde G_{B_e}=G_{B_e}
\min\left(1,\frac{\tau_B}{\|G_{B_e}\|_F+\epsilon}\right),
\qquad
\Delta B_e=-\eta_B\widetilde G_{B_e}.
$$

All other slices are masked before `step()`. A zero masked gradient must produce bitwise-identical unselected parameter values; there is no optimizer state for those slices beyond the current gradient.

#### Excluded stage

qkv `MergedLinear` LoRA is outside the first design. The final inference qkv merge state is established once after checkpoint loading. Committed and differentiable online forwards both keep the whole tracker in `model.eval()`; autograd is enabled locally without calling `model.train()`, and train/eval mode is never toggled during an episode. qkv/base tensors are excluded from the online optimizer and hashed before and after every diagnostic episode. Canonical merge/unmerge conversion is a separate future experiment and requires independent serialization tests.

### 5.7 Transaction and rollback

Before every memory or parameter update, prepare a transaction containing:

- candidate spectral/replay delta;
- student parameters to be changed;
- optimizer state;
- EMA teacher;
- last-safe metadata.

Apply the candidate update tentatively. Audit the immutable anchor and a small trusted replay sample for:

- finite loss, gradients, parameters, and spectral factors;
- anchor prediction degradation;
- replay loss increase;
- target-token route KL;
- parameter and gradient norms;
- expert-load discontinuity;
- invalid or exploding box scale.

If an audit fails, atomically restore all parts of the transaction and enter a ten-frame cooldown. The EMA teacher changes only after a safe commit:

$$
\bar\theta\leftarrow\mu\bar\theta+(1-\mu)\theta.
$$

If later frames produce $K_{bad}=3$ consecutive low-reliability or high-drift events, restore the last safe snapshot. If that does not recover, restore the sequence-start checkpoint. The cooldown and $K_{bad}$ defaults are fixed before test and receive a validation-only sensitivity analysis.

Two explicit lifecycle modes are required:

- Standard OPE and every official restart call `begin_episode(reset_global=True)`, restoring router/LoRA from the immutable base checkpoint and resetting optimizer, EMA, snapshots, target memories, replay, hysteresis, and cooldown.
- The dedicated cross-sequence stress runner calls `begin_episode(reset_global=False)`, retaining carried router/LoRA, optimizer, and EMA while resetting every object-specific memory, replay unit, reliability state, and cooldown. Its rollback baseline is the carry-in snapshot, not the original base checkpoint.

Standard dataset runners may never select carry mode.

## 6. Offline training data flow

### 6.1 Checkpoint families

Two checkpoint families are required.

#### Modality-specific OPE checkpoints

- RGB-T: LasHeR training sequences only;
- RGB-D: DepthTrack training sequences only;
- RGB-E: VisEvent training sequences only, with a fixed, published sequence-level partition because the current loader has no validation split.

Before any experiment, each modality's non-test sequences are partitioned by committed, sequence-disjoint fit, calibration, and locked gate-confirmation manifests. VisEvent uses a fixed 90/10 development/gate-confirmation sequence-level manifest, never a runtime random split; the 90% development pool has its own committed fit/calibration partition. The manifests, split seeds, dataset hashes, checkpoint-selection metric, and freeze date are versioned. Model and estimator parameters are learned only on fit sequences. Checkpoint selection, temperature calibration, coefficients, and thresholds use only calibration sequences. Gate-confirmation sequences are opened only for the predeclared M1/S0/R1/M2--M4 decisions; nothing is retuned there, and all choices are frozen before any benchmark test is run.

All matched methods use identical base weights, sample count, optimizer steps, resolution, augmentations, checkpoint-selection rule, and three independently trained base-checkpoint seeds. Online randomness is nested within each base seed and reported separately; rerunning one base checkpoint with three online seeds is not called three training seeds. Confidence intervals use paired hierarchical bootstrap, resampling training seeds and then sequences within seed.

#### Joint RGB-X checkpoint

A joint T/D/E model is trained only for unified and cross-sequence stress experiments. T, D, and E are sampled with equal modality probability. Every baseline receives the same total optimizer steps and clip budget. Modality-specific checkpoints are the primary OPE models and are never chosen per test benchmark. The joint checkpoint is reported as a separate unified setting, not used opportunistically when it performs better on an individual dataset.

### 6.2 Ordered chronological clips

The current sampler's `causal` mode randomly chooses future search frames and does not provide an ordered online stream. Stage-II training therefore requires a separate chronological clip sampler.

The dedicated rollout sampler draws one sequence, uses its first frame as $z_0$, and selects the strictly ordered, distinct clip:

$$
[z_0,x_{t-2},x_{t-1},x_t,x_{t+1}].
$$

- $z_0$ initializes the anchor;
- $x_{t-2},x_{t-1}$ build detached history;
- $x_t$ creates a candidate memory/parameter update;
- $x_{t+1}$ supplies only the offline counterfactual outcome and outer tracking loss.

The sampler performs no visibility filtering, replacement sampling, or gap expansion. Occluded and invisible frames remain in temporal order; visibility is an offline label, never an input to the rollout. State resets at every clip boundary.

The dedicated rollout processor must reproduce inference geometry. After initialization, every search crop is produced by `sample_target` around the previous committed predicted state, including scale, rather than around the current GT box. GT may form the outer tracking loss and counterfactual labels only; it may not determine crop center/scale, reliability observables, memory admission, target/background weights, or any online inner-loss input. Unit tests cover strict ordering, uniqueness, invisible-frame retention, absence of gap expansion, and prediction-centered crops under deliberately offset predictions.

### 6.3 Training phases

1. Train the base MoE-LoRA tracker with the existing tracking objective.
2. Roll chronological clips through a frozen base tracker to generate reliability observables and counterfactual labels.
3. Train the dual-reliability estimator on fit sequences and temperature-calibrate it on the disjoint calibration manifest.
4. Train the bounded forward-routing coefficients and online controller with the test-time inner objective and a next-frame outer tracking objective.
5. Start with oracle admission to establish the mechanism ceiling, then replace it with estimated admission.

No test sequence, test attribute, test corruption boundary, or future test frame is used for training or threshold selection.

### 6.4 Corruption curriculum

Training clips include clean data and temporally coherent corruption bursts:

- RGB: low light, gamma, blur, Gaussian/shot noise, JPEG, blackout;
- thermal: crossover, saturation, dead pixels, blur, blackout;
- depth: holes, quantization, scale clipping, speckle, edge misregistration, blackout;
- event: dropout, background activity, hot pixels, time-bin mismatch, polarity corruption;
- paired: spatial misregistration, temporal delay, intermittent single-modality loss, and dual-modality degradation.

Corruption identity is available only for training augmentation and analysis. It is not an input at test time.

## 7. Evaluation design

### 7.1 Preregistered core scope

The preregistered core is Stage 0 plus Stage R on one RGB-T pair (LasHeR fit/calibration/gate-confirmation manifests and its official test protocol) and one RGB-D pair (DepthTrack fit/calibration/gate-confirmation manifests and its official test protocol), using HMoE blocks 5 and 9. Its locked controls are frozen, naive router update, random orthogonal basis, pooled SAME-style covariance, confidence-only bounded memory, a validation-only oracle schedule, and estimated reliability. This is the minimum experiment capable of testing the routing, drift, spectral-shaping, and admission claims.

The scalar core endpoint is fixed before runs as:

$$
J_{core}=\tfrac12(J_{LasHeR}+J_{DepthTrack}),
$$

where each $J$ is that benchmark's single preregistered official primary metric expressed in percentage points. The exact toolkit version, metric field, direction, and aggregation are committed in the registry; secondary metrics cannot replace either term after results are seen.

Stage E, the complete corruption suite, joint T/D/E training, the COESOT adapter, external-method reimplementations, and cross-sequence continual streams are conditional extensions. They begin only after their prerequisite gates pass and cannot be used to obscure a failed core mechanism.

### 7.2 Standard sequence-reset OPE is primary

At every sequence start, reset:

- router and LoRA to the same base checkpoint;
- optimizer, EMA, snapshots, and cooldown;
- target spectral memories and replay.

Under OPE, only the first-frame ground-truth box is visible and initialization occurs exactly once per sequence.

Primary and extension datasets are:

- RGB-T: LasHeR, RGBT234, VTUAV-ST, VTUAV-LT, with GTOT as extension;
- RGB-D: DepthTrack and CDTB; VOT-RGBD is reported separately under its official restart protocol;
- RGB-E: VisEvent, with COESOT after a dataset adapter is added and validated.

The main paper must show gains on at least two X modalities. LasHeR-only success is insufficient.

VOT restart evaluation is never pooled with OPE. The official restart box is the sole post-initialization GT exception: every official restart starts a new episode and fully resets router/LoRA, optimizer, EMA, snapshots, memories, replay, reliability hysteresis, and cooldown before using the restart initialization box.

Report each benchmark's official metrics. Also report three-seed mean/std and paired hierarchical per-sequence bootstrap 95% confidence intervals.

### 7.3 Modality-failure and recovery protocol

Corruptions occur in contiguous bursts. Validation fixes three severity levels that make frozen SEATrack drop approximately 5, 10, and 20 percentage points. Test severity is not retuned.

Define:

- failure: IoU below 0.1 for five consecutive frames;
- recovery: IoU at least 0.5 for five consecutive frames.

Report:

- failure count;
- median recovery frames;
- success 10, 30, and 100 frames after a burst;
- degradation-and-recovery AUC;
- false memory admission and rollback count.

Recovery is a time-to-event endpoint. Sequences that do not recover before the evaluation horizon are retained as right-censored observations; they are never dropped from a median. Report the Kaplan--Meier recovery curve and restricted mean recovery time over the fixed horizon. Report median recovery only when the estimated curve reaches 50%, together with the unrecovered fraction.

### 7.4 Secondary cross-sequence continual stream

This protocol evaluates deployment safety and is not mixed with OPE results.

- Target-specific $C_{id/private/dyn/bg}$ and template replay reset for every new object.
- Router/LoRA and optimizer may carry across sequences.
- No old raw frames carry across sequences.
- Evaluate all six T/D/E block permutations and a task-free interleaved stream.
- Use at least three sequence-order seeds.
- Adapt streams and probe sets have no sequence overlap.

In block mode, modality labels are used only by the evaluator to construct the six orders and are never inputs to the tracker. In task-free mode, T/D/E sequences are interleaved by a committed seed; the tracker receives neither task/modality identity nor a boundary flag, object-specific state resets at each ordinary sequence initialization, and only the declared global router/LoRA/optimizer/EMA state carries. Probe checkpoints occur after predetermined sequence counts rather than detected task changes.

After each block or predetermined task-free checkpoint, clone the carried global parameters. For every fixed probe sequence, fully reset object-specific state, enable the ordinary causal spectral memory, and disable all parameter/optimizer updates; this single `probe-memory-on/updates-off` mode is used at every stream stage and appears in metric names. An `all-adaptive-state-frozen` probe is a separate ablation and cannot replace the normative mode. Report average accuracy, backward transfer, average forgetting, worst-stage accuracy, and order standard deviation.

Because the approved core method has no separate global cross-sequence spectrum, this is a stress analysis rather than a guaranteed forgetting contribution. It becomes a claimed contribution only if it reduces forgetting relative to matched naive online LoRA by at least 25%.

### 7.5 Matched baselines

The mechanism-isolation tables match base checkpoint, trainable parameters, optimizer, update frequency, accepted-frame schedule, online loss, memory bytes, and replay capacity. Each such table uses either a fixed GT-free periodic schedule or one pre-recorded validation-calibrated causal schedule shared by every row; the schedule and accepted indices are published, and no compared method may suppress a difficult update:

1. original released frozen SEATrack checkpoint as an external reference;
2. matched retrained frozen SEATrack MoE-LoRA base, with instrumentation and routing strength disabled, used to initialize every mechanism row;
3. naive router-only update;
4. naive LoRA-only update;
5. naive router+LoRA update;
6. small learning rate / gradient clipping;
7. random orthogonal basis;
8. pooled SAME-style accumulated covariance;
9. target/template/search-balanced covariance;
10. EMoE-style current-input projection-energy routing;
11. confidence-only bounded temporal memory;
12. full four-spectrum method under the same diagnostic schedule.

The deployable table compares confidence-only causal admission and the full method with estimated dual reliability. Each row uses its own causal admission policy and reports memory-write coverage, proposed/accepted/rejected update counts, rollback rate, latency, and state bytes. These results are not presented as if they shared the same accepted frames.

The validation-only oracle-ceiling table uses the common post-prediction GT schedule defined by M3. It includes random, pooled, confidence, and four-spectrum controls and is never mixed with deployable test results.

Faithful SEATrack adaptations of DTPTrack-style reliability memory, SPMTrack-style temporal references, PURA-style update recovery, and GOLA-style LoRA rank constraints belong in component-matched comparisons only under a protocol written before their results are inspected: public implementation or fully specified equations, frozen adaptation mapping into SEATrack, matched parameter/update/memory budget, and named success/failure criteria. Integration failures are reported rather than silently dropping an unfavorable baseline. Native paper results remain in a separate external-reference table and are never presented as matched numbers.

### 7.6 Normative registry and freezing policy

Before gate-confirmation runs, commit one machine-readable registry containing:

- reliability-estimator architecture and parameter count;
- $\beta$, $\lambda_0$, allowed ranks, minimum effective mass, and eigengap rule;
- $\alpha_{budget}$, the four-scalar parameterization/checkpoint rule, $\kappa$, $\rho$ construction, $s_{min}^{w}$, $s_{max}^{w}$, $s_{raw}$, and residual-logit clip;
- $\tau_M$, $\tau_U$, anchor/cycle/augmentation thresholds, and calibration method;
- augmentation cadence and the required candidate-cycle procedure and latency budget;
- memory-write cadence, optimizer-step cadence, selected-expert counts, $\lambda_A$, and expert tie-break rule;
- replay-audit sample size and replay byte limit;
- cooldown, $K_{bad}$, rollback tolerances, and snapshot cadence;
- online learning rates, EMA coefficient, $\tau_B$, AMP policy, and finite-value tolerances;
- default values, permitted fit/calibration search domains, selection metric, split manifests, code/checkpoint hashes, named timing hardware, and freeze date;
- exact $J_{LasHeR}$/$J_{DepthTrack}$ toolkit versions and metric fields, $J_{core}$ construction, clean endpoint, recovery horizon, and shuffle-attenuation computation.

Gate-confirmation data can accept or reject the frozen registry but cannot retune it. Test benchmarks are never used for model, threshold, rank, checkpoint, or corruption-severity selection. Any post-freeze change creates a new registry version and reruns all affected controls.

### 7.7 Metrics

#### Tracking

- official benchmark metrics;
- paired hierarchical-bootstrap 95% confidence intervals;
- three-seed mean/std;
- failure and recovery metrics.

#### Router drift

- target-token Combine JSD;
- top-1/top-2 expert overlap;
- target-token logit $L_2$ drift;
- matched-slot Dispatch drift as a secondary measure;
- expert-load entropy and standard deviation;
- correlation between route drift and tracking degradation.

#### Basis drift

- projector chordal distance:
  $$
  \frac{\|P_t-P_{t-1}\|_F}{\sqrt{2k}};
  $$
- principal angles and maximum angle;
- retained energy and effective rank;
- subspace overlap;
- basis update norm during low-confidence bursts.

#### Reliability

- AUROC and AUPRC;
- ECE and Brier score;
- risk-coverage curve;
- memory contamination;
- false admission/rejection;
- counterfactual update-benefit precision;
- accepted update rate.

#### Efficiency

- frozen forward FPS;
- amortized end-to-end FPS including online backward, SVD, replay, and rollback;
- p50 and p95 latency;
- accepted-update latency;
- peak allocated/reserved GPU memory;
- CPU replay bytes and persistent spectral bytes;
- trainable parameters and actual optimizer-state bytes.

Timing uses explicit CUDA synchronization around the measured region on the named hardware at batch size 1 with the locked admission policy. Results state the number and coverage of proposed and accepted updates per sequence. A method cannot satisfy the FPS gate by rejecting almost all updates: efficiency is reported as a coverage--latency--accuracy Pareto curve, and the headline point must meet the frozen minimum admission coverage. Memory includes every spectral buffer, replay unit, EMA copy, snapshot, optimizer/scaler tensor, and temporary audit allocation.

## 8. Predeclared mechanism gates

### M0: zero-behavior instrumentation

- instrumentation disabled, empty state, or routing strength zero is output-identical to legacy HMoE;
- only explicit state buffers may change;
- template/search, RGB/X, target/background, block, and attention/FFN statistics remain separable;
- DDP and single-GPU projectors differ by less than $10^{-4}$ relative error on matched data.

Failure stops the experiment.

### M1: low-rank state semantics

- choose the smallest common rank in $\{8,16,32\}$ that captures at least 90% trace energy for every stored factor family/block on the calibration diagnostic set; if none passes, M1 fails rather than expanding the search; lock that rank before gate confirmation, and M1 passes only if the same locked rank also captures at least 90% for every family/block there without retuning;
- incremental aggregation is invariant to batch partition within numerical tolerance;
- zero-admission updates leave factors and effective mass bitwise unchanged;
- retained eigenvalues are sorted and nonnegative, projectors are finite, and basis orthogonality error is within the registered tolerance.

### R1: online optimizer semantics after S0

- the realized router gradient is left-shaped and finite after AMP unscale;
- the realized SGD step matches $-\eta S_tG_g$ within numerical tolerance;
- frozen values and optimizer moments are bitwise unchanged;
- predicted protected drift $\operatorname{tr}(\Delta\Theta^\top C_{protect}\Delta\Theta)$ correlates with measured anchor-logit drift at Spearman $\rho\ge0.5$.

### S0: state-only forward-routing value

On locked gate-confirmation sequences, freeze all model parameters and apply one pre-recorded GT-free causal memory-admission schedule to confidence-only bounded memory, shuffled-pair/random-projector memory, target-balanced identity memory, and the full four-spectrum forward router. The schedule is generated causally by the confidence control with thresholds frozen on calibration, then replayed identically for every row. Oracle admission is excluded from S0 and appears only in the validation-only M3 ceiling.

Before any online optimizer is introduced, the full state-only router must satisfy at least one preregistered co-primary condition relative to confidence-only memory, with one-sided paired hierarchical-bootstrap 97.5% lower confidence bounds, while reducing neither $J_{LasHeR}$ nor $J_{DepthTrack}$ by more than 0.3 percentage points:

- improve $J_{core}$ by at least 0.3 percentage points; or
- reduce restricted mean recovery time by at least 10% under the single corruption burst, severity, and horizon named in the registry.

Let $G_{full}>0$ be the signed improvement on the passing endpoint, preferring $J_{core}$ if both pass. For each of RGB-X pair shuffling, temporal-order shuffling, and target/background-mask shuffling, compute its identically signed gain $G_k$ over confidence-only memory. At least two controls and their mean must satisfy $G_k\le0.5G_{full}$. Otherwise history-conditioned forward routing is not identified and the online-update workstream stops.

### M2: router drift exists

With naive router update and fixed experts/backbone/input projection, every method uses the same 50 attempted indices from a GT-free periodic diagnostic schedule: after a five-frame burn-in, attempt an update every fifth frame on each locked long-sequence stream. M2 disables performance-based rollback and reliability selection; only a nonfinite update may abort the diagnostic, and such an abort fails M2 rather than extending the stream to obtain 50 safer commits. All methods therefore have the same attempts and time horizon. After the fiftieth attempted index, at least one condition holds:

- target-token Combine JSD increases by at least 0.02;
- top-1 overlap drops by at least five percentage points.

Define later decline as frozen-baseline IoU minus updated-tracker IoU, so larger values mean worse tracking. Route drift must have the predeclared positive Spearman correlation $\rho\ge0.3$ with later decline and recur in at least two datasets or corruption conditions. Correlation is computed from per-sequence drift and later-decline summaries, with sequence-level bootstrap intervals; autocorrelated frames are not treated as independent samples. Otherwise the spectral-consolidation causal story stops.

### M3: oracle spectral mechanism

M3 is run only on the locked, sequence-disjoint gate-confirmation diagnostic. After each frame's prediction has been committed, the evaluator constructs one common GT-derived update schedule and provides only the resulting update/no-update decision to the diagnostic controller. The tracker, reliability estimator, memory factors, and inner loss never receive GT or GT-derived magnitudes. Candidate re-forwards never replace the already committed prediction for frame $t$.

Relative to random, pooled SAME, and confidence controls under that identical schedule, the four-spectrum method must:

- reduce anchor router drift by at least 20%;
- improve the preregistered $J_{core}$ endpoint by at least 0.5 percentage points;
- lose no more than 0.3 percentage points against naive router update on the accepted frames' next-frame committed tracking metric.

Primary M3 outcomes are the next-frame and full committed-stream metrics. GT scheduling is a validation-only oracle diagnostic, never a deployable method and never a test-table row.

### M4: estimated reliability

- report positive/negative class counts, AUROC, AUPRC, ECE, Brier score, and risk--coverage separately for $q^{mem}$ and $q^{upd}$;
- both tasks have AUROC at least 0.75 and ECE at most 0.10 on gate confirmation;
- admitted-memory contamination is at most 10% at or above the frozen minimum coverage;
- let $J_c$ be the strongest matched M3 control, $J_o$ the four-spectrum oracle result, and $J_e$ the estimated-reliability result on the same higher-is-better committed-stream metric; require $(J_e-J_c)/(J_o-J_c)\ge0.5$.

The retained-gain ratio is undefined, and M4 cannot pass, if M3 fails or $J_o\le J_c$. Lower-is-better metrics are sign-normalized before applying the same control-subtracted definition.

Failure returns the paper to a memory-only result; it cannot claim safe continual parameter learning.

## 9. Final go/no-go criteria

The project expands into a top-conference submission only if all conditions hold:

- significant gains on at least two RGB-X modality types;
- at least 0.5 percentage-point gain on one standard in-domain benchmark and one long-term/stress benchmark, with paired hierarchical-bootstrap 95% lower bounds above zero;
- no clean primary benchmark drops by more than 0.3 percentage points;
- full spectrum beats random basis, pooled SAME, and confidence-only memory under matched budgets;
- M4 reliability gates pass;
- amortized FPS is at least 80% of frozen baseline on the preregistered hardware, batch size 1, and locked policy while meeting the frozen minimum admission coverage;
- peak GPU memory is at most 1.25 times frozen baseline;
- persistent spectral state is at most 8 MiB;
- all replay and persistent-state bytes are reported and matched in memory baselines;
- if cross-sequence forgetting is claimed, forgetting falls by at least 25% relative to matched naive online LoRA.

If oracle spectral routing fails M3, stop the spectral main line rather than tuning the reliability estimator. If oracle works and estimated reliability fails M4, the bottleneck is reliability; report that distinction explicitly.

## 10. Invariants and failure handling

The implementation must preserve these invariants:

- no current response affects its own first prediction;
- no post-initialization test GT, future frame, attribute, or corruption boundary is consumed, except the official initialization box after a VOT restart, which starts a fully reset new episode;
- `track()` rejects forbidden evaluator keys including GT boxes, visibility, attributes, corruption identity, and corruption boundaries;
- empty spectral state produces legacy output;
- all stored features, pseudo-labels, responses, and replay targets are detached;
- memory rank, replay count, and update rate are bounded;
- `linear1` remains frozen online;
- template anchor never decays;
- low-confidence frames write neither memory nor gradients;
- one-modality failure does not contaminate paired identity/private states;
- rollback restores parameters, optimizer, EMA, and memory atomically;
- every ordinary sequence initialization resets all object-specific state; standard OPE also resets global online state, while only the dedicated cross-sequence runner may retain the declared global state;
- both committed and differentiable online forwards stay in evaluation mode, and frozen qkv/base hashes remain unchanged;
- current dense HMoE remains dense and makes no sparse-speed claim.

Expected failure cases and responses:

- occlusion/target absence: reject memory and update, then rollback after persistent anomalies;
- single-modality failure: route using the last committed asymmetry but write none of the paired spectra in the first implementation;
- dual-modality failure: freeze all adaptation;
- confident distractor drift: anchor/cycle veto and rollback;
- rapid real deformation: admit dynamic memory before allowing parameter updates;
- repeated or flat spectrum: keep the projector inactive until eigengap/effective-sample gates pass;
- modality misregistration: match common indices, reject low-alignment pairs, and fail the common/private gate if shuffled controls are indistinguishable;
- route saturation: clip residual logits and roll back;
- selected-slice optimizer leakage: reject the update if frozen slices or moments change.

## 11. Scope exclusions

The first implementation does not include:

- qkv MergedLinear LoRA online updates;
- top-k expert execution or sparse speed claims;
- unbounded raw-frame replay;
- a learned online RGB-X alignment map;
- a global cross-sequence spectral memory;
- simultaneous GRA, BiLift, ProbAlign, LiftTrack, and spectral adaptations;
- default entropy/load-balancing losses without collapse evidence;
- claims that spectral routing fixes the shared rank-4 bottleneck.

These exclusions isolate the causal mechanism and keep the first implementation plan reviewable.

## 12. Implementation decomposition

This design is a gated research program, not one atomic implementation task. It is decomposed into three independently reviewed subprojects:

### Workstream A: observability and Stage 0

Scope:

- paired spectral observation;
- bounded eigenspace bank;
- history-conditioned forward routing;
- ordered chronological clips sufficient to fit bounded routing coefficients and replay sequence-local state;
- output-identity, low-rank, and drift diagnostics;
- frozen-parameter spectral-memory experiment through S0.

The first implementation plan covers only Workstream A. It contains no test-time optimizer and no reliability-network training; S0 uses only the common recorded GT-free causal diagnostic schedule.

### Workstream B: oracle router adaptation, dual reliability, and Stage R

Prerequisite: Workstream A passes S0.

Scope:

- online optimizer semantics through R1;
- naive and spectrally shaped oracle router-update diagnostics for M2/M3;
- extension of chronological clips for counterfactual update labels and corruption bursts;
- counterfactual reliability labels and calibration;
- weak/strong online objective and replay;
- transactional router-only updates, reset, and rollback;
- M4 and matched router-update experiments.

Workstream B receives its own implementation plan after S0 evidence is reviewed. Within Workstream B, reliability training does not start unless M2 establishes harmful router drift and M3 establishes an oracle spectral advantage.

### Workstream C: Stage E and full evaluation

Prerequisite: Workstream B passes M4 and the clean-performance/runtime gates.

Scope:

- selected HMoE LoRA optimizer semantics;
- expert-state bitwise-freeze validation;
- modality-specific OPE, corruption recovery, and optional cross-sequence stress;
- paper-scale ablations and efficiency accounting.

Workstream C receives its own implementation plan after Stage R evidence is reviewed. A failed gate ends or narrows the later workstream; it is not bypassed by adding more modules.

## 13. Verification strategy

Verification must proceed in this order:

1. shape, state, and output-identity tests;
2. full-matrix versus incremental low-rank numerical tests;
3. causal frozen-parameter forward-routing integration on a short sequence;
4. S0 state-only forward-routing experiment;
5. R1 single-step gradient orientation, realized-step, and frozen-state tests;
6. M2 router-drift diagnostic;
7. M3 oracle mechanism experiment;
8. transaction commit/rollback and hard-reset tests;
9. M4 reliability calibration experiment;
10. full modality-specific OPE;
11. long-term/corruption recovery;
12. optional cross-sequence continual stress.

No later stage starts if its prerequisite gate fails.
