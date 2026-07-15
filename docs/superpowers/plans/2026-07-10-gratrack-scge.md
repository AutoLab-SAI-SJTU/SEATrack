# GRATrack SCGE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and experimentally validate a clean dual-stream OSTrack variant with single-pass counterfactual gain estimation, factorized four-action routing, low-rank directional adapters, and sparse hard inference.

**Architecture:** The new path disables legacy AMG-LoRA attention exchange and dense HMoE completely. Three post-block sidecars at layers 3, 7, and 11 apply always-on private low-rank adapters and two hard-routed directional cross adapters; one ordinary task backward supplies detached first-order gain targets, while every tenth training step computes exact four-action tracking losses at the final routing layer for calibration.

**Tech Stack:** Python 3.12, PyTorch 2.12, unittest, YAML, existing OSTrack/SEATrack training and Center Head infrastructure.

## Global Constraints

- Preserve all existing user changes; do not revert unrelated files.
- Work on branch `gratrack-scge-experiment`, not `main`.
- New counterfactual runs must instantiate neither legacy HMoE nor AMG-LoRA `MergedLinear` modules.
- No OT, Sinkhorn, Wasserstein, transport, ProbAlign, old GRA, or RGAE term may enter the new forward or loss path.
- Existing experiment YAML files remain behavior-compatible; new behavior is opt-in.
- Use test-first development for every production behavior.
- Do not launch a full 60-epoch run until unit tests, synthetic forward/backward, one-sample real-data smoke, and short diagnostic gates pass.
- Because the shared worktree already contains uncommitted user work, do not create commits that would capture unrelated modifications.

---

### Task 1: Training Integrity Guards

**Files:**
- Modify: `lib/train/data/transforms.py`
- Modify: `lib/train/actors/seatrack.py`
- Create: `tests/test_training_integrity.py`

**Interfaces:**
- `ToGrayscale.transform_image(image, do_grayscale)` accepts `H x W x 3` and `H x W x 6` NumPy arrays.
- `SEATrackActor.compute_losses()` propagates invalid-box/GIoU failures instead of replacing them with CUDA zero tensors.

- [ ] **Step 1: Write a failing six-channel grayscale test**

```python
def test_grayscale_converts_each_modality_without_dropping_channels(self):
    image = np.zeros((2, 2, 6), dtype=np.uint8)
    image[..., :3] = np.array([10, 20, 30], dtype=np.uint8)
    image[..., 3:] = np.array([90, 60, 30], dtype=np.uint8)
    result = ToGrayscale(probability=1.0).transform_image(image, True)
    self.assertEqual(result.shape, image.shape)
    np.testing.assert_array_equal(result[..., 0], result[..., 1])
    np.testing.assert_array_equal(result[..., 3], result[..., 4])
```

- [ ] **Step 2: Run the test and verify the current OpenCV channel error**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_training_integrity -v`

Expected: FAIL because `cv.cvtColor(..., COLOR_RGB2GRAY)` rejects six channels.

- [ ] **Step 3: Convert RGB triplets independently and reject unsupported channel counts**

```python
if image.ndim != 3 or image.shape[2] not in (3, 6):
    raise ValueError("ToGrayscale expects an HxWx3 or HxWx6 image")
groups = [image[..., index:index + 3] for index in range(0, image.shape[2], 3)]
gray_groups = [np.repeat(cv.cvtColor(group, cv.COLOR_RGB2GRAY)[..., None], 3, axis=2)
               for group in groups]
return np.concatenate(gray_groups, axis=2)
```

- [ ] **Step 4: Write and verify a failing actor test for silent GIoU recovery**

Use a deterministic objective whose `giou` callable raises `AssertionError`; assert that `compute_losses()` raises the same error. The current test must fail because the actor swallows it.

- [ ] **Step 5: Remove the bare GIoU exception and run integrity tests**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_training_integrity -v`

Expected: PASS.

### Task 2: Counterfactual Gain Sidecar

**Files:**
- Create: `lib/models/layers/counterfactual_gain.py`
- Create: `tests/test_counterfactual_gain.py`

**Interfaces:**
- `factorized_action_scores(gains: Tensor) -> Tensor`, mapping `[q_x2r, q_r2x, q_interaction]` to `[Q00, Q10, Q01, Q11]`.
- `LowRankResidualAdapter(dim: int, rank: int, dropout: float)` returns a zero-initialized residual at construction.
- `CounterfactualGainSidecar.forward(rgb, x, lens_t, collect_exact=False)` returns routed RGB/X tensors and stores detached diagnostics.
- `CounterfactualGainSidecar.router_loss_from_gate_grads(grad_scale=1.0)` returns a fresh router-only graph and scalar diagnostics after task backward.

- [ ] **Step 1: Write failing action-factorization and zero-initialization tests**

```python
gains = torch.tensor([[2.0, -1.0, 0.5]])
self.assertTrue(torch.equal(factorized_action_scores(gains),
                            torch.tensor([[0.0, 2.0, -1.0, 1.5]])))
self.assertTrue(torch.equal(adapter(torch.randn(2, 5, 16)), torch.zeros(2, 5, 16)))
```

- [ ] **Step 2: Run tests and verify imports fail because the module is absent**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_counterfactual_gain -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement low-rank adapters and the factorized router**

Use parameter-free layer normalization, `Linear(dim, rank, bias=False)`, GELU, dropout, and a zero-initialized `Linear(rank, dim, bias=False)`. Build action scores as:

```python
q_x2r, q_r2x, q_interaction = gains.unbind(dim=-1)
return torch.stack((torch.zeros_like(q_x2r), q_x2r, q_r2x,
                    q_x2r + q_r2x + q_interaction), dim=-1)
```

- [ ] **Step 4: Write failing routing tests**

Verify all four forced actions map to gate bits `00`, `10`, `01`, `11`; eval routing is deterministic; training routing is one-hot; and `collect_exact=True` returns four algebraic candidates without four backbone calls.

- [ ] **Step 5: Implement hard routing and action-grouped sparse inference**

Training uses hard Gumbel softmax for exploration but detaches the selected gate into per-sample leaf tensors so task loss does not directly train the router. Evaluation uses argmax and computes only adapters needed by non-empty action groups, followed by indexed scatter.

- [ ] **Step 6: Write a failing Taylor-target test**

Construct a scalar loss from routed outputs, call backward, and require finite gain targets for both directional gates. Confirm the target sign is `-dL/dg` and AMP scale division recovers the unscaled value.

- [ ] **Step 7: Implement detached Taylor targets and router-only regression**

Cache detached summaries during forward. After task backward, normalize `-gate.grad / grad_scale` by detached mean absolute magnitude, supervise `q_x2r/q_r2x` with Smooth L1, and supervise `q_interaction` only when exact labels are present.

- [ ] **Step 8: Run sidecar tests**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_counterfactual_gain -v`

Expected: PASS with finite gradients for private adapters, selected cross adapters, and router parameters.

### Task 3: Clean OSTrack Integration

**Files:**
- Modify: `lib/config/seatrack/config.py`
- Modify: `lib/models/layers/attn_blocks.py`
- Modify: `lib/models/seatrack/vit_ci.py`
- Modify: `lib/models/seatrack/seatrack.py`
- Modify: `lib/train/base_functions.py`
- Create: `tests/test_counterfactual_integration.py`

**Interfaces:**
- `MODEL.LEGACY_SEA_ENABLED: bool` defaults to `True` for old YAML compatibility.
- `MODEL.COUNTERFACTUAL.{ENABLED,LAYERS,RANK,ROUTER_DIM,TEMPERATURE,DROPOUT,FUSION_MAX_DELTA}` configures the new path.
- `VisionTransformerCE.collect_counterfactual_router_loss(grad_scale)` aggregates sidecar losses and statistics.

- [ ] **Step 1: Write failing clean-path construction tests**

Instantiate a small `CEBlock_AP` with counterfactual routing enabled and assert:

```python
self.assertFalse(hasattr(block, "attn_moe"))
self.assertFalse(hasattr(block, "ffn_moe"))
self.assertIsInstance(block.attn.qkv, nn.Linear)
self.assertIsNotNone(block.counterfactual_gain)
```

- [ ] **Step 2: Verify the tests fail because no clean/new path exists**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_counterfactual_integration -v`

Expected: FAIL on unknown constructor/config fields.

- [ ] **Step 3: Add opt-in config and propagate it through the model builder**

Reject `LEGACY_SEA_ENABLED=True` together with `COUNTERFACTUAL.ENABLED=True`. For counterfactual or clean dual-OSTrack modes, pass empty legacy LoRA/MoE layer lists into every block.

- [ ] **Step 4: Insert sidecars after each selected block MLP**

The normal attention and MLP execute independently for RGB and X. Selected layers then run private residuals and routed directional residuals. Existing GRA/RGAE code remains available only to legacy configurations.

- [ ] **Step 5: Replace fixed final sum only in the new path**

Use the last action to preserve total feature scale:

```python
strength = fusion_max_delta * torch.tanh(self.gain_fusion_strength)
delta = strength * (gate_x2r - gate_r2x)
fused = (1.0 + delta) * rgb + (1.0 - delta) * x
```

Initialize `gain_fusion_strength` to zero so the pretrained output is unchanged at step zero.

- [ ] **Step 6: Extend the PEFT allowlist narrowly**

Add only `counterfactual_gain` and `gain_fusion_strength`; verify no old `moe`, `rgae`, or legacy scaling parameter is trainable in the new run.

- [ ] **Step 7: Run integration and regression tests**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS, including existing logging tests.

### Task 4: Post-Backward SCGE Training Hook

**Files:**
- Modify: `lib/train/actors/seatrack.py`
- Modify: `lib/train/trainers/ltr_trainer.py`
- Modify: `tests/test_counterfactual_integration.py`

**Interfaces:**
- `SEATrackActor.after_backward(grad_scale: float) -> tuple[Tensor | None, dict]`.
- The trainer performs task backward, constructs the router-only loss, performs a second cheap backward, then clips and steps once.

- [ ] **Step 1: Write a failing post-backward hook test**

Use a dummy network returning a known auxiliary loss after gate gradients exist. Require that actor weighting and returned `SCGE/router_loss` statistics are exact.

- [ ] **Step 2: Implement the actor hook with `TRAIN.COUNTERFACTUAL_LOSS_WEIGHT`**

Return `None, {}` when counterfactual routing is disabled or during validation.

- [ ] **Step 3: Update trainer backward ordering with AMP-safe scaling**

```python
self.scaler.scale(task_loss).backward()
router_loss, router_stats = self.actor.after_backward(self.scaler.get_scale())
if router_loss is not None:
    self.scaler.scale(router_loss).backward()
stats.update(router_stats)
```

Unscale once before clipping and optimizer step.

- [ ] **Step 4: Run all unit tests and a synthetic CUDA backward**

Expected: gate gradients, router gradients, adapter gradients, and optimizer update are finite; no second-order graph is created.

### Task 5: Exact Final-Layer Calibration

**Files:**
- Modify: `lib/models/layers/counterfactual_gain.py`
- Modify: `lib/models/seatrack/vit_ci.py`
- Modify: `lib/models/seatrack/seatrack.py`
- Modify: `lib/train/actors/seatrack.py`
- Modify: `tests/test_counterfactual_gain.py`
- Modify: `tests/test_counterfactual_integration.py`

**Interfaces:**
- Every `TRAIN.COUNTERFACTUAL_EXACT_INTERVAL` steps, the final routing sidecar returns four candidate states ordered `00,10,01,11`.
- `SEATrackActor.compute_per_sample_losses()` returns `[B]` task losses.
- Exact utility labels are `q1=L00-L10`, `q2=L00-L01`, and `q12=L10+L01-L11-L00`.

- [ ] **Step 1: Write failing exact-label algebra tests**

For losses `[L00,L10,L01,L11]=[10,7,8,4]`, require labels `[3,2,1]`.

- [ ] **Step 2: Implement per-sample GIoU, L1, and focal losses**

Use the same weights and target heatmap as the main task loss, but retain the batch dimension. Verify their mean matches the ordinary loss within numerical tolerance.

- [ ] **Step 3: Return final-layer candidates without duplicating the backbone**

Stack only the four post-layer-11 candidate states along batch, run final norm/fusion and the frozen Center Head under `torch.no_grad()`, and discard candidate outputs immediately after labels are cached.

- [ ] **Step 4: Add deterministic 10% calibration cadence**

Set `COUNTERFACTUAL_EXACT_INTERVAL=10`. Calibration is disabled during validation and inference. Record exact/Taylor sign agreement, Spearman-ready paired values, and interaction magnitude.

- [ ] **Step 5: Run exact-calibration tests**

Expected: candidate ordering and labels are exact, main prediction is unchanged by enabling diagnostics, and no candidate tensor survives into the next step.

### Task 6: Experiment Configurations and Smoke Gates

**Files:**
- Create: `experiments/seatrack/rgbt_ostrack_dual_clean.yaml`
- Create: `experiments/seatrack/rgbt_gratrack_scge_short.yaml`
- Create: `experiments/seatrack/rgbt_gratrack_scge.yaml`
- Create: `knowledge_base/GRATrack-SCGE实验记录.md`

**Interfaces:**
- Clean config: legacy disabled, counterfactual disabled, same OSTrack checkpoint/data/head.
- SCGE config: legacy disabled, counterfactual enabled at `[3,7,11]`, rank 8, router dim 32, exact interval 10.

- [ ] **Step 1: Create matched clean and SCGE YAMLs**

Keep input size, data, optimizer, epoch, and checkpoint identical. The short config uses `LasHeR_smoke`, batch 1, one epoch, and one sample.

- [ ] **Step 2: Validate configs and parameter purity**

Print trainable names and require:

```text
counterfactual_gain.*
gain_fusion_strength
```

Reject any trainable name containing `attn_moe`, `ffn_moe`, `rgae`, `r2dte_scaling`, or `dte2r_scaling`.

- [ ] **Step 3: Run CPU shape smoke and CUDA synthetic forward/backward**

Require `pred_boxes=[B,1,4]`, `score_map=[B,1,H,W]`, finite task/router losses, and peak allocation within the 32 GiB GPU.

- [ ] **Step 4: Run the one-sample real-data smoke training**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack --config rgbt_gratrack_scge_short \
  --save_dir /mnt/tipro4t/seatrack_train_runs/gratrack_scge_short_20260710 --mode single
```

Expected: one checkpoint, finite task and router losses, populated action/Taylor diagnostics, and no legacy-module keys.

- [ ] **Step 5: Record exact commands, logs, checkpoint, metrics, and failures**

Write factual evidence into `knowledge_base/GRATrack-SCGE实验记录.md`; do not label a smoke run as an accuracy result.

### Task 7: Short Diagnostic Experiment and Go/No-Go

**Files:**
- Modify: `knowledge_base/GRATrack-SCGE实验记录.md`

- [ ] **Step 1: Run matched clean and SCGE diagnostic subsets**

Use the same sample order, seed, checkpoint, and 200-500 samples. Capture loss, IoU, peak memory, samples/s, action proportions, gain distributions, and exact/Taylor pairs.

- [ ] **Step 2: Apply predeclared gates**

Proceed toward a long run only if:

```text
finite batches = 100%
legacy modules in SCGE graph = 0
exact/Taylor directional sign agreement >= 70%
all four actions are explored during training
router action max fraction <= 90%
SCGE peak memory <= 1.30 x clean
SCGE step time <= 1.30 x clean
```

- [ ] **Step 3: Decide the next experiment from evidence**

If gates pass, launch a recoverable 5-epoch pilot before any 60-epoch run. If sign agreement fails, stop and revise the Taylor target/calibration rather than tuning benchmark hyperparameters. If action collapse occurs, adjust exploration/temperature only and rerun the same diagnostic subset.

- [ ] **Step 4: Run final verification**

Run all unit tests, inspect `git diff --check`, list changed files, and verify no required process remains running before reporting results.
