# LiftTrack Rank-Collapsed BiLift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and experimentally falsify a clean LoRA-plus-BiLift replacement for SEATrack's AMG and HMoE, proceeding to long training only if accuracy and strict compute-parity gates pass.

**Architecture:** Existing K/V LoRA remains mergeable at inference. AMG and HMoE are disabled only for the opt-in LiftTrack path, and two rank-8 sequential additive coupling units are inserted after attention at blocks 5 and 9. Tests enforce exact zero-initialized equivalence, reversible coupling, parameter purity, and backward compatibility before any real training.

**Tech Stack:** Python 3.12, PyTorch 2.12, unittest, YAML, existing SEATrack/OSTrack ViT-B and Center Head infrastructure.

## Global Constraints

- Preserve all existing user changes and do not revert unrelated files.
- Work on branch `gratrack-scge-experiment`; this dirty branch is the authoritative research state.
- Do not create commits that capture pre-existing changes in shared files.
- Existing experiment YAML files must remain behavior-compatible.
- New behavior is opt-in and defaults to disabled.
- Keep standard K/V LoRA in LiftTrack; disable only AMG guidance and HMoE.
- LiftTrack must instantiate no HMoE, AMG scaling, GRA, or RGAE parameters.
- Use test-first development for every production behavior.
- Do not launch a 60-epoch run before unit, integration, real-data, efficiency, and five-epoch pilot gates pass.
- Record every command, configuration, checkpoint, metric, and failure in `knowledge_base/LiftTrack-实验记录.md`.

---

### Task 1: BiLift Core

**Files:**
- Create: `lib/models/layers/bilift.py`
- Create: `tests/test_bilift.py`

**Interfaces:**
- `LowRankCrossUpdate(dim: int, rank: int, dropout: float = 0.0)` maps `[B,N,D]` to an update with the same shape.
- `BiLift(dim: int, rank: int, dropout: float = 0.0, reverse: bool = False, diagnostics: bool = False)` returns `(rgb_out, x_out)`.
- `BiLift.inverse(rgb_out, x_out)` reconstructs the two inputs.
- `BiLift.last_stats` contains detached scalar diagnostics only when diagnostics are enabled.

- [ ] **Step 1: Write failing zero-initialization and validation tests**

```python
class BiLiftTests(unittest.TestCase):
    def test_zero_initialized_bilift_is_exact_identity(self):
        module = BiLift(dim=16, rank=4).eval()
        rgb = torch.randn(2, 5, 16)
        x = torch.randn(2, 5, 16)
        rgb_out, x_out = module(rgb, x)
        self.assertTrue(torch.equal(rgb_out, rgb))
        self.assertTrue(torch.equal(x_out, x))

    def test_invalid_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            BiLift(dim=16, rank=0)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_bilift -v`

Expected: FAIL with `ModuleNotFoundError` for `lib.models.layers.bilift`.

- [ ] **Step 3: Implement minimal low-rank updates and forward coupling**

```python
class LowRankCrossUpdate(nn.Module):
    def __init__(self, dim, rank, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.xavier_normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, source):
        return self.up(self.dropout(self.act(self.down(self.norm(source)))))
```

`BiLift.forward()` must use the exact sequential equations in the design specification and reverse their order only when `reverse=True`.

- [ ] **Step 4: Run the identity tests and verify GREEN**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_bilift -v`

Expected: PASS for identity and validation tests.

- [ ] **Step 5: Write failing inverse and gradient tests**

Randomize both `up.weight` tensors after construction, call forward followed by inverse, and require max absolute reconstruction error below `1e-5` for both orders. Backpropagate `rgb_out.square().mean() + x_out.square().mean()` and require finite gradients on both up projections.

- [ ] **Step 6: Run the new tests and verify RED**

Expected: FAIL because `inverse()` and diagnostics are absent.

- [ ] **Step 7: Implement inverse and optional detached diagnostics**

Record:

```text
BiLift/x2r_update_ratio
BiLift/r2x_update_ratio
BiLift/difference_ratio
```

Ratios use `clamp_min(1e-6)` denominators and are created under `torch.no_grad()`. Diagnostics must not retain activation tensors.

- [ ] **Step 8: Run BiLift tests**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_bilift -v`

Expected: all BiLift tests PASS for normal and reverse order.

### Task 2: Transformer Block Integration

**Files:**
- Modify: `lib/models/layers/attn_blocks.py`
- Create: `tests/test_bilift_integration.py`

**Interfaces:**
- `CEBlock_AP(..., amg_enabled=True, hmoe_enabled=True, bilift_enabled=False, bilift_rank=8, bilift_reverse=False, bilift_dropout=0.0, bilift_diagnostics=False)`.
- `CEBlock_AP.bilift` exists only when enabled at that block.
- `CEBlock_AP.bilift_stats` is empty unless diagnostics are enabled.

- [ ] **Step 1: Write failing clean-construction tests**

Create a `CEBlock_AP` with `dim=32`, `num_heads=4`, `lora_layers=[1]`, `moe_layers=[1]`, `layer=1`, AMG/HMoE disabled, and BiLift enabled. Assert:

```python
self.assertFalse(hasattr(block, "attn_moe"))
self.assertFalse(hasattr(block, "ffn_moe"))
self.assertFalse(hasattr(block, "r2dte_scaling"))
self.assertFalse(hasattr(block, "dte2r_scaling"))
self.assertIsNotNone(block.bilift)
self.assertEqual(block.attn.qkv.r, 4)
```

Pass `amglora_rank=4` and confirm the constructor honors it instead of hard-coding rank 8.

- [ ] **Step 2: Run the integration test and verify RED**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_bilift_integration -v`

Expected: FAIL on unknown constructor fields or hard-coded rank.

- [ ] **Step 3: Separate LoRA construction from AMG behavior**

Use the configured `amglora_rank` in `Attention`. Instantiate AMG scalars only when `amg_enabled` and the layer is in `lora_layers`. When LoRA is enabled but AMG is disabled, calculate normal self-attention from the LoRA-adjusted QKV without guidance.

- [ ] **Step 4: Make HMoE construction conditional**

Instantiate `attn_moe` and `ffn_moe` only when `hmoe_enabled` and the layer is selected. Use the configured `hmoe_rank` instead of literal 4.

- [ ] **Step 5: Insert BiLift after attention residuals**

After:

```python
x[0] = x[0] + self.drop_path(xrgb_attn)
x[1] = x[1] + self.drop_path(xdte_attn)
```

apply `x[0], x[1] = self.bilift(x[0], x[1])` before legacy HMoE and candidate elimination. Copy scalar diagnostics into `self.bilift_stats`.

- [ ] **Step 6: Test output identity and compatibility**

Construct matched blocks with and without zero-initialized BiLift, copy shared state, run evaluation with identical inputs and indices, and require exact output equality. Separately construct the default legacy block and require AMG scalars and both HMoE modules to remain present.

- [ ] **Step 7: Run block integration and existing tests**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_bilift_integration tests.test_bilift -v`

Expected: PASS.

### Task 3: Model, Configuration, and PEFT Integration

**Files:**
- Modify: `lib/config/seatrack/config.py`
- Modify: `lib/models/seatrack/vit_ci.py`
- Modify: `lib/models/seatrack/seatrack.py`
- Modify: `lib/train/base_functions.py`
- Modify: `tests/test_bilift_integration.py`

**Interfaces:**
- `MODEL.AMG_ENABLED` and `MODEL.HMOE_ENABLED` default to true.
- `MODEL.BILIFT.{ENABLED,LAYERS,RANK,DROPOUT,DIAGNOSTICS}` follows the design contract.
- `VisionTransformerCE` alternates BiLift order by position in `BILIFT.LAYERS`.
- The PEFT allowlist includes `bilift` and no broad new substring.

- [ ] **Step 1: Write failing config and invalid-combination tests**

Assert defaults preserve legacy behavior. Assert model construction raises `ValueError` when BiLift is enabled together with AMG, HMoE, GRA behavior, or GRA diagnostics.

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because the new config fields and model arguments do not exist.

- [ ] **Step 3: Add config defaults and builder validation**

Add exact defaults from the design specification. In `build_seatrack`, reject incompatible configurations before constructing the backbone.

- [ ] **Step 4: Propagate configuration through ViT construction**

For block index `i`, enable BiLift only if `i in bilift_layers`. Set `bilift_reverse` from the selected-layer position:

```python
bilift_reverse = bilift_layers.index(i) % 2 == 1
```

Keep LoRA layers `[1,3,5,7,9,11]`; pass an empty MoE layer list when HMoE is disabled.

- [ ] **Step 5: Aggregate BiLift diagnostics**

Collect detached scalar values from each enabled block into `aux_dict['bilift_stats']`. Update the actor status using the existing generic statistics path without retaining full tensors.

- [ ] **Step 6: Narrowly extend PEFT selection**

Add only `bilift` to the existing trainable-name allowlist. A test must verify that a LiftTrack model has no trainable parameter containing `moe`, `rgae`, `r2dte_scaling`, or `dte2r_scaling`.

- [ ] **Step 7: Verify expected parameter count**

For ViT-B rank 8 and layers `[5,9]`, require BiLift parameters equal 49,152 and total LoRA-plus-BiLift trainable parameters equal 196,608 before any optional head parameter.

- [ ] **Step 8: Run all unit tests**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

### Task 4: Experiment Configurations and Evidence Record

**Files:**
- Create: `experiments/seatrack/rgbt_lora_only.yaml`
- Create: `experiments/seatrack/rgbt_seatrack_pilot.yaml`
- Create: `experiments/seatrack/rgbt_lora_only_pilot.yaml`
- Create: `experiments/seatrack/rgbt_lifttrack_short.yaml`
- Create: `experiments/seatrack/rgbt_lifttrack_pilot.yaml`
- Create: `experiments/seatrack/rgbt_lifttrack.yaml`
- Create: `knowledge_base/LiftTrack-实验记录.md`

**Interfaces:**
- SEATrack, LoRA-only, and LiftTrack pilot configurations use identical data, optimizer, initialization, input size, validation cadence, and save cadence.
- Short config uses one LasHeR smoke sample, batch 1, no validation, and one epoch.
- Pilot uses 5 epochs, 60,000 training and validation samples per epoch, validation every epoch, and saves every epoch.
- Full config uses the paper's 60-epoch RGB-T protocol and saves every 5 epochs.

- [ ] **Step 1: Create matched YAML files**

LoRA-only sets AMG/HMoE/BiLift false. LiftTrack sets AMG/HMoE false and BiLift true at `[5,9]`, rank 8. Full SEATrack keeps AMG/HMoE enabled and BiLift disabled. All pilot configs use five epochs, validation and checkpointing every epoch, and all GRA options disabled.

- [ ] **Step 2: Add configuration audit tests**

Load all six YAMLs into a deep-copied default config and assert the exact model flags, dataset, batch size, epochs, validation cadence, and save cadence.

- [ ] **Step 3: Initialize the evidence record**

Record the frozen hypothesis, analytical budget, predeclared gates, baseline test result, checkpoint paths, hardware, branch, and current dirty-worktree constraint. Do not record smoke metrics as accuracy results.

- [ ] **Step 4: Run config tests and YAML parsing**

Run: `/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_bilift_integration -v`

Expected: PASS with no unknown config keys.

### Task 5: Synthetic and Real-Data Smoke Gates

**Files:**
- Create: `tools/profile_lifttrack.py`
- Modify: `knowledge_base/LiftTrack-实验记录.md`

**Interfaces:**
- Profiler accepts `--config`, `--checkpoint`, `--warmup`, `--iterations`, and `--output`.
- Output JSON includes parameter counts, HMoE module count, BiLift module count, finite-output status, mean/median/P90 latency, FPS, peak allocated/reserved bytes, and protocol metadata.

- [ ] **Step 1: Write failing profiler helper tests**

Test percentile calculation, module counting, and JSON schema using small real PyTorch modules. Keep CUDA measurement behind a runtime availability check.

- [ ] **Step 2: Implement profiler helpers and verify tests**

Run the helper tests until PASS before adding the CLI path.

- [ ] **Step 3: Run CPU and CUDA synthetic forward/backward**

Require finite `[B,1,4]` boxes, `[B,1,16,16]` score maps, finite task loss, finite LoRA/BiLift gradients, HMoE count zero, and peak allocation below 32 GiB.

- [ ] **Step 4: Run one-sample real-data smoke training**

```bash
CUDA_VISIBLE_DEVICES=0 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack \
  --config rgbt_lifttrack_short \
  --save_dir /mnt/tipro4t/seatrack_train_runs/lifttrack_short_20260711 \
  --mode single
```

Expected: one checkpoint, finite losses, finite BiLift diagnostics, and no legacy module keys.

- [ ] **Step 5: Run paired inference profiling**

Use the same real LasHeR template/search pair, load order `SEATrack, LiftTrack, LiftTrack, SEATrack`, 20 warmups per load, and 200 total measured iterations per variant. Use CUDA events with synchronization.

- [ ] **Step 6: Apply strict efficiency gates**

Reject the candidate if mean latency exceeds `1.00x`, P90 exceeds `1.02x`, peak allocated memory exceeds `1.00x`, or analytical MACs exceed `1.00x` reproduced SEATrack.

- [ ] **Step 7: Record exact evidence**

Append commands, JSON paths, logs, checkpoint, finite checks, and gate decisions to the evidence record.

### Task 6: Five-Epoch Matched Pilot

**Files:**
- Modify: `knowledge_base/LiftTrack-实验记录.md`

**Interfaces:**
- Pilot run IDs include method, seed, and timestamp.
- Compared methods are full SEATrack, LoRA-only, and LiftTrack.
- Seed 0 uses identical sample counts and optimizer protocol; additional seeds are run only after seed-0 engineering validity.

- [ ] **Step 1: Launch recoverable seed-0 pilots sequentially**

Run one method at a time on the same GPU. Save every epoch and keep complete model/config/data/system/train logs.

- [ ] **Step 2: Monitor every epoch**

Record train loss, validation loss, IoU, FPS, step time, peak memory, BiLift update ratios, and exceptions. Stop immediately on NaN, invalid boxes, OOM, or missing gradients.

- [ ] **Step 3: Compare predeclared checkpoints**

At epochs 1 through 5, calculate LiftTrack minus SEATrack and LiftTrack minus LoRA-only validation IoU without selecting a favorable epoch after seeing results.

- [ ] **Step 4: Apply pilot go/no-go**

Proceed only if epoch-5 LiftTrack exceeds LoRA-only and no LiftTrack checkpoint is more than 0.002 below matched SEATrack. Otherwise stop and test one predeclared change: rank 16 first, then layers `[7,11]`; do not tune both simultaneously.

- [ ] **Step 5: Repeat the winning pilot for seeds 1 and 2**

Require the mean result to satisfy the same gate and report standard deviation. Do not launch full training from a single favorable seed.

### Task 7: Full RGB-T Experiment and Benchmark

**Files:**
- Modify: `knowledge_base/LiftTrack-实验记录.md`

- [ ] **Step 1: Launch a recoverable 60-epoch LiftTrack run**

Use the approved rank/layers without further benchmark-driven tuning. Save every 5 epochs and maintain a run manifest.

- [ ] **Step 2: Select checkpoint only from validation**

Select the best checkpoint using predeclared validation IoU. Record the last checkpoint separately. Do not evaluate multiple epochs on LasHeR test to choose the best.

- [ ] **Step 3: Evaluate LasHeR and RGBT234 once per seed**

Require complete sequence sets and zero missing outputs. Report PR, NPR, SR, MPR, MSR, paired sequence bootstrap confidence intervals, and full runtime protocol.

- [ ] **Step 4: Run primary ablations**

Compare LoRA-only, reproduced SEATrack, parameter-matched parallel adapters, and BiLift. Rank and layer ablations use the same training budget and selection protocol.

- [ ] **Step 5: Decide cross-modal expansion**

Expand to DepthTrack/VOT-RGBD2022 and VisEvent only if LiftTrack is no slower and improves or statistically matches SEATrack on both RGB-T benchmarks.

### Task 8: Completion Audit

**Files:**
- Modify: `knowledge_base/LiftTrack-实验记录.md`

- [ ] **Step 1: Run all tests and static checks**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

- [ ] **Step 2: Audit every claim against an artifact**

Map method correctness to tests, efficiency to paired profiler JSON, accuracy to manifests and benchmark outputs, and robustness to controlled-degradation results. Mark unsupported claims as pending rather than inferred.

- [ ] **Step 3: Verify runtime state**

Confirm no failed or orphaned training/evaluation process remains and all intended checkpoints are readable.

- [ ] **Step 4: Complete only on effective evidence**

The research goal is complete only when a method is simpler than SEATrack, passes strict compute parity, and has verified effective benchmark results. A working implementation or smoke run alone is not completion.
