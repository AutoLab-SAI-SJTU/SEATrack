# RGB-X Target-Spectral Workstream A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and experimentally falsify a strictly causal, bounded target-spectral memory that observes the existing SEATrack HMoE pre-router features and changes only search-token routing logits, while all tracker parameters remain frozen through the Stage 0 (S0) decision.

**Architecture:** An initialization-only legacy-routing calibration forward builds an immutable target anchor. Each real frame receives one cloned pre-frame `MemorySnapshot`; blocks 5 and 9 use that same snapshot for attention and FFN HMoE search routing, while template routing and expert inputs remain legacy. Detached observations are converted to target/private/dynamic/background uncentered-second-moment factors only after the prediction is committed, and their prepared write becomes visible on the next frame. A separate chronological, prediction-centred rollout fits one global four-coefficient vector offline; it causally replays an unlabeled same-sequence prefix before the fixed five-frame optimization clip. Every manifested attempt is processed independent of current family activity, and a frozen real-data feasibility audit must show all key/family routes plus full-rank, nondegenerate loss sensitivity on the three-dimensional softmax-coefficient tangent. S0 replays one schedule recorded from the routing-disabled frozen baseline across all matched controls.

**Tech Stack:** Python 3.12, PyTorch 2.12, NumPy/SciPy, unittest, YAML/JSONL, existing SEATrack ViT-B/HMoE, Center Head, LasHeR and DepthTrack loaders.

## Global Constraints

- The base design is `docs/superpowers/specs/2026-07-13-rgbx-target-spectral-continual-moe-lora-design.md`, SHA-256 `36eaf659a0b6550aefee1db9548a67caca875b6dd839857baacde099ee9049de`. Decisions 1–6 below are user-approved; proposed Decision 7 becomes part of the normative basis only after explicit user ratification. Any further divergence requires amendment and review before code changes.
- Work only on branch `gratrack-scge-experiment`. Preserve unrelated user changes and stage only files named by the current task.
- This plan covers Workstream A only: A0 causal contract, observability, bounded spectral state, frozen forward routing, ordered rollout, coefficient fitting, M0/M1, and S0.
- Do not add a test-time optimizer, reliability-network training, persistent replay buffer/crop cache, EMA teacher, rollback, Stage R, Stage E, expert updates, qkv LoRA updates, sparse expert execution, or cross-sequence global memory. Decision 7's offline, image-only, same-sequence prefix pass is chronological coefficient-fit input, not test-time replay or retained replay state.
- The admissible Workstream A claim is only **sequence-local causal spectral-memory routing with frozen tracker parameters**. Do not call it continual parameter learning, a general safety guarantee, or proof that target-spectral geometry is uniquely beneficial; passing controls supports specificity relative only to those registered controls, never uniqueness.
- All SEATrack parameters, including `linear1`, `gate_thi`, temperatures, expert LoRA, backbone, and head, remain `requires_grad=False` and in evaluation mode during coefficient fitting and S0. The only offline fitted tensor is one external `u.shape == (4,)`; freeze it before gate confirmation.
- Existing HMoE remains enabled at legacy layers `[1,3,5,7,9,11]`. Spectral observation/routing is opt-in only at blocks `[5,9]`, for attention and FFN HMoE.
- Core routing is search-only. Template HMoE routing stays legacy; template features may be observed only during the initialization anchor forward. Template-routing is a later ablation only after S0 passes.
- The initialization calibration forward uses the first image, its legal initialization box, the normal template crop, and a search-size crop centred on that box. It disables active spectral routing, discards the prediction, and writes only immutable anchor factors.
- Every real frame uses one immutable pre-frame snapshot across all blocks, modalities, and attention/FFN calls. Observer writes are staged, detached, and committed only after the frame prediction and state are committed.
- Spectral transforms affect HMoE logits only. Dispatch-weighted expert inputs must continue to use the raw legacy pre-router tensor.
- The disabled/no-context, empty-state, and zero-strength paths must take an early bypass and be bitwise output-identical to legacy evaluation output. The zero-strength bypass must not execute observer RMS normalization, spectral application, or residual-logit arithmetic.
- When `TARGET_SPECTRAL.ENABLED=False`, do not construct a controller, create an x0 crop, run an initialization forward, or emit diagnostics; initialization and each track call retain the exact legacy forward count and operation order.
- Observer-only anchor capture is an explicit mode distinct from zero-strength routing: it may stage detached features but must still leave outputs bitwise identical.
- Use “uncentered second moment,” not covariance, in Workstream A code and reports. Call `Pi` a shrinkage-weighted spectral operator throughout.
- Candidate elimination is out of scope. Every Workstream A config must set `CE_KEEP_RATIO: [1.0, 1.0, 1.0]`; fail construction if any spectral-enabled keep ratio is below 1.
- GRA, BiLift, ProbAlign, LiftTrack, and target-spectral behavior are mutually exclusive in the first pilot. Treat GRA or BiLift as active when either its `ENABLED` or `DIAGNOSTICS` flag is true, matching the existing construction paths; fail construction instead of composing them.
- Post-initialization `track()` input must contain no GT box, visibility, valid flag, attribute, corruption identity, corruption severity, corruption boundary, or future-frame field. The official VOT restart box is a new fully reset initialization episode, not a tracking-frame exception.
- S0 schedule generation runs once per base/dataset/condition from routing-disabled, frozen legacy outputs with a calibration-frozen GT-free confidence rule. It must not be generated by an adaptive confidence-memory row. The sealed record contains the common `(scheduled_admit, q_memory, asymmetry, paired_valid)` tuple, not only an admit bit.
- S0 model rows replay identical admitted frame indices. Oracle schedules are excluded.
- All controls match retained rank/state bytes, four-coefficient capacity, operator cap, residual-logit budgets, layers/modules, schedule exposure, and calibration exposure. Inactive branches use reported inert padding.
- Active full/control rows use fixed routing `strength=1.0`; only `zero_strength_instrumented` uses `0.0`. Strength has no calibration search domain.
- Mandatory diagnostics include identity/private/dynamic/background-only, cumulative, and raw leave-one-out rows. Binding attribution additionally requires all four strength-matched leave-one-out rows, `random_orthogonal`, `pooled_same`, `target_balanced_identity`, RGB-X pair shuffle, temporal-order shuffle, and target/background-mask shuffle.
- Active routing has no bitwise no-loss guarantee. Operational protection is per benchmark and comparator: simultaneous `LCB(Delta J_b) > -0.3` percentage points for both LasHeR and DepthTrack against `confidence_only_scalar_history` and `routing_disabled_legacy`.
- Gate-confirmation data may accept or reject a frozen registry but may not retune it. Do not run gate confirmation until manifests, thresholds, rank, coefficients, checkpoint hashes, benchmark-evaluator provenance, recovery estimand, and registry hash are committed.
- Use `.venv/bin/python`, never system `python`, for tests and scripts. The repository uses `unittest`, not pytest.
- Every new directly executed `tools/*.py` bootstraps the repository root before importing `lib`. Every new directly executed `tracking/*.py` imports `tracking/_init_paths.py` before any `lib` import. Tests invoke every new tool/tracking CLI with `--help` from both the repository root and `/tmp` by absolute script path.
- Apply TDD for every production behavior: write a focused failing test, run it and confirm the intended failure, implement the minimum behavior, rerun focused tests, then run affected regression tests.
- Commit after each completed task using the exact task-local commit command. Do not push until explicitly requested.

---

## Approved Decisions and Proposed Feasibility Amendment

Decisions 1–6 record the three-expert review accepted by the user on 2026-07-13 and supersede only the corresponding Workstream A details in the base design:

1. Core Stage 0 routing is search-only at blocks 5 and 9, attention and FFN. Template HMoE stays legacy and contributes only initialization anchor observations.
2. The common diagnostic schedule is generated from routing-disabled frozen legacy predictions, then sealed and replayed. It is not generated by the confidence-memory comparator.
3. The recovery co-primary is burst-aligned time-to-stable-success on one registered X-blackout burst per sequence, with a treatment-independent risk set and event time at the fifth qualifying frame.
4. One global four-scalar `u` is jointly fitted across all six frozen base checkpoints (three base seeds by two modalities) and then shared unchanged by every seed.
5. Clean noninferiority is required against both the confidence-only comparator and matched routing-disabled legacy.
6. Modality evidence comes from detached RGB/X template-to-search attention distributions with global indices; no RGB-only or X-only box/head prediction is introduced.
Expert-proposed Decision 7 requires explicit user ratification before implementation because it resolves a contradiction in the base design's exact five-frame sampler:

7. The fixed optimization clip remains `[z0,x_(t-2),x_(t-1),x_t,x_(t+1)]`, but after every clip reset the fitter first replays the unlabeled same-sequence prefix `x_1...x_(t-3)` in strict causal order. A deterministic manifest fixes every attempted clip without inspecting labels, family activity, confidence, loss, or coefficient values. Every attempt is processed and recorded; only an `x_(t+1)` validity label read after the outer crop/forward may suppress its supervised loss, and no attempt is replaced or reweighted. Before fitting and again at the selected checkpoint, a frozen real-data feasibility audit must show in every one of the six base/modality strata registered coverage for every adaptive `(block,site,family)` route and a full-rank, nondegenerate signed outer-loss gradient design on the three-dimensional tangent of the fixed-budget four-alpha simplex; per-leaf alpha sensitivity remains diagnostic. Optimizer checkpoints count successful six-stratum steps separately from attempted supersteps and fitting fails at the registered attempt ceiling. This correction is required because three unit-mass commits cannot reach the registered effective-mass/rank activation gates, while conditioning individual losses on all families being active would create activity-dependent selection bias.

This plan is conditionally reviewed until Decision 7 is ratified. Before implementation code starts, Task 0 verifies that ratification, then writes the exact seven-decision addendum to a separately hashed file. The calibration registry references both the base-design path and addendum path; the freeze tool computes their hashes.

## Workstream A Claim Boundary

Workstream A tests only whether bounded, sequence-local, strictly causal uncentered-second-moment state can improve search-token routing in the existing dense SEATrack HMoE while every tracker and LoRA parameter remains frozen under standard sequence-reset OPE. Before S0 passes, this is a falsification study and no performance contribution is established.

If and only if the effect, clean noninferiority, geometry-specificity, strength-matched LOO, and all three semantic-shuffle gates pass, the supported claim is limited to: under a calibration-frozen GT-free admission schedule, bounded target-specific RGB-X common/private/dynamic/background spectral operators causally modulate the existing dense HMoE search routing and improve the preregistered Stage-0 endpoint relative to the registered scalar-confidence history, random-basis, pooled-history, and identity-only geometry controls.

Workstream A does not establish continual parameter learning, online router/LoRA adaptation, cross-sequence retention, sparse execution, expert skipping, a mathematical safety guarantee, or guaranteed performance preservation. Every effect, noninferiority, and attribution interval is conditional on the frozen shared `u`, checkpoint set `{0,1,2}`, registered control seeds, registered datasets/splits, selected corruption, and frozen hardware where relevant. A passing effect supports only the selected registered-corruption co-primary, not general clean OPE, arbitrary retraining seeds, future sequences, or other deployments. Any claim about safely updating router, expert, or LoRA parameters requires a separately reviewed Workstream B on unopened confirmation data.

## File and Data-Flow Map

Existing files to modify:

- `.gitignore`
- `lib/config/seatrack/config.py`
- `lib/models/layers/attn.py`
- `lib/models/layers/attn_blocks.py`
- `lib/models/seatrack/vit_ci.py`
- `lib/models/seatrack/seatrack.py`
- `lib/train/trainers/base_trainer.py`
- `lib/test/evaluation/tracker.py`
- `lib/test/vot/seatrack_class.py`
- `lib/test/tracker/basetracker.py`
- `lib/test/tracker/seatrack.py`
- `lib/test/tracker/ostrack.py`
- `lib/train/base_functions.py`
- `lib/train/data/__init__.py`
- `lib/train/dataset/lasher.py`
- `lib/train/dataset/depthtrack.py`

New target-spectral model package:

- `lib/models/target_spectral/__init__.py`
- `lib/models/target_spectral/types.py`
- `lib/models/target_spectral/memory.py`
- `lib/models/target_spectral/observation.py`
- `lib/models/target_spectral/routing.py`
- `lib/models/target_spectral/stage0.py`
- `lib/models/seatrack/checkpoint.py`

New rollout, fit, and evaluation modules:

- `lib/train/data/rollout_sampler.py`
- `lib/train/data/rollout_processing.py`
- `lib/train/spectral/__init__.py`
- `lib/train/spectral/coefficient_fit.py`
- `lib/test/evaluation/causal.py`
- `lib/test/evaluation/spectral_s0.py`
- `lib/test/evaluation/spectral_corruption.py`
- `lib/test/evaluation/benchmark_metrics.py`
- `lib/test/evaluation/spectral_statistics.py`
- `tracking/fit_spectral_coefficients.py`
- `tracking/record_spectral_schedule.py`
- `tracking/run_spectral_s0.py`
- `tracking/analyze_spectral_s0.py`
- `tracking/evaluate_benchmark_metrics.py`
- `tools/freeze_target_spectral_splits.py`
- `tools/index_target_spectral_checkpoints.py`
- `tools/freeze_spectral_s0_registry.py`
- `tools/profile_spectral_s0.py`
- `tools/validate_benchmark_evaluators.py`

New configuration and evidence files:

- `experiments/seatrack/rgbt_spectral_s0.yaml`
- `experiments/seatrack/rgbd_spectral_s0.yaml`
- `experiments/seatrack/rgbt_spectral_s0_short.yaml`
- `experiments/seatrack/rgbt_spectral_base.yaml`
- `experiments/seatrack/rgbd_spectral_base.yaml`
- `experiments/seatrack/registries/spectral_s0_v1.calibration.yaml`
- generated and committed `experiments/seatrack/registries/spectral_s0_v1.frozen.yaml`
- generated and committed `experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json`
- runtime `output/spectral_s0/base/base_checkpoints.json`, with its exact committed provenance copy at `knowledge_base/Target-Spectral-S0-base-checkpoints.json`
- six fit/calibration/gate-confirmation manifests under `lib/train/data_specs/`
- `knowledge_base/Target-Spectral-S0-实验记录.md`
- `knowledge_base/Target-Spectral-S0-gate.json`
- `docs/superpowers/specs/2026-07-13-target-spectral-workstream-a-addendum.md`

New tests:

- `tests/test_spectral_causality.py`
- `tests/test_spectral_memory.py`
- `tests/test_spectral_observation.py`
- `tests/test_spectral_routing.py`
- `tests/test_spectral_integration.py`
- `tests/test_spectral_stage0_tracker.py`
- `tests/test_spectral_rollout.py`
- `tests/test_spectral_coefficients.py`
- `tests/test_spectral_config_registry.py`
- `tests/test_spectral_s0_evaluation.py`
- `tests/test_profile_spectral_s0.py`

Runtime flow:

~~~text
begin_episode(reset_global=True)
  -> initialization crops z0 and x0 around the legal init box
  -> observer-only legacy forward
  -> immutable C_init factors

before frame t
  -> clone MemorySnapshot(version=t-1)
  -> build previous-box search prior
  -> one FrameRouteContext shared by blocks 5/9, attn/FFN

forward frame t
  -> raw_h = linear1(norm(x))
  -> observe detached raw_h when enabled
  -> alter search logits only from snapshot t-1
  -> Dispatch/Combine from bounded logits
  -> expert inputs from raw_h

commit prediction t
  -> derive GT-free target/background weights and confidence
  -> prepare detached factor write
  -> apply recorded admission bit
  -> atomically commit memory version t
~~~

### Task 0: Ratify and Freeze the Workstream A Addendum

**Files:**

- Create: `docs/superpowers/specs/2026-07-13-target-spectral-workstream-a-addendum.md`

- [ ] **Step 1a: Confirm the Decision 7 ratification gate**

Confirm that the immediately preceding user decision explicitly approves Decision 7 with the prefix-replay wording and that its ratification date is 2026-07-13.

Expected: explicit approval is present; otherwise STOP before any repository change.

- [ ] **Step 1b: Create the frozen addendum with `apply_patch`**

Create the file with this exact content; if approval occurs on another date, stop and amend this plan before writing it:

~~~markdown
# RGB-X Target-Spectral Workstream A Addendum

- Status: approved
- Base design: `docs/superpowers/specs/2026-07-13-rgbx-target-spectral-continual-moe-lora-design.md`
- Base design SHA-256: `36eaf659a0b6550aefee1db9548a67caca875b6dd839857baacde099ee9049de`
- Decisions 1–6 approval date: 2026-07-13
- Decision 7 ratification date: 2026-07-13

1. Core Stage 0 routing is search-only at blocks 5 and 9, attention and FFN. Template HMoE stays legacy and contributes only initialization anchor observations.
2. The common diagnostic schedule is generated from routing-disabled frozen legacy predictions, then sealed and replayed. It is not generated by the confidence-memory comparator.
3. The recovery co-primary is burst-aligned time-to-stable-success on one registered X-blackout burst per sequence, with a treatment-independent risk set and event time at the fifth qualifying frame.
4. One global four-scalar `u` is jointly fitted across all six frozen base checkpoints (three base seeds by two modalities) and then shared unchanged by every seed.
5. Clean noninferiority is required against both the confidence-only comparator and matched routing-disabled legacy.
6. Modality evidence comes from detached RGB/X template-to-search attention distributions with global indices; no RGB-only or X-only box/head prediction is introduced.
7. The fixed optimization clip remains `[z0,x_(t-2),x_(t-1),x_t,x_(t+1)]`, but after every clip reset the fitter first replays the unlabeled same-sequence prefix `x_1...x_(t-3)` in strict causal order. A deterministic manifest fixes every attempted clip without inspecting labels, family activity, confidence, loss, or coefficient values. Every attempt is processed and recorded; only an `x_(t+1)` validity label read after the outer crop/forward may suppress its supervised loss, and no attempt is replaced or reweighted. Before fitting and again at the selected checkpoint, a frozen real-data feasibility audit must show in every one of the six base/modality strata registered coverage for every adaptive `(block,site,family)` route and a full-rank, nondegenerate signed outer-loss gradient design on the three-dimensional tangent of the fixed-budget four-alpha simplex; per-leaf alpha sensitivity remains diagnostic. Optimizer checkpoints count successful six-stratum steps separately from attempted supersteps and fitting fails at the registered attempt ceiling. This correction is required because three unit-mass commits cannot reach the registered effective-mass/rank activation gates, while conditioning individual losses on all families being active would create activity-dependent selection bias.

All other requirements of the base design remain in force.
~~~

Expected: exactly one new Markdown file; no source, config, or test file changes.

- [ ] **Step 1c: Check the file body before hashing**

~~~bash
test -f docs/superpowers/specs/2026-07-13-target-spectral-workstream-a-addendum.md
git diff --check -- docs/superpowers/specs/2026-07-13-target-spectral-workstream-a-addendum.md
git status --short -- docs/superpowers/specs/2026-07-13-target-spectral-workstream-a-addendum.md
~~~

Expected: `test` and `git diff --check` exit zero; status contains exactly one new addendum file.

- [ ] **Step 2: Verify the addendum is exact and hashable**

~~~bash
sha256sum docs/superpowers/specs/2026-07-13-target-spectral-workstream-a-addendum.md
rg -n "search-only|routing-disabled frozen legacy|fifth qualifying|one global|both.*legacy|attention distributions|same-sequence prefix" \
  docs/superpowers/specs/2026-07-13-target-spectral-workstream-a-addendum.md
~~~

Expected: one SHA-256 line and seven matching decision lines.

- [ ] **Step 3: Commit the addendum before code**

~~~bash
git add docs/superpowers/specs/2026-07-13-target-spectral-workstream-a-addendum.md
git commit -m "docs: freeze target spectral workstream a addendum"
~~~

### Task 1: A0 Causal Evaluator and Tracker Contract

**Files:**

- Create: `lib/test/evaluation/causal.py`
- Modify: `lib/test/evaluation/tracker.py:63-156`
- Modify: `lib/test/tracker/basetracker.py:20-26`
- Modify: `lib/test/tracker/seatrack.py:46-81`
- Modify: `lib/test/tracker/ostrack.py:74-113`
- Modify: `lib/test/vot/seatrack_class.py:19-47`
- Create: `tests/test_spectral_causality.py`

**Interfaces:**

- `CausalFrameRecord(frame_index, previous_output)` is frozen and exposes only prediction history.
- `sanitize_previous_output()` retains `target_bbox`, `best_score`, `all_boxes`, and `all_scores`.
- `assert_causal_tracker_info()` accepts only root keys `frame_index/previous_output` and only the four prediction-history keys; unknown keys fail closed.
- `BaseTracker.begin_episode(reset_global: bool = True)` is the episode lifecycle hook.
- `BaseTracker.track(image, info=None)` is the canonical signature.
- `Tracker.create_tracker(params, mode=None)` fixes the current missing-argument failure.

- [ ] **Step 1: Write failing sanitizer, sentinel, binding, and reset tests**

~~~python
import unittest

from lib.test.evaluation.causal import (
    CausalFrameRecord,
    assert_causal_tracker_info,
)

class RecordingTracker:
    def __init__(self):
        self.begin_episode_calls = []
        self.initialize_infos = []
        self.track_infos = []

    def begin_episode(self, reset_global=True):
        self.begin_episode_calls.append(bool(reset_global))

    def initialize(self, image, info):
        self.initialize_infos.append(dict(info))
        return None

    def track(self, image, *, info):
        assert_causal_tracker_info(info)
        self.track_infos.append(info)
        return {"target_bbox": [1.0, 2.0, 3.0, 4.0], "best_score": 0.5}

def _run_one_causal_frame(tracker):
    image = object()
    tracker.begin_episode(reset_global=True)
    tracker.initialize(image, {"init_bbox": [1.0, 2.0, 3.0, 4.0]})
    safe_info = CausalFrameRecord.from_evaluator(
        frame_index=1,
        previous_output={
            "target_bbox": [1.0, 2.0, 3.0, 4.0],
            "best_score": 0.5,
            "gt_bbox": [9.0, 9.0, 9.0, 9.0],
            "target_spectral": {"must_not_recur": True},
        },
    ).as_tracker_info()
    return tracker.track(image, info=safe_info)

def run_one_synthetic_sequence(tracker):
    return _run_one_causal_frame(tracker)

def run_one_video_frame(tracker):
    return _run_one_causal_frame(tracker)

def run_one_vot_frame(tracker):
    return _run_one_causal_frame(tracker)

class CausalContractTests(unittest.TestCase):
    def test_sanitizer_exposes_only_frame_and_prediction(self):
        record = CausalFrameRecord.from_evaluator(
            frame_index=7,
            previous_output={"target_bbox": [1, 2, 3, 4], "best_score": 0.8, "gt_bbox": [9, 9, 9, 9]},
        )
        self.assertEqual(set(record.as_tracker_info()), {"frame_index", "previous_output"})
        self.assertNotIn("gt_bbox", record.as_tracker_info()["previous_output"])

    def test_unknown_keys_fail_closed_at_both_schema_levels(self):
        with self.assertRaisesRegex(AssertionError, "unexpected tracker-info key"):
            assert_causal_tracker_info({"lookahead_hint": 9})
        with self.assertRaisesRegex(AssertionError, "unexpected prediction key"):
            assert_causal_tracker_info({
                "frame_index": 2,
                "previous_output": {"corruption_severity": 1.0},
            })

    def test_evaluator_calls_track_with_keyword_info_and_no_gt(self):
        tracker = RecordingTracker()
        run_one_synthetic_sequence(tracker)
        self.assertEqual(tracker.begin_episode_calls, [True])
        self.assertEqual(set(tracker.track_infos[0]), {"frame_index", "previous_output"})

    def test_direct_entry_points_construct_causal_records(self):
        for entry_point in (run_one_video_frame, run_one_vot_frame):
            tracker = RecordingTracker()
            entry_point(tracker)
            assert_causal_tracker_info(tracker.track_infos[0])
~~~

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_causality -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'lib.test.evaluation.causal'`.

- [ ] **Step 3: Implement the immutable whitelist**

~~~python
PREDICTION_KEYS = frozenset({"target_bbox", "best_score", "all_boxes", "all_scores"})
ROOT_KEYS = frozenset({"frame_index", "previous_output"})

def _clone_prediction_value(key, value):
    if key == "best_score":
        assert isinstance(value, (int, float)), "best_score must be numeric"
        return float(value)
    if isinstance(value, torch.Tensor):
        assert value.dtype.is_floating_point, f"{key} tensor must be floating point"
        return value.detach().cpu().clone()
    assert isinstance(value, (list, tuple)), f"{key} must be a numeric sequence"
    assert all(isinstance(item, (int, float)) for item in value), (
        f"{key} must contain only numbers"
    )
    return tuple(float(item) for item in value)

def sanitize_previous_output(previous_output):
    assert isinstance(previous_output, Mapping), "previous_output must be a mapping"
    return {
        key: _clone_prediction_value(key, value)
        for key, value in previous_output.items()
        if key in PREDICTION_KEYS
    }

def assert_causal_tracker_info(info):
    unexpected_root = set(info) - ROOT_KEYS
    assert not unexpected_root, f"unexpected tracker-info key: {sorted(unexpected_root)}"
    assert type(info.get("frame_index")) is int, "frame_index must be an int"
    assert info["frame_index"] >= 1, "tracking frame_index must be positive"
    previous = info.get("previous_output", {})
    assert isinstance(previous, Mapping), "previous_output must be a mapping"
    unexpected_prediction = set(previous) - PREDICTION_KEYS
    assert not unexpected_prediction, (
        f"unexpected prediction key: {sorted(unexpected_prediction)}"
    )
    for key, value in previous.items():
        _clone_prediction_value(key, value)

@dataclass(frozen=True)
class CausalFrameRecord:
    frame_index: int
    previous_output: Mapping[str, object]

    @classmethod
    def from_evaluator(cls, frame_index, previous_output):
        safe = sanitize_previous_output(previous_output)
        return cls(int(frame_index), MappingProxyType(safe))

    def as_tracker_info(self):
        return {
            "frame_index": self.frame_index,
            "previous_output": dict(self.previous_output),
        }
~~~

- [ ] **Step 4: Fix evaluator lifecycle and keyword binding**

Use `create_tracker(self, params, mode=None)`. At sequence start call `begin_episode(reset_global=True)`. Replace the post-initialization `seq.frame_info()`/GT mutation with:

~~~python
safe_info = CausalFrameRecord.from_evaluator(
    frame_index=frame_num,
    previous_output=prev_output,
).as_tracker_info()
out = tracker.track(image, info=safe_info)
~~~

Delete the `info['gt_bbox']` assignment. Keep GT in `Sequence` for evaluator-side metrics only.

- [ ] **Step 5: Normalize tracker signatures and remove GT-only debug access**

Implement the base hook:

~~~python
def begin_episode(self, reset_global=True):
    self._episode_pending_initialization = True

def track(self, image, info=None):
    raise NotImplementedError
~~~

Change `SEATrack.track` to `track(self, image, info=None)`, require `info is not None`, call `assert_causal_tracker_info(info)` first, and remove the unused positional `dataset_name/save_name/seq_name` arguments. Missing causal info fails closed rather than silently inventing a frame. Remove the OSTrack Visdom `info['gt_bbox']` overlay; evaluator-side visualization may render GT without passing it to the tracker.

In Task 1, `SEATrack.begin_episode(reset_global=True)` is controller-neutral: it clears only legacy tracker `state/frame_id/_last_output` and sets the pending-initialization flag, without importing or assuming a Stage 0 controller that is not created until Task 6. `SEATrack.initialize` calls `begin_episode(True)` itself only when no explicit entry point has prepared an episode, then consumes the flag. Add `begin_episode(True)` before initialization in standard OPE, `run_video`, and the VOT wrapper so every official restart is a full reset without a double reset. `run_video` and the VOT wrapper each maintain their own monotonically increasing frame index and sanitized previous prediction, and always call `tracker.track(image, info=CausalFrameRecord(...).as_tracker_info())`.

- [ ] **Step 6: Verify GREEN and legacy evaluation regression**

Run:

~~~bash
.venv/bin/python -m unittest tests.test_spectral_causality -v
.venv/bin/python -m unittest tests.test_bilift_integration tests.test_training_integrity -v
~~~

Expected: all selected tests PASS; the sentinel records zero forbidden-key accesses.

- [ ] **Step 7: Commit A0**

~~~bash
git add lib/test/evaluation/causal.py lib/test/evaluation/tracker.py \
  lib/test/tracker/basetracker.py lib/test/tracker/seatrack.py \
  lib/test/tracker/ostrack.py lib/test/vot/seatrack_class.py \
  tests/test_spectral_causality.py
git commit -m "fix: enforce causal tracker evaluation contract"
~~~

### Task 2: Low-Rank Uncentered-Second-Moment State

**Files:**

- Create: `lib/models/target_spectral/__init__.py`
- Create: `lib/models/target_spectral/types.py`
- Create: `lib/models/target_spectral/memory.py`
- Create: `tests/test_spectral_memory.py`

**Interfaces:**

- `SpectralKey(block: int, site: Literal["attn", "ffn"])`
- `WeightedFactors(vectors: Tensor[M,D], weights: Tensor[M])`
- `MomentState(basis: Tensor[D,R], eigenvalues: Tensor[R], effective_mass: Tensor[], total_trace: Tensor[], next_eigenvalue: Tensor[], relative_eigengap: Tensor[])`
- `FamilySnapshot(anchor, adaptive)` keeps immutable anchor and adaptive identity separate.
- `MemorySnapshot(version, states, trusted_common_means, history_confidence, trusted_asymmetry)` owns detached clones and has no mutating methods.
- `BoundedEigenspaceBank.prepare_frame(factor_chunks_by_key, proposed_trusted_means, q_memory, observed_asymmetry, frame_index)` applies decay once per logical frame across all keys and returns one atomic state-plus-trusted-means-plus-scalar write.
- `BoundedEigenspaceBank.commit(prepared_write)` checks the source version.
- `merge_factor_chunks_across_ranks()` gathers detached factors in deterministic `(rank, chunk, row)` order before the single thin SVD.

- **Memory-test implementation:** add these groups as separate actions:

- [ ] **Step 1a: Add deterministic state/factor/bank fixtures**
- [ ] **Step 1b: Add low-rank update, atomic commit, and next-frame tests**
- [ ] **Step 1c: Add common-rank, alias-byte, and temporary-rank tests**

Cover:

~~~python
import unittest
import dataclasses
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import torch

from lib.models.target_spectral.memory import (
    BoundedEigenspaceBank,
    factor_columns,
    operator_from_thin_svd,
    peak_temporary_bytes,
    select_common_rank,
    trace_energy,
    unique_tensor_storage_bytes,
    update_low_rank_moment,
)
from lib.models.target_spectral.types import MomentState, SpectralKey, WeightedFactors

KEY = SpectralKey(block=5, site="attn")
key = KEY

def empty_state(dimension, dtype=torch.float64):
    scalar = torch.zeros((), dtype=dtype)
    return MomentState(
        basis=torch.empty(dimension, 0, dtype=dtype),
        eigenvalues=torch.empty(0, dtype=dtype),
        effective_mass=scalar.clone(),
        total_trace=scalar.clone(),
        next_eigenvalue=scalar.clone(),
        relative_eigengap=scalar.clone(),
    )

def make_nonempty_state(dimension=12, rank=8):
    eigenvalues = torch.linspace(2.0, 0.6, rank, dtype=torch.float64)
    return MomentState(
        basis=torch.eye(dimension, dtype=torch.float64)[:, :rank],
        eigenvalues=eigenvalues,
        effective_mass=torch.tensor(5.0, dtype=torch.float64),
        total_trace=eigenvalues.sum(),
        next_eigenvalue=torch.tensor(0.4, dtype=torch.float64),
        relative_eigengap=(eigenvalues[-1] - 0.4) / eigenvalues[-1],
    )

def make_factors(dimension, intrinsic_rank, rows=18):
    generator = torch.Generator().manual_seed(20260713)
    latent = torch.randn(
        rows, intrinsic_rank, generator=generator, dtype=torch.float64
    )
    projection = torch.randn(
        intrinsic_rank, dimension, generator=generator, dtype=torch.float64
    )
    vectors = latent @ projection
    weights = torch.arange(1, rows + 1, dtype=torch.float64)
    return WeightedFactors(vectors=vectors, weights=weights)

def make_full_rank_factors(dimension):
    eigenvalues = torch.linspace(2.0, 0.2, dimension, dtype=torch.float64)
    vectors = torch.diag((eigenvalues * dimension).sqrt())
    return WeightedFactors(
        vectors=vectors,
        weights=torch.ones(dimension, dtype=torch.float64),
    )

def chunks():
    value = make_factors(dimension=12, intrinsic_rank=6, rows=10)
    return [
        WeightedFactors(value.vectors[:4], value.weights[:4]),
        WeightedFactors(value.vectors[4:], value.weights[4:]),
    ]

def concat_chunks(*factor_chunks):
    return WeightedFactors(
        vectors=torch.cat([chunk.vectors for chunk in factor_chunks], dim=0),
        weights=torch.cat([chunk.weights for chunk in factor_chunks], dim=0),
    )

def weighted_second_moment(vectors, weights):
    return vectors.transpose(0, 1) @ (weights[:, None] * vectors) / weights.sum()

def best_psd_rank_r(moment, rank):
    eigenvalues, basis = torch.linalg.eigh(moment.to(torch.float64))
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    kept_values = eigenvalues[order].clamp_min(0)
    kept_basis = basis[:, order]
    result = (
        kept_basis * kept_values.unsqueeze(0)
    ) @ kept_basis.transpose(0, 1)
    return result.to(moment.dtype)

def state_with_top_spectrum_and_omitted_trace():
    eigenvalues = torch.tensor(
        [2.0, 1.5, 1.2, 1.0, 0.9, 0.8, 0.6, 0.5],
        dtype=torch.float64,
    )
    return MomentState(
        basis=torch.eye(12, dtype=torch.float64)[:, :8],
        eigenvalues=eigenvalues,
        effective_mass=torch.tensor(20.0, dtype=torch.float64),
        total_trace=torch.tensor(10.0, dtype=torch.float64),
        next_eigenvalue=torch.tensor(0.4, dtype=torch.float64),
        relative_eigengap=torch.tensor(0.2, dtype=torch.float64),
    )

def _assert_tree_close(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(path)
        torch.testing.assert_close(left, right, msg=lambda msg: f"{path}: {msg}")
        return
    if is_dataclass(left):
        if type(left) is not type(right):
            raise AssertionError(path)
        for field in fields(left):
            _assert_tree_close(
                getattr(left, field.name), getattr(right, field.name),
                f"{path}.{field.name}",
            )
        return
    if isinstance(left, Mapping):
        if set(left) != set(right):
            raise AssertionError(path)
        for item_key in left:
            _assert_tree_close(
                left[item_key], right[item_key], f"{path}[{item_key!r}]"
            )
        return
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_tree_close(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")

def assert_states_close(left, right):
    _assert_tree_close(left, right)

def make_bank(history_confidence=1.0, trusted_asymmetry=0.0, scalar_beta=0.90):
    return BoundedEigenspaceBank(
        dimensions={KEY: 12},
        families=("identity", "dynamic", "private", "background"),
        rank=8,
        sketch_rank=32,
        beta=0.95,
        scalar_beta=scalar_beta,
        minimum_effective_mass=4.0,
        eigengap_relative=0.01,
        history_confidence=history_confidence,
        trusted_asymmetry=trusted_asymmetry,
    )

chunk_a, chunk_b = chunks()
mean_a = torch.zeros(12, dtype=torch.float64)
proposed_mean = torch.linspace(-0.5, 0.5, 12, dtype=torch.float64)
factors = {"identity": chunks()}
bank = make_bank()

def test_zero_admission_returns_bitwise_same_state(self):
    before = make_nonempty_state()
    after = update_low_rank_moment(before, chunks(), q=0.0, beta=0.95, rank=8)
    self.assertIs(after, before)
    self.assertTrue(torch.equal(after.basis, before.basis))
    self.assertTrue(torch.equal(after.effective_mass, before.effective_mass))
    self.assertTrue(torch.equal(after.total_trace, before.total_trace))
    self.assertTrue(torch.equal(after.next_eigenvalue, before.next_eigenvalue))

def test_incremental_matches_explicit_uncentered_second_moment(self):
    factors = make_factors(dimension=12, intrinsic_rank=6)
    state = update_low_rank_moment(empty_state(12), [factors], q=0.8, beta=0.95, rank=8)
    explicit = weighted_second_moment(factors.vectors, factors.weights)
    torch.testing.assert_close(state.as_dense(), explicit, atol=1e-5, rtol=1e-4)

def test_rank_cap_matches_best_truncation_on_full_rank_first_update(self):
    factors = make_full_rank_factors(dimension=12)
    state = update_low_rank_moment(empty_state(12), [factors], q=1.0, beta=0.95, rank=8)
    explicit_rank8 = best_psd_rank_r(weighted_second_moment(
        factors.vectors, factors.weights
    ), rank=8)
    torch.testing.assert_close(state.as_dense(), explicit_rank8, atol=1e-5, rtol=1e-4)

def test_frame_partition_does_not_repeat_decay(self):
    one_chunk = bank.prepare_frame(
        {key: {"identity": [chunk_a, chunk_b]}},
        proposed_trusted_means={key: mean_a}, q_memory=0.7,
        observed_asymmetry=0.1, frame_index=4,
    )
    two_chunks = bank.prepare_frame(
        {key: {"identity": [concat_chunks(chunk_a, chunk_b)]}},
        proposed_trusted_means={key: mean_a}, q_memory=0.7,
        observed_asymmetry=0.1, frame_index=4,
    )
    assert_states_close(one_chunk, two_chunks)

def test_rank_energy_uses_exact_total_trace(self):
    state = state_with_top_spectrum_and_omitted_trace()
    self.assertLess(trace_energy(state, rank=8), 0.90)
    self.assertEqual(trace_energy(state, rank=8), state.eigenvalues[:8].sum() / state.total_trace)

def test_commit_is_atomic_and_scalar_state_is_next_frame_only(self):
    bank = make_bank(history_confidence=0.25, trusted_asymmetry=0.10, scalar_beta=0.90)
    before = bank.snapshot()
    prepared = bank.prepare_frame(
        {key: factors}, proposed_trusted_means={key: proposed_mean},
        q_memory=0.8, observed_asymmetry=-0.2, frame_index=3
    )
    self.assertEqual(bank.snapshot().version, before.version)
    bank.commit(prepared)
    after = bank.snapshot()
    self.assertEqual(after.version, before.version + 1)
    self.assertAlmostEqual(float(after.history_confidence), 0.90 * 0.25 + 0.10 * 0.80)
    self.assertAlmostEqual(float(after.trusted_asymmetry), 0.90 * 0.10 + 0.10 * -0.20)
    torch.testing.assert_close(after.trusted_common_means[key], proposed_mean)
~~~

Also test sorted nonnegative eigenvalues, finite shrinkage weights, basis orthogonality, rank cap, rank-deficient first updates, exact boundary eigengap, stale-version commit rejection, anchor immutability, trusted-mean clone isolation/no-admit identity, one version increment for all four keys, deterministic two-rank factor merge, equality of single-process/DDP states, and the exact inactive-to-active transition at the registered effective-mass/eigengap boundary becoming visible only on the following frame.

~~~python
class RankAndAccountingTests(unittest.TestCase):
    def test_common_rank_uses_exact_trace_and_boundary_gap(self):
        eigenvalues = torch.cat((
            torch.ones(16, dtype=torch.float64),
            torch.full((16,), 0.01, dtype=torch.float64),
        ))
        state = MomentState(
            basis=torch.eye(32, dtype=torch.float64),
            eigenvalues=eigenvalues,
            effective_mass=torch.tensor(40.0, dtype=torch.float64),
            total_trace=eigenvalues.sum(),
            next_eigenvalue=torch.zeros((), dtype=torch.float64),
            relative_eigengap=torch.zeros((), dtype=torch.float64),
        )
        self.assertEqual(select_common_rank({"one": state}), 16)

    def test_no_candidate_fails_closed(self):
        state = make_nonempty_state()
        state = MomentState(
            basis=state.basis,
            eigenvalues=state.eigenvalues,
            effective_mass=state.effective_mass,
            total_trace=torch.tensor(1000.0, dtype=torch.float64),
            next_eigenvalue=state.next_eigenvalue,
            relative_eigengap=state.relative_eigengap,
        )
        with self.assertRaisesRegex(ValueError, "no common rank"):
            select_common_rank({"one": state})

    def test_storage_views_are_counted_once(self):
        base = torch.empty(64, dtype=torch.float32)
        self.assertEqual(
            unique_tensor_storage_bytes((base, base[:8], base[8:])),
            base.untyped_storage().nbytes(),
        )

    def test_peak_workspace_counts_distinct_storage(self):
        left = torch.empty(16, dtype=torch.float32)
        right = torch.empty(8, dtype=torch.float64)
        self.assertEqual(
            peak_temporary_bytes(left, right),
            left.untyped_storage().nbytes() + right.untyped_storage().nbytes(),
        )

    def test_orthogonal_anchor_and_adaptive_keep_exact_rank_two_r(self):
        anchor = MomentState(
            basis=torch.eye(4, dtype=torch.float64)[:, :2],
            eigenvalues=torch.ones(2, dtype=torch.float64),
            effective_mass=torch.tensor(20.0, dtype=torch.float64),
            total_trace=torch.tensor(2.0, dtype=torch.float64),
            next_eigenvalue=torch.zeros((), dtype=torch.float64),
            relative_eigengap=torch.ones((), dtype=torch.float64),
        )
        adaptive = dataclasses.replace(
            anchor, basis=torch.eye(4, dtype=torch.float64)[:, 2:]
        )
        columns = torch.cat((
            math.sqrt(0.5) * factor_columns(anchor),
            math.sqrt(0.5) * factor_columns(adaptive),
        ), dim=1)
        operator = operator_from_thin_svd(
            columns, max_rank=4, truncate_nonzero=False
        )
        self.assertEqual(operator.basis.shape[1], 4)
        self.assertEqual(int(torch.linalg.matrix_rank(operator.basis)), 4)
~~~

Run: `.venv/bin/python -m unittest tests.test_spectral_memory.RankAndAccountingTests -v`

Expected before implementation: FAIL because the rank/accounting functions are not exported. Expected after implementation: five tests PASS and aliased views are counted once.

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_memory -v`

Expected: FAIL because `lib.models.target_spectral.memory` does not exist.

- [ ] **Step 3: Implement the thin-SVD update without constructing a dense moment**

~~~python
@dataclass(frozen=True)
class MomentState:
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    effective_mass: torch.Tensor
    total_trace: torch.Tensor
    next_eigenvalue: torch.Tensor
    relative_eigengap: torch.Tensor

    def as_dense(self):
        return (
            self.basis * self.eigenvalues.clamp_min(0).unsqueeze(0)
        ) @ self.basis.transpose(0, 1)

def update_low_rank_moment(state, factor_chunks, q, beta, rank, sketch_rank=32):
    q_value = float(q)
    if q_value == 0.0:
        return state

    vectors = torch.cat([chunk.vectors for chunk in factor_chunks], dim=0)
    weights = torch.cat([chunk.weights for chunk in factor_chunks], dim=0)
    mass = weights.sum()
    new_mass = beta * state.effective_mass + q_value * mass
    weighted_norm2 = (weights * vectors.square().sum(dim=1)).sum()
    new_total_trace = (
        beta * state.effective_mass * state.total_trace
        + q_value * weighted_norm2
    ) / new_mass
    columns = []
    if state.eigenvalues.numel():
        old = state.basis * state.eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
        columns.append(old * (beta * state.effective_mass / new_mass).sqrt())
    new = vectors.transpose(0, 1) * weights.clamp_min(0).sqrt().unsqueeze(0)
    columns.append(new * (q_value / new_mass).sqrt())
    matrix = torch.cat(columns, dim=1)
    basis, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    eigenvalues = singular.square()
    order = torch.argsort(eigenvalues, descending=True)
    kept_count = min(rank, sketch_rank, eigenvalues.numel())
    kept = order[:kept_count]
    boundary = (
        eigenvalues[order[rank]]
        if eigenvalues.numel() > rank else eigenvalues.new_zeros(())
    )
    lambda_r = (
        eigenvalues[order[rank - 1]]
        if eigenvalues.numel() >= rank else eigenvalues.new_zeros(())
    )
    relative_gap = torch.where(
        lambda_r > 1e-12,
        (lambda_r - boundary) / lambda_r.clamp_min(1e-12),
        lambda_r.new_zeros(()),
    )
    return MomentState(
        basis=basis[:, kept].detach(),
        eigenvalues=eigenvalues[kept].clamp_min(0).detach(),
        effective_mass=new_mass.detach(),
        total_trace=new_total_trace.detach(),
        next_eigenvalue=boundary.detach(),
        relative_eigengap=relative_gap.detach(),
    )
~~~

Calibration calls this with `rank=32` and retains `lambda_33`; a frozen gate call uses the registry-selected rank and stores its corresponding `lambda_(r+1)`. Empty mass, negative/nonfinite weights, or dimension mismatch rejects that family before any write. `merge_factor_chunks_across_ranks()` uses `torch.distributed.all_gather_object`, sorts CPU-detached chunks by source rank and local ordinal, and lets rank 0 perform the canonical float64 thin SVD. For each basis column, rank 0 makes the largest-absolute entry (lowest-index tie) nonnegative, then broadcasts the complete detached `MomentState` to every rank. DDP ranks require byte-identical state hashes after broadcast; single-process versus DDP compares the matrix-free operator at relative tolerance `1e-4`, not raw bases, because a degenerate eigenspace may rotate without changing the operator.

- [ ] **Step 4: Implement matrix-free shrinkage operators and separate identity anchor**

~~~python
@dataclass(frozen=True)
class SpectralOperator:
    basis: torch.Tensor
    shrinkage: torch.Tensor

    def apply_rows(self, rows):
        projected = rows @ self.basis
        return (projected * self.shrinkage) @ self.basis.transpose(-1, -2)

def state_operator(state, eps):
    mean_eigenvalue = state.eigenvalues.mean().clamp_min(eps)
    shrinkage = state.eigenvalues / (state.eigenvalues + eps * mean_eigenvalue)
    return SpectralOperator(state.basis, shrinkage)

def state_is_active(state, locked_rank, minimum_effective_mass, eigengap_relative):
    tensors = (
        state.basis, state.eigenvalues, state.effective_mass, state.total_trace,
        state.next_eigenvalue, state.relative_eigengap,
    )
    return (
        locked_rank > 0
        and state.basis.ndim == 2
        and state.eigenvalues.ndim == 1
        and state.basis.shape[1] == state.eigenvalues.numel()
        and state.eigenvalues.numel() >= locked_rank
        and all(bool(torch.isfinite(value).all()) for value in tensors)
        and bool((state.eigenvalues >= 0).all())
        and float(state.effective_mass) >= float(minimum_effective_mass)
        and float(state.relative_eigengap) >= float(eigengap_relative)
    )

def factor_columns(state):
    return state.basis * state.eigenvalues.clamp_min(0).sqrt().unsqueeze(0)

def operator_from_thin_svd(columns, max_rank, truncate_nonzero=False, eps=1e-12):
    if columns.ndim != 2:
        raise ValueError("factor columns must have shape [D,K]")
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    if columns.shape[1] == 0:
        return SpectralOperator(
            basis=columns.new_empty(columns.shape[0], 0),
            shrinkage=columns.new_empty(0),
        )
    basis, singular, _ = torch.linalg.svd(columns, full_matrices=False)
    kept_count = min(int(max_rank), singular.numel())
    if truncate_nonzero and singular.numel():
        tolerance = (
            singular.max() * max(columns.shape) * torch.finfo(singular.dtype).eps
        )
        kept_count = min(kept_count, int((singular > tolerance).sum()))
    basis = basis[:, :kept_count]
    eigenvalues = singular[:kept_count].square()
    if eigenvalues.numel() == 0:
        shrinkage = eigenvalues
    else:
        mean_eigenvalue = eigenvalues.mean().clamp_min(eps)
        shrinkage = eigenvalues / (eigenvalues + eps * mean_eigenvalue)
    return SpectralOperator(
        basis=basis.detach(), shrinkage=shrinkage.detach()
    )
~~~

`state_operator()` is called only after `state_is_active()` passes. For adaptive identity, dynamic, private, and background, activity requires finite tensors, `effective_mass >= minimum_effective_mass`, at least `locked_rank` retained eigenvalues, and the stored locked-boundary `relative_eigengap >= eigengap_relative`. The immutable initialization anchor is the sole exception and is active immediately after successful anchor finalization. Before an adaptive identity state passes, the identity operator contains only `sqrt(lambda_0) * anchor` columns; other inactive families contribute exactly zero. A commit that first crosses an activation boundary cannot route the current frame and becomes visible only in snapshot `t+1`.

For identity evaluation, normalize the template anchor and search anchor independently and form `M_anchor = 0.5*M_template + 0.5*M_search`. Then apply the registry identity-anchor weight `lambda_0` exactly:

~~~python
identity_columns = torch.cat([
    math.sqrt(lambda_0) * factor_columns(anchor_state),
    math.sqrt(1.0 - lambda_0) * factor_columns(adaptive_identity_state),
], dim=1)
pi_identity = operator_from_thin_svd(
    identity_columns,
    max_rank=2 * locked_rank,
    truncate_nonzero=False,
)
~~~

This represents `lambda_0*M_anchor + (1-lambda_0)*M_adaptive`. The immutable anchor and adaptive identity are each individually rank-capped at `locked_rank`; the temporary evaluation operator retains every nonzero singular direction of their concatenation, up to `2*locked_rank`, and is never jointly truncated back to rank `r`. Never pool template/search by token count and never write the temporary combined operator back to either source. Test an orthogonal rank-`r` anchor/adaptive fixture whose temporary operator has exact rank `2r`, and include its basis/SVD workspace in reported peak temporary bytes.

- [ ] **Step 5: Add rank selection and byte accounting**

Use exact total trace, the stored boundary eigenvalue, and unique tensor-storage bytes:

~~~python
import dataclasses
from collections.abc import Mapping

def trace_energy(state, rank):
    if rank < 1 or state.eigenvalues.numel() < rank:
        return state.total_trace.new_zeros(())
    return state.eigenvalues[:rank].sum() / state.total_trace.clamp_min(1e-12)

def boundary_eigenvalue(state, rank):
    if state.eigenvalues.numel() > rank:
        return state.eigenvalues[rank]
    if state.eigenvalues.numel() == rank:
        return state.next_eigenvalue
    return state.eigenvalues.new_zeros(())

def rank_gap(state, rank, eps=1e-12):
    if state.eigenvalues.numel() < rank:
        return state.eigenvalues.new_zeros(())
    lambda_r = state.eigenvalues[rank - 1]
    lambda_next = boundary_eigenvalue(state, rank)
    return (lambda_r - lambda_next).clamp_min(0) / lambda_r.clamp_min(eps)

def select_common_rank(states, candidates=(8, 16, 32), energy_min=0.90, gap_min=0.01):
    ordered_states = tuple(states[key] for key in sorted(states, key=str))
    if not ordered_states:
        raise ValueError("rank selection requires persistent states")
    for rank in candidates:
        if all(
            float(trace_energy(state, rank)) >= energy_min
            and float(rank_gap(state, rank)) >= gap_min
            for state in ordered_states
        ):
            return int(rank)
    raise ValueError("no common rank reaches exact-trace energy and eigengap gates")

def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _iter_tensors(getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _iter_tensors(value[key])
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)

def unique_tensor_storage_bytes(objects):
    storages = {}
    for value in objects:
        for tensor in _iter_tensors(value):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            storages[key] = storage.nbytes()
    return sum(storages.values())

def persistent_state_bytes(snapshot, inert_padding=()):
    return unique_tensor_storage_bytes((snapshot, inert_padding))

def peak_temporary_bytes(*workspace_tensors):
    return unique_tensor_storage_bytes(workspace_tensors)
~~~

Gate confirmation calls `trace_energy()` and `rank_gap()` at the already locked rank for every persistent family/key and may only accept or reject. The temporary up-to-`2r` identity concatenation is passed to `peak_temporary_bytes()` together with its SVD basis/singular-value workspace and is exempt from a second rank selection because its two persistent sources already passed the locked-rank rule.

- [ ] **Step 6: Verify focused tests**

Run: `.venv/bin/python -m unittest tests.test_spectral_memory -v`

Expected: all spectral-memory tests PASS; the rank-at-most-eight fixture matches its full matrix and the full-rank fixture matches the explicit best rank-eight truncation within relative error `1e-4`.

- [ ] **Step 7: Commit spectral memory**

~~~bash
git add lib/models/target_spectral/__init__.py \
  lib/models/target_spectral/types.py lib/models/target_spectral/memory.py \
  tests/test_spectral_memory.py
git commit -m "feat: add bounded target spectral memory"
~~~

### Task 3: Paired RGB-X Observation and Exact Target/Background Factors

**Files:**

- Create: `lib/models/target_spectral/observation.py`
- Modify: `lib/models/target_spectral/__init__.py`
- Create: `tests/test_spectral_observation.py`

**Interfaces:**

- `PairedSpectralObserver.capture(key, scope, raw_h, rgb_indices, x_indices, slots)`
- `PairedObservation` contains detached aligned RGB/X raw features, repeated token-global indices, and slot IDs.
- `downsample_valid_mask(padding_mask, grid_hw)` converts `sample_target`'s `True=padding` mask by nearest-neighbour downsampling and inversion.
- `response_target_weights(score_map, output_window, predicted_box_crop_xywh, valid_mask, crop_hw, temperature, eps)`
- `hard_background_weights(full_probability, predicted_box_crop_xywh, valid_mask, crop_hw, box_scale_multiplier, top_fraction, eps)`
- `build_frame_factors(observations, target_weights, background_weights, p_trusted, paired_valid)`

- **Observation-test implementation:** add these focused groups separately:

- [ ] **Step 1a: Add raw-tensor pairing, detachment, and slot-mass tests**
- [ ] **Step 1b: Add batched target, padding, rasterization, and background tests**
- [ ] **Step 1c: Add four-family unit-mass and dynamic-mean tests**

Tests must assert:

- RMS normalization is observer-only and never mutates `raw_h`.
- RGB/X are paired by the sorted intersection of global indices.
- RGB/X rows are paired by `(token_global_index, slot_id)` and each token weight is divided by `slots`, so slot expansion preserves unit mass.
- Shuffling changes only paired observer rows, never the legacy input tensor.
- Target response is normalized over the full valid grid before predicted-box masking and re-normalization.
- An all-zero/invalid target mass rejects the entire write.
- `response`, `valid_mask`, and rasterized masks all have shape `[B,1,H,W]`; a two-batch test prevents accidental `B x B` broadcasting.
- Box coordinates are current search-crop pixel `xywh`; response-cell centres and crop clipping match the registered rasterization rule.
- A box wholly outside the left/top crop remains empty after clipping; clipping must not translate it into the crop.
- Background uses only top-response valid tokens outside a predicted box whose width and height are multiplied about the fixed centre by the registered `background_box_scale_multiplier`; `1.0` means no expansion and `1.5` means exactly `1.5w x 1.5h`, not a 150% additive expansion.
- Equal-score background ties select ascending flattened global index.
- No eligible background returns `None` and skips only the background family.
- A `True` padding pixel is invalid after nearest-neighbour grid downsampling, including initialization `x0`.
- Target, background, template-anchor, and search-anchor weights each retain unit mass after slot expansion.
- Dynamic uses `delta_p = p_t - p_trusted`; the trusted mean changes only after an actual admitted commit.
- All staged tensors have `requires_grad=False` and no `grad_fn`.

~~~python
import math
import unittest

import torch

from lib.models.target_spectral.observation import (
    PairedObservation,
    PairedSpectralObserver,
    build_frame_factors,
    downsample_valid_mask,
    hard_background_weights,
    response_target_weights,
)
from lib.models.target_spectral.types import SpectralKey

KEY = SpectralKey(block=5, site="attn")

class SpectralObservationTests(unittest.TestCase):
    def test_capture_pairs_sorted_global_index_and_slot_without_mutation(self):
        raw_h = torch.tensor([[
            [1., 2.], [2., 1.], [3., 4.], [4., 3.],
            [5., 6.], [6., 5.], [7., 8.], [8., 7.],
        ]], requires_grad=True)
        before = raw_h.detach().clone()
        observer = PairedSpectralObserver(eps=1e-6)
        observer.capture(
            KEY, "search", raw_h,
            rgb_indices=torch.tensor([[5, 2]]),
            x_indices=torch.tensor([[2, 5]]),
            slots=2,
        )
        observation = observer._staged[0]
        torch.testing.assert_close(raw_h.detach(), before)
        self.assertEqual(observation.global_indices.tolist(), [2, 2, 5, 5])
        self.assertEqual(observation.slot_ids.tolist(), [0, 1, 0, 1])
        torch.testing.assert_close(
            observation.rgb.square().mean(-1).sqrt(),
            torch.ones(4), atol=2e-6, rtol=0,
        )
        for tensor in (
            observation.rgb, observation.x,
            observation.global_indices, observation.slot_ids,
        ):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_slot_expansion_preserves_unit_mass(self):
        token_weights = torch.tensor([0.25, 0.75])
        row_weights = token_weights.repeat_interleave(2) / 2
        self.assertEqual(float(row_weights.sum()), 1.0)
        torch.testing.assert_close(
            row_weights.reshape(2, 2).sum(-1), token_weights
        )

    def test_masks_are_batched_without_batch_by_batch_broadcast(self):
        score = torch.tensor([
            [[[1., 2.], [3., 4.]]],
            [[[4., 3.], [2., 1.]]],
        ])
        valid = torch.ones_like(score, dtype=torch.bool)
        boxes = torch.tensor([[0., 0., 2., 4.], [2., 0., 2., 4.]])
        target, full = response_target_weights(
            score, torch.ones(1, 1, 2, 2), boxes, valid,
            crop_hw=(4, 4), temperature=1.0, eps=1e-8,
        )
        self.assertEqual(target.shape, (2, 1, 2, 2))
        self.assertEqual(full.shape, (2, 1, 2, 2))
        torch.testing.assert_close(full.sum((-2, -1)), torch.ones(2, 1))
        torch.testing.assert_close(target.sum((-2, -1)), torch.ones(2, 1))
        self.assertEqual(int(torch.count_nonzero(target[0, :, :, 1])), 0)
        self.assertEqual(int(torch.count_nonzero(target[1, :, :, 0])), 0)

    def test_zero_or_outside_target_rejects_without_translating_box(self):
        valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        target, full = response_target_weights(
            torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2),
            torch.tensor([[0., 0., 4., 4.]]), valid, crop_hw=(4, 4),
        )
        self.assertIsNone(target)
        self.assertIsNone(full)
        target, full = response_target_weights(
            torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2),
            torch.tensor([[-3., -3., 1., 1.]]), valid, crop_hw=(4, 4),
        )
        self.assertIsNone(target)
        self.assertIsNotNone(full)

    def test_padding_downsample_inverts_true_padding(self):
        padding = torch.zeros(4, 4, dtype=torch.bool)
        padding[:2, :2] = True
        valid = downsample_valid_mask(padding, (2, 2))
        expected = torch.tensor([[[[False, True], [True, True]]]])
        self.assertTrue(torch.equal(valid, expected))

    def test_background_ties_use_lowest_eligible_global_index(self):
        full = torch.full((1, 1, 2, 2), 0.25)
        valid = torch.ones_like(full, dtype=torch.bool)
        background = hard_background_weights(
            full, torch.tensor([[0., 0., 2., 4.]]), valid,
            crop_hw=(4, 4), box_scale_multiplier=1.0,
            top_fraction=0.5, eps=1e-8,
        )
        torch.testing.assert_close(
            background.flatten(), torch.tensor([0., 1., 0., 0.])
        )
        none_left = hard_background_weights(
            full, torch.tensor([[0., 0., 4., 4.]]), valid,
            crop_hw=(4, 4), box_scale_multiplier=1.0,
            top_fraction=0.5, eps=1e-8,
        )
        self.assertIsNone(none_left)

    def test_registered_background_multiplier_is_exact_and_clips_without_translation(self):
        full = torch.full((1, 1, 6, 6), 1.0 / 36.0)
        valid = torch.ones_like(full, dtype=torch.bool)
        background = hard_background_weights(
            full, torch.tensor([[2., 2., 2., 2.]]), valid,
            crop_hw=(6, 6), box_scale_multiplier=1.5,
            top_fraction=1.0, eps=1e-8,
        )
        self.assertEqual(int(torch.count_nonzero(background)), 27)
        self.assertEqual(int(torch.count_nonzero(background[..., 1:4, 1:4])), 0)
        clipped = hard_background_weights(
            full, torch.tensor([[0., 0., 2., 2.]]), valid,
            crop_hw=(6, 6), box_scale_multiplier=1.5,
            top_fraction=1.0, eps=1e-8,
        )
        self.assertEqual(int(torch.count_nonzero(clipped)), 32)
        self.assertEqual(int(torch.count_nonzero(clipped[..., :2, :2])), 0)

    def test_four_factor_families_are_detached_and_unit_mass(self):
        rgb = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        x = torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]])
        observation = PairedObservation(
            key=KEY, scope="search", rgb=rgb, x=x,
            global_indices=torch.tensor([0, 0, 1, 1]),
            slot_ids=torch.tensor([0, 1, 0, 1]), slots=2,
        )
        target = torch.tensor([[[[0.75, 0.25]]]])
        background = torch.tensor([[[[0.25, 0.75]]]])
        trusted = torch.tensor([0.1, -0.2])
        frame = build_frame_factors(
            [observation], target, background, trusted, paired_valid=True
        )
        for family in (frame.identity, frame.private, frame.background):
            self.assertAlmostEqual(float(family.weights.sum()), 1.0)
            self.assertFalse(family.vectors.requires_grad)
            self.assertFalse(family.weights.requires_grad)
        common = (rgb + x) / math.sqrt(2.0)
        row_weights = torch.tensor([0.375, 0.375, 0.125, 0.125])
        expected_mean = (common * row_weights[:, None]).sum(0)
        torch.testing.assert_close(frame.proposed_trusted_mean, expected_mean)
        torch.testing.assert_close(
            frame.dynamic.vectors[0], expected_mean - trusted
        )
        self.assertEqual(float(frame.dynamic.weights.sum()), 1.0)
        self.assertIsNone(build_frame_factors(
            [observation], target, background, trusted, paired_valid=False
        ))
~~~

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_observation -v`

Expected: test discovery succeeds, then import fails only because `lib.models.target_spectral.observation` has not been implemented; no `NameError` or undefined fixture is permitted.

- [ ] **Step 3: Implement paired raw-feature capture**

~~~python
def observer_rms_copy(raw, eps):
    rms = raw.square().mean(dim=-1, keepdim=True).sqrt()
    return raw.detach() / (rms.detach() + eps)

def common_private(rgb, x):
    scale = math.sqrt(2.0)
    return (rgb + x) / scale, (rgb - x) / scale

class PairedSpectralObserver:
    def capture(self, key, scope, raw_h, rgb_indices, x_indices, slots):
        rgb_raw, x_raw = split_modalities(raw_h, slots)
        rgb_rows, x_rows, common_indices, slot_ids = intersect_slot_rows(
            rgb_raw, x_raw, rgb_indices, x_indices, slots
        )
        self._staged.append(PairedObservation(
            key=key,
            scope=scope,
            rgb=observer_rms_copy(rgb_rows, self.eps).clone(),
            x=observer_rms_copy(x_rows, self.eps).clone(),
            global_indices=common_indices.detach().clone(),
            slot_ids=slot_ids.detach().clone(),
            slots=int(slots),
        ))
~~~

- [ ] **Step 4: Implement registry-driven weights**

~~~python
def downsample_valid_mask(padding_mask, grid_hw, device=None):
    padding_mask = torch.as_tensor(padding_mask, device=device, dtype=torch.bool)
    if padding_mask.ndim == 2:
        padding_mask = padding_mask.unsqueeze(0)
    assert padding_mask.ndim == 3
    padding = F.interpolate(
        padding_mask[:, None].float(), size=grid_hw, mode="nearest"
    ).bool()
    return ~padding

def rasterize_box_crop_xywh(boxes, grid_hw, crop_hw):
    batch = boxes.shape[0]
    height, width = grid_hw
    crop_height, crop_width = crop_hw
    x = (torch.arange(width, device=boxes.device) + 0.5) * crop_width / width
    y = (torch.arange(height, device=boxes.device) + 0.5) * crop_height / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    raw_x0, raw_y0, box_w, box_h = boxes.unbind(dim=-1)
    raw_x1 = raw_x0 + box_w.clamp_min(0)
    raw_y1 = raw_y0 + box_h.clamp_min(0)
    x0 = raw_x0.clamp(0, crop_width)
    y0 = raw_y0.clamp(0, crop_height)
    x1 = raw_x1.clamp(0, crop_width)
    y1 = raw_y1.clamp(0, crop_height)
    return (
        (xx[None] >= x0[:, None, None]) & (xx[None] < x1[:, None, None])
        & (yy[None] >= y0[:, None, None]) & (yy[None] < y1[:, None, None])
    )[:, None]

def response_target_weights(
    score_map, output_window, box_crop_xywh, valid_mask, crop_hw,
    temperature=1.0, eps=1e-8,
):
    assert score_map.ndim == valid_mask.ndim == 4
    assert score_map.shape[1] == valid_mask.shape[1] == 1
    response = (score_map * output_window).clamp_min(0).pow(1.0 / temperature)
    valid_mass = response * valid_mask.to(response.dtype)
    full_mass = valid_mass.sum(dim=(-2, -1), keepdim=True)
    if (full_mass <= eps).any() or not torch.isfinite(full_mass).all():
        return None, None
    full = valid_mass / full_mass
    inside = rasterize_box_crop_xywh(box_crop_xywh, response.shape[-2:], crop_hw)
    assert inside.shape == valid_mask.shape == response.shape
    selected = full * inside
    mass = selected.sum(dim=(-2, -1), keepdim=True)
    if (mass <= eps).any() or not torch.isfinite(mass).all():
        return None, full
    return selected / mass, full
~~~

Implement background selection against that same full-grid probability:

~~~python
def hard_background_weights(
    full_probability, box_crop_xywh, valid_mask, crop_hw,
    box_scale_multiplier=1.5, top_fraction=0.10, eps=1e-8,
):
    if full_probability.shape != valid_mask.shape or full_probability.ndim != 4:
        raise ValueError("background probability/valid mask must share [B,1,H,W]")
    scale = float(box_scale_multiplier)
    fraction = float(top_fraction)
    boxes = torch.as_tensor(
        box_crop_xywh, device=full_probability.device, dtype=full_probability.dtype
    )
    if boxes.ndim != 2 or boxes.shape != (full_probability.shape[0], 4):
        raise ValueError("background boxes must be finite [B,4] crop xywh")
    if (
        not math.isfinite(scale)
        or not math.isfinite(fraction)
        or not math.isfinite(float(eps))
        or scale < 1.0
        or not 0.0 < fraction <= 1.0
        or eps <= 0.0
        or not bool(torch.isfinite(boxes).all())
        or bool((boxes[:, 2:] < 0).any())
    ):
        raise ValueError("invalid background selection parameters")
    x, y, width, height = boxes.unbind(-1)
    cx = x + 0.5 * width
    cy = y + 0.5 * height
    dilated_width = width * scale
    dilated_height = height * scale
    dilated = torch.stack((
        cx - 0.5 * dilated_width,
        cy - 0.5 * dilated_height,
        dilated_width,
        dilated_height,
    ), dim=-1)
    inside = rasterize_box_crop_xywh(
        dilated, full_probability.shape[-2:], crop_hw
    )
    eligible = valid_mask & ~inside
    output = torch.zeros_like(full_probability)
    for batch_index in range(full_probability.shape[0]):
        eligible_index = torch.nonzero(
            eligible[batch_index].reshape(-1), as_tuple=False
        ).flatten()
        if eligible_index.numel() == 0:
            return None
        values = full_probability[batch_index].reshape(-1)[eligible_index]
        order = torch.argsort(values, descending=True, stable=True)
        keep_count = math.ceil(fraction * eligible_index.numel())
        kept_index = eligible_index[order[:keep_count]]
        output[batch_index].reshape(-1)[kept_index] = (
            full_probability[batch_index].reshape(-1)[kept_index]
        )
        mass = output[batch_index].sum()
        if not torch.isfinite(mass) or float(mass) <= eps:
            return None
        output[batch_index] /= mass
    return output
~~~

Stable descending sort makes equal-score ties select the lower flattened global index. A missing eligible cell skips only the background family. Registry freezing stores both `background_box_scale_multiplier=1.5` and the derived semantic string `width_height_multiplier_about_center`; validation rejects the old ambiguous `background_box_dilation` key.

- [ ] **Step 5: Build four operational factor families**

For each key/scope, align response weights to sorted global indices and read `p_trusted = transaction.snapshot.trusted_common_means.get(key)`. Expand token weights to rows as `row_weight = token_weight[global_index] / observation.slots`; assert the sum over all slots for a token equals its original token weight and each normalized family mass remains one. If `paired_valid=False`, skip all four paired families. Otherwise use common target rows for identity, difference target rows for private, common hard-background rows for background, and compute:

~~~python
target_common = common[target_indices]
target_weight = target_weights[target_global_indices] / observation.slots
p_t = (target_common * target_weight[:, None]).sum(0) / target_weight.sum()
dynamic = None if p_trusted is None else WeightedFactors(
    vectors=(p_t - p_trusted).unsqueeze(0),
    weights=p_t.new_ones(1),
)
return FrameFactors(
    identity=WeightedFactors(target_common, target_weight),
    private=WeightedFactors(private[target_indices], target_weight),
    dynamic=dynamic,
    background=background_factors,
    proposed_trusted_mean=p_t.detach(),
)
~~~

This represents `(p_t-p_trusted)(p_t-p_trusted)^T` in the factor update. Initialization sets `p_trusted` from the separately normalized anchor-search common mean and leaves dynamic inactive. Only an actual admitted commit replaces it with `proposed_trusted_mean`; scheduled admission followed by factor rejection leaves it bitwise unchanged. Do not store a signed private mean.

- [ ] **Step 6: Verify observation behavior**

~~~bash
.venv/bin/python -m unittest tests.test_spectral_observation tests.test_spectral_memory -v
~~~

Expected: tests PASS; no observation tensor retains a graph.

- [ ] **Step 7: Commit exact observation geometry**

~~~bash
git add lib/models/target_spectral/__init__.py \
  lib/models/target_spectral/observation.py tests/test_spectral_observation.py
git commit -m "feat: observe paired target and background spectra"
~~~

### Task 4: History-Conditioned Search Routing and Temperature-Normalized Bounds

**Files:**

- Create: `lib/models/target_spectral/routing.py`
- Modify: `lib/models/target_spectral/types.py`
- Modify: `lib/models/target_spectral/__init__.py`
- Create: `tests/test_spectral_routing.py`

**Interfaces:**

- `SharedRoutingCoefficients(alpha_budget: float, initial_u: Tensor | None = None)` produces coefficients ordered `identity,dynamic,private,background`.
- `HMoECallContext` identifies key, scope, immutable snapshot, prior, observer mode, route mode, strength, and control. Confidence and asymmetry are read only from that snapshot.
- `HistoryConditionedRouter.route_rows(raw_h, call_context)` returns a separate routed copy.
- `bound_router_delta(delta, d_temp, c_temp, dispatch_budget, combine_budget, slots, temp_floor)`.

- [ ] **Step 1: Write failing route-direction, search-only, and bound tests**

~~~python
def test_coefficients_are_global_nonnegative_and_budgeted(self):
    coeff = SharedRoutingCoefficients(
        alpha_budget=0.25,
        initial_u=torch.tensor([0.2, -0.1, 0.7, 0.0]),
    )
    self.assertTrue((coeff.alpha >= 0).all())
    torch.testing.assert_close(coeff.alpha.sum(), torch.tensor(0.25))

def test_template_scope_is_legacy(self):
    routed = router.route_rows(raw, template_call)
    self.assertTrue(torch.equal(routed, raw))

def test_normalized_dispatch_and_combine_residuals_are_bounded(self):
    bounded = bound_router_delta(delta, d_temp, c_temp, 0.2, 0.1, slots=2)
    self.assertLessEqual((bounded / d_temp.abs().clamp_min(1e-4)).abs().max(), 0.2)
    self.assertLessEqual(
        (bounded * 4 / c_temp.abs().clamp_min(1e-4)).abs().max(), 0.1
    )

def test_nonfinite_or_subfloor_real_temperature_fails_closed(self):
    for invalid in (torch.tensor(0.0), torch.tensor(1e-5), torch.tensor(float("nan"))):
        with self.assertRaisesRegex(ValueError, "temperature"):
            bound_router_delta(
                delta, invalid, torch.tensor(1.0), 0.2, 0.1,
                slots=2, temp_floor=1e-4,
            )
~~~

Also test opposite private signs for RGB/X, background subtraction, causal prior expansion over slots, empty/inactive families, operator cap, and zero-strength early bypass with a spy that raises if observer/operator code executes.

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_routing -v`

Expected: FAIL on missing `routing.py`.

- [ ] **Step 3: Implement one global softmax coefficient vector**

~~~python
class SharedRoutingCoefficients(nn.Module):
    ORDER = ("identity", "dynamic", "private", "background")

    def __init__(self, alpha_budget, initial_u=None):
        super().__init__()
        value = torch.zeros(4) if initial_u is None else initial_u.detach().clone()
        self.u = nn.Parameter(value)
        self.alpha_budget = float(alpha_budget)

    @property
    def alpha(self):
        return self.alpha_budget * torch.softmax(self.u, dim=0)
~~~

The fitted module lives in the offline fitter only. Construct it on the single asserted fit device; its master `u/alpha` stay float32, and routing casts `alpha` differentiably to the current `rows.dtype` without changing device. S0 loads `alpha.detach()` once as an external frozen float32 tensor on the tracker's device, validates four finite entries and exact budget, and casts it to each call's row dtype; it does not register `u` on the tracker model. CPU and CUDA tests fail on device mismatch and compare the same synthetic routed result within registered tolerance.

- [ ] **Step 4: Implement matrix-free RGB/X row routing**

For each search token, apply:

~~~python
target_update = (
    alpha_id * pi_identity.apply_rows(rows)
    + alpha_dynamic * pi_dynamic.apply_rows(rows)
    + private_sign * alpha_private * pi_private.apply_rows(rows)
)
background_update = alpha_background * pi_background.apply_rows(rows)
update = prior * target_update - (1.0 - prior) * background_update
routed = rows + strength * snapshot.history_confidence * operator_scale * update
~~~

Use `private_sign=snapshot.trusted_asymmetry` for RGB and its negative for X. The snapshot scalars are detached committed values from frame `t-1`; do not read controller mutable fields. Because each shrinkage operator has norm at most one and coefficient mass is `alpha_budget`, set `operator_scale=min(1, kappa/max(alpha_budget, eps))`; require `kappa < 1`. Except for the explicit observer-only anchor mode, `FrameRouteContext.call_for()` converts `strength == 0` or a snapshot with no active operator to `hard_disabled=True` before HMoE is called, so no observer/operator/residual arithmetic runs.

- [ ] **Step 5: Implement the common raw residual-logit bound**

~~~python
def bound_router_delta(
    delta, d_temp, c_temp, dispatch_budget, combine_budget, slots=2, temp_floor=1e-4
):
    if (
        not isinstance(slots, int)
        or slots < 1
        or not math.isfinite(float(temp_floor))
        or temp_floor <= 0.0
        or not math.isfinite(float(dispatch_budget))
        or not math.isfinite(float(combine_budget))
        or dispatch_budget < 0.0
        or combine_budget < 0.0
        or not bool(torch.isfinite(delta).all())
    ):
        raise ValueError("invalid residual-logit bound inputs")
    d = d_temp.detach().abs()
    c = c_temp.detach().abs()
    if (
        d.numel() != 1
        or c.numel() != 1
        or not bool(torch.isfinite(d).all())
        or not bool(torch.isfinite(c).all())
        or bool((d < temp_floor).any())
        or bool((c < temp_floor).any())
    ):
        raise ValueError("real HMoE temperature is nonfinite or below floor")
    raw_limit = torch.minimum(
        d * dispatch_budget,
        c * combine_budget / float(slots * slots),
    )
    return delta.clamp(min=-raw_limit, max=raw_limit)
~~~

Do not modify learned temperature values. At construction and before every active routed call, require each learned dispatch/combine temperature to be finite with absolute value at least the registered floor; fail closed otherwise. The bound uses the real detached absolute temperatures, never an upward clamp, so the normalized perturbation cannot exceed its budget. Disabled/no-context paths do not add this validation and preserve legacy behavior.

- [ ] **Step 6: Verify bounded routing**

~~~bash
.venv/bin/python -m unittest tests.test_spectral_routing tests.test_spectral_memory -v
~~~

Expected: all tests PASS; the same raw delta satisfies Dispatch and Combine normalized budgets.

- [ ] **Step 7: Commit bounded routing**

~~~bash
git add lib/models/target_spectral/__init__.py \
  lib/models/target_spectral/types.py lib/models/target_spectral/routing.py \
  tests/test_spectral_routing.py
git commit -m "feat: add bounded history conditioned routing"
~~~

### Task 5: HMoE, Transformer, and Configuration Integration

**Files:**

- Modify: `lib/models/layers/attn.py:66-112`
- Modify: `lib/models/layers/attn_blocks.py:279-352`
- Modify: `lib/models/seatrack/vit_ci.py:31-262`
- Modify: `lib/models/seatrack/seatrack.py:19-60,158-189`
- Modify: `lib/config/seatrack/config.py`
- Create: `tests/test_spectral_integration.py`

**Interfaces:**

- `HMoE.forward(x, mode=None, route_call=None)`.
- `CEBlock_AP.forward(x, global_index_template, global_search_idx, mask=None, ce_template_mask=None, keep_ratio_search=None, spectral_context=None)`.
- `VisionTransformerCE.forward_features(z, x, mask_z=None, mask_x=None, ce_template_mask=None, ce_keep_rate=None, return_last_attn=False, spectral_context=None)`.
- `VisionTransformerCE.forward(z, x, ce_template_mask=None, ce_keep_rate=None, tnc_keep_rate=None, return_last_attn=False, spectral_context=None)`.
- `SEATrack.forward(template, search, ce_template_mask=None, ce_keep_rate=None, return_last_attn=False, spectral_context=None)`.
- `SmokeSEATrack.forward(..., spectral_context=None)` accepts disabled/no-context calls and raises for active target-spectral routing.
- `SpectralDiagnostics` captures detached per-token Combine and attention evidence keyed by `(block, site, scope, modality, partition)`.
- `MODEL.TARGET_SPECTRAL` defaults disabled and preserves all existing YAML behavior.

- [ ] **Step 1: Write failing exact-identity and raw-expert-input tests**

Test matched HMoE modules with and without disabled/empty/zero-strength contexts using `torch.equal`. Add a capture expert:

~~~python
class CaptureExperts(nn.Module):
    def forward(self, expert_inputs):
        self.last_input = expert_inputs.detach().clone()
        return torch.zeros_like(expert_inputs)

def test_routing_changes_logits_but_expert_dispatch_uses_raw_h(self):
    module = make_hmoe()
    module.experts = CaptureExperts()
    captured = {}
    original = module._routing_weights
    def capture_weights(logits):
        dispatch, combine = original(logits)
        captured["dispatch"] = dispatch.detach()
        return dispatch, combine
    module._routing_weights = capture_weights
    dim = tokens.shape[-1]
    raw_h = module.linear1(module.norm(tokens)).reshape(
        1, -1, dim // module.size_slots
    )
    module(tokens, mode="search", route_call=active_call)
    expected_flat = torch.bmm(
        captured["dispatch"].transpose(1, 2), raw_h
    )
    expected_expert_input = expected_flat.reshape(
        tokens.shape[0], module.size_experts, module.size_slots,
        dim // module.size_slots,
    )
    self.assertEqual(
        module.experts.last_input.shape, expected_expert_input.shape
    )
    torch.testing.assert_close(
        module.experts.last_input, expected_expert_input
    )
    torch.testing.assert_close(
        module.experts.last_input.flatten(), expected_flat.flatten()
    )
~~~

Also test one snapshot object identity at all eight calls (2 blocks x 2 sites x 2 scopes), search-only routing, blocks 5/9 only, `VisionTransformerCE.forward` propagation, disabled smoke-model compatibility, and separable block/site/scope/modality/target-background diagnostics. A spy must prove observer-only capture does not execute routed `bmm`, delta bounding, clamp, or `base + 0` arithmetic.

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_integration -v`

Expected: FAIL because HMoE and model forwards do not accept spectral contexts.

- [ ] **Step 3: Preserve raw HMoE semantics while bounding routed logits**

Refactor `HMoE.forward` in this order:

~~~python
def _routing_weights(self, logits):
    batch = logits.shape[0]
    tokens = logits.shape[1] // self.size_slots
    dispatch = F.softmax(logits / self.D_temp, dim=1)
    combine_logits = logits.reshape(
        batch, tokens, self.size_slots,
        self.size_slots * self.size_experts,
    ).sum(dim=2).reshape(
        batch, tokens, self.size_experts, self.size_slots,
    ).sum(dim=-1)
    combine = F.softmax(combine_logits / self.C_temp, dim=-1)
    return dispatch, combine

self.router_stats = {}
raw_h = self.linear1(self.norm(x)).reshape(
    batch, tokens * self.size_slots, dim // self.size_slots
)
gate = self.gate_thi.unsqueeze(0).expand(batch, -1, -1)
base_logits = torch.bmm(raw_h, gate)
logit_drift = None

if route_call is None or route_call.hard_disabled:
    logits = base_logits
elif route_call.observe and not route_call.route:
    route_call.observer.capture_from_call(route_call, raw_h)
    logits = base_logits
else:
    if route_call.observe:
        route_call.observer.capture_from_call(route_call, raw_h)
    routed_h = route_call.router.route_rows(raw_h, route_call)
    routed_logits = torch.bmm(routed_h, gate)
    delta = bound_router_delta(
        routed_logits - base_logits,
        self.D_temp,
        self.C_temp,
        route_call.dispatch_budget,
        route_call.combine_budget,
        slots=self.size_slots,
        temp_floor=route_call.temp_floor,
    )
    logits = base_logits + delta
    logit_drift = delta.detach()

dispatch, combine = self._routing_weights(logits)
with torch.no_grad():
    combine_detached = combine.detach().clamp_min(1e-9)
    entropy = -(combine_detached * combine_detached.log()).sum(dim=-1)
    entropy = entropy / math.log(max(self.size_experts, 2))
    expert_load = combine_detached.mean(dim=(0, 1))
    self.router_stats = {
        "Router/entropy": entropy.mean(),
        "Router/expert_load_max": expert_load.max(),
        "Router/expert_load_std": expert_load.std(unbiased=False),
    }
if (
    route_call is not None
    and not route_call.hard_disabled
    and route_call.observe
):
    route_call.observer.capture_combine(
        route_call, combine.detach(), logit_drift
    )
dispatched_h = torch.bmm(dispatch.transpose(1, 2), raw_h)
experts_inputs = dispatched_h.reshape(
    batch, self.size_experts, self.size_slots, dim // self.size_slots
)
experts_outputs = self.experts(experts_inputs).reshape(
    batch, self.size_experts, dim
)
experts_outputs = self.linear2(experts_outputs)
moe_out = torch.bmm(combine, experts_outputs)
return moe_out
~~~

Never use `routed_h` for `experts_inputs`. Preserve the exact legacy four-dimensional `MultiExpertLinear` input shape `[B,E,S,D/S]`, expert output flattening `[B,E,D]`, `router_stats` reset, and entropy/load keys exactly as shown so existing expert semantics, LoRA adapters, diagnostics, and regressions remain intact. The raw-expert-input test must fail if either the expert/slot axes are flattened before `MultiExpertLinear` or their order is permuted.

- [ ] **Step 4: Propagate one frame context without changing other HMoE layers**

At blocks 5 and 9, request calls with keys `(block,"attn")` and `(block,"ffn")`. Each call carries current RGB/X global token indices and the valid token mask. Template calls are observer-only during anchor capture and otherwise hard-disabled. Search calls may observe and route. All other HMoE layers receive `None` and run the exact legacy branch. Thread `spectral_context` explicitly through both `VisionTransformerCE.forward` and `forward_features`; never store it on a module attribute.

Reuse the parameter-free attention calculation already embodied by `compute_gra_stats` without enabling GRA. In each spectral block, compute the template-to-search distributions directly from `brgb_attn` and `bdte_attn`, align them to `global_search_idx`, mask invalid search tokens, normalize each distribution over common valid support, detach, and call:

~~~python
if spectral_context is not None and spectral_context.observe_evidence(i):
    spectral_context.observer.capture_attention_evidence(
        block=i,
        rgb_probability=spectral_attention_probability(
            brgb_attn, lens_t, global_search_idx[0], valid_search_mask
        ).detach(),
        x_probability=spectral_attention_probability(
            bdte_attn, lens_t, global_search_idx[1], valid_search_mask
        ).detach(),
        rgb_global_indices=global_search_idx[0].detach(),
        x_global_indices=global_search_idx[1].detach(),
)
~~~

`spectral_attention_probability()` softmaxes attention logits, selects template-query to search-key entries, averages template queries and heads with equal weight, gathers sorted valid global search indices, and L1-normalizes. After both blocks complete, intersect RGB/X indices across blocks 5 and 9, gather each already-normalized block distribution to that support and renormalize, then define `p_rgb = 0.5*p_rgb_block5 + 0.5*p_rgb_block9` and likewise for X. Renormalize the two aggregates once more. This fixed block/head/query aggregation is registry-versioned; no data-dependent block weighting is allowed.

The production `aux_dict["target_spectral"]` payload contains detached per-token Combine, attention evidence, global indices, and residual-logit drift for every block/site/scope/modality. After the crop-space prediction is available, `Stage0Controller.after_prediction()` combines that payload with the registered masks and returns a detached `FrameSpectralDiagnostics` containing target/background Combine, RGB-vs-X Jensen-Shannon divergence, top-1 and top-2 expert overlap, and normalized residual-logit L2 drift. Tests inspect these returned payloads; production modules do not retain test activations.

- [ ] **Step 5: Add fail-closed configuration**

Add disabled defaults:

~~~python
cfg.MODEL.TARGET_SPECTRAL = edict()
cfg.MODEL.TARGET_SPECTRAL.ENABLED = False
cfg.MODEL.TARGET_SPECTRAL.STAGE = "disabled"
cfg.MODEL.TARGET_SPECTRAL.LAYERS = [5, 9]
cfg.MODEL.TARGET_SPECTRAL.MODULES = ["attn", "ffn"]
cfg.MODEL.TARGET_SPECTRAL.OBSERVE_SCOPES = ["template", "search"]
cfg.MODEL.TARGET_SPECTRAL.ROUTE_SCOPES = ["search"]
cfg.MODEL.TARGET_SPECTRAL.RANK = 16
cfg.MODEL.TARGET_SPECTRAL.RANK_CANDIDATES = [8, 16, 32]
cfg.MODEL.TARGET_SPECTRAL.RANK_SOURCE = "calibration_fallback"
cfg.MODEL.TARGET_SPECTRAL.CONTROL = "routing_disabled_legacy"
cfg.MODEL.TARGET_SPECTRAL.STRENGTH = 1.0
cfg.MODEL.TARGET_SPECTRAL.COEFFICIENT_CHECKPOINT = ""
cfg.MODEL.TARGET_SPECTRAL.REGISTRY = ""
cfg.MODEL.CHECKPOINT_LORA_WEIGHT_STATE = "merged"
~~~

`validate_target_spectral_config()` returns immediately when `TARGET_SPECTRAL.ENABLED=False`; existing GRA/BiLift/legacy experiments must not be constrained by an inactive feature. Calibration YAML may use provisional rank 16 with `RANK_SOURCE: "calibration_fallback"`. Every frozen/gate YAML uses `RANK_SOURCE: "registry"`; enabled construction fails if the frozen registry lacks `selected_rank` or if the YAML rank disagrees. Enabled construction also rejects non-search route scope, layers other than `[5,9]` in the core config, CE keep ratio below one, disabled HMoE, `GRA.ENABLED`, `GRA.DIAGNOSTICS`, `BILIFT.ENABLED`, `BILIFT.DIAGNOSTICS`, any ProbAlign/LiftTrack construction flag, nonpositive temperature floor, any active block-5/9 HMoE dispatch/combine temperature that is nonfinite or has absolute value below that floor, `kappa >= 1`, active strength other than exactly `1.0`, or unsupported Stage R/E keys. Tests enumerate those flags independently so diagnostics-only GRA/BiLift cannot bypass mutual exclusion. The named zero-strength control overrides strength to exactly `0.0` and hard-bypasses; every other active row must expose the registry value `1.0` exactly. `SmokeSEATrack` ignores `None`/hard-disabled contexts and raises `ValueError("smoke model has no target-spectral routing")` for an active context.

- [ ] **Step 6: Verify legacy and spectral integration**

Run:

~~~bash
.venv/bin/python -m unittest tests.test_spectral_integration tests.test_bilift_integration -v
.venv/bin/python -m unittest tests.test_training_integrity -v
~~~

Expected: all tests PASS; a disabled full model remains bitwise identical for the same state dict and input.

- [ ] **Step 7: Commit integration**

~~~bash
git add lib/config/seatrack/config.py lib/models/layers/attn.py \
  lib/models/layers/attn_blocks.py lib/models/seatrack/vit_ci.py \
  lib/models/seatrack/seatrack.py tests/test_spectral_integration.py
git commit -m "feat: integrate target spectral HMoE routing"
~~~

### Task 6: Stage 0 Snapshot Controller and Initialization Anchor

**Files:**

- Create: `lib/models/target_spectral/stage0.py`
- Modify: `lib/models/target_spectral/__init__.py`
- Modify: `lib/models/seatrack/vit_ci.py`
- Modify: `lib/models/seatrack/seatrack.py`
- Modify: `lib/train/data/processing_utils.py`
- Modify: `lib/test/tracker/seatrack.py`
- Create: `tests/test_spectral_stage0_tracker.py`

**Interfaces:**

- `SEATrack.spectral_controller` is `None` unless `cfg.MODEL.TARGET_SPECTRAL.ENABLED` is true; disabled lifecycle and tracking never enter Stage 0 code.
- `Stage0Controller.begin_episode(reset_global=True)`
- `Stage0Controller.anchor_context(*, init_box, template_transform, search_transform, template_valid_mask, search_valid_mask) -> FrameRouteContext`
- `Stage0Controller.finalize_anchor(anchor_context) -> None`
- `Stage0Controller.before_frame(frame_index, crop_transform, target_prior, valid_mask) -> tuple[FrameTransaction, FrameRouteContext]`
- `Stage0Controller.after_prediction(transaction, outputs, predicted_box_crop_xywh, committed_box_image_xywh) -> FrameSpectralDiagnostics`
- `FrameTransaction` carries one snapshot version and rejects double commit.
- `AdmissionRecord(scheduled_admit, q_memory, asymmetry, paired_valid)` is immutable and, in S0, comes from the sealed common schedule.
- `FrozenRuleAdmissionSource.after_prediction(frame_key, observables) -> AdmissionRecord` is used on fit/calibration streams after threshold/floor lock.
- `SealedScheduleAdmissionSource.after_prediction(frame_key, observables) -> AdmissionRecord` is used by matched S0 rows and validates the sealed key/hash.
- `decode_windowed_center_head(box_head, outputs, output_window, search_size)` returns the exact batched crop-pixel `cxcywh [B,4]`, `xywh [B,4]`, best-score tensor `[B]`, and fused response `[B,1,H,W]` used by tracking.
- `CropTransform(origin_xy, crop_size, resize_factor, output_size)` reproduces `sample_target`'s exact Python `ceil/round` geometry and is the only image/crop coordinate mapping used by active anchor, prior, and rollout evidence paths.

Add the shared transform without changing `sample_target` itself, its return tuple, or any disabled-path crop arithmetic:

~~~python
@dataclass(frozen=True)
class CropTransform:
    origin_xy: tuple[int, int]
    crop_size: int
    resize_factor: float
    output_size: int

    @classmethod
    def from_target(cls, target_bb, search_area_factor, output_size):
        values = torch.as_tensor(target_bb).detach().cpu().tolist()
        if len(values) != 4:
            raise ValueError("target box must be xywh")
        x, y, width, height = (float(value) for value in values)
        factor = float(search_area_factor)
        output = int(output_size)
        if (
            not all(math.isfinite(value) for value in (x, y, width, height, factor))
            or width <= 0.0
            or height <= 0.0
            or factor <= 0.0
            or output < 1
            or float(output_size) != float(output)
        ):
            raise ValueError("invalid target crop geometry")
        crop_size = math.ceil(
            math.sqrt(width * height) * factor
        )
        if crop_size < 1:
            raise ValueError("too small bounding box")
        origin_x = round(x + 0.5 * width - 0.5 * crop_size)
        origin_y = round(y + 0.5 * height - 0.5 * crop_size)
        return cls(
            origin_xy=(int(origin_x), int(origin_y)),
            crop_size=int(crop_size),
            resize_factor=float(output) / float(crop_size),
            output_size=output,
        )

    def __post_init__(self):
        if (
            self.crop_size < 1
            or self.output_size < 1
            or not math.isfinite(float(self.resize_factor))
            or self.resize_factor <= 0.0
        ):
            raise ValueError("crop/output size must be positive")
        expected = self.output_size / self.crop_size
        if not math.isclose(self.resize_factor, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("crop transform resize factor is inconsistent")

    def image_xywh_to_crop(self, box, *, device=None, dtype=torch.float32):
        value = torch.as_tensor(box, device=device, dtype=dtype).clone()
        origin = value.new_tensor(self.origin_xy)
        value[..., :2] = (value[..., :2] - origin) * self.resize_factor
        value[..., 2:] *= self.resize_factor
        return value

    def crop_cxcywh_to_image_xywh(self, box):
        value = torch.as_tensor(box).clone()
        origin = value.new_tensor(self.origin_xy)
        centre = value[..., :2] / self.resize_factor + origin
        size = value[..., 2:] / self.resize_factor
        return torch.cat((centre - 0.5 * size, size), dim=-1)
~~~

Active code constructs `CropTransform.from_target()` only after the unchanged three-return `sample_target()` call and requires its `resize_factor` to equal the returned factor exactly. Add a golden test that runs `sample_target` and `CropTransform.from_target` on the same image/box, requires equal resize factors and exact origin/size for half-integer, negative-origin, and padded-boundary cases. Patch `CropTransform.from_target` to raise in the disabled tracker test; disabled initialize/track must still pass, proving the disabled path never constructs it.

- **Lifecycle-test implementation:** add these CPU-only fixtures as separate actions:

- [ ] **Step 1a: Add fake Center Head, network, preprocessor, and crop spy**
- [ ] **Step 1b: Add disabled and enabled initialization tests**
- [ ] **Step 1c: Add enabled predict-before-commit and all-padding tests**
- [ ] **Step 1d: Add stale, premature, and double-transaction tests**

Test:

- With the default disabled config, no controller is constructed; `begin_episode()` makes no controller call, `initialize()` makes no search-size `x0` crop/network forward, and each `track()` performs exactly the original single no-context forward/decode with no `target_spectral` payload. Its output is bitwise equal to a pre-change legacy fixture.
- Initialization runs exactly one discarded calibration forward with normal template and search-size crops centred on the initialization box; the keyword-only anchor API receives the two crop-specific `CropTransform` objects and valid masks without swapping them.
- Calibration has observer enabled and routing disabled.
- Only immutable identity anchor becomes active; private/dynamic/background remain inactive.
- The first real tracking forward can see the anchor.
- All calls within a frame share the exact same snapshot object.
- Bank version cannot change during forward or before prediction commit.
- Frame `t` write is first visible at `t+1`.
- Double commit, stale transaction, and commit-before-prediction raise.
- An admission-source spy raises if the current frame rule/schedule is consulted before prediction commit.
- `begin_episode(reset_global=True)` clears object state and rebuilds the anchor.
- Initialization scalar state is exactly `history_confidence=1`, `trusted_asymmetry=0`; the first real frame sees those values.
- No-admit and factor-rejection frames preserve scalar state bitwise and increment separate counters from actual commits.
- Changing only padded response cells does not change `q_memory`; block/head permutation preserving values does not change the fixed evidence aggregate. Controller observables retain the legacy Center-Head `best_score` as a detached finite tensor of shape `[B]` for diagnostics even though the public tracker result exposes a Python scalar, but this unmasked score is never used in `q_memory`.
- Add an end-to-end decoder/controller padding fixture, not only a helper test: hold the decoded Center-Head box/score fixed, run two otherwise identical pending-frame transactions whose observer tensors differ only in padded cells, and require identical `q_memory`, schedule/admission/factor-rejection/actual-commit bits, post-frame snapshot hash, public box, and public score. In the all-padding case both transactions must produce zero valid evidence, zero prior mass, no actual commit, and an unchanged state hash.
- The causal prior is the exact binary cell-centre raster of the previous committed box in the current crop; padding is always zero, and a boundary/all-padding fixture cannot create prior mass.
- `SEATrack.track()` returns detached `target_spectral` diagnostics to the S0 runner, while `sanitize_previous_output()` drops that field so it never feeds the next frame. A runner integration test records the payload verbatim, then feeds the whole tracker output through the sanitizer and proves the payload is absent from the next frame's `previous_output`.
- Model parameter hashes are unchanged across a synthetic sequence.

~~~python
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from lib.models.target_spectral.stage0 import FrameTransaction, valid_response_peak
from lib.test.evaluation.causal import CausalFrameRecord, sanitize_previous_output
from lib.test.tracker.seatrack import SEATrack
from lib.train.data.processing_utils import CropTransform

class FakeBoxHead:
    def cal_bbox(self, response, size_map, offset_map, return_score=True):
        batch = response.shape[0]
        boxes = response.new_tensor([0.5, 0.5, 0.25, 0.25])
        boxes = boxes.reshape(1, 1, 4).expand(batch, 1, 4).clone()
        scores = response.new_full((batch, 1), 0.9)
        return boxes, scores

class FakeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unchanged_parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.box_head = FakeBoxHead()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        batch = kwargs["search"].shape[0]
        score = kwargs["search"].new_tensor(
            [[[[0.1, 0.2], [0.3, 0.9]]]]
        ).expand(batch, -1, -1, -1).clone()
        score = score + 0.0 * self.unchanged_parameter
        return {
            "score_map": score,
            "size_map": torch.zeros_like(score).expand(-1, 2, -1, -1),
            "offset_map": torch.zeros_like(score).expand(-1, 2, -1, -1),
        }

class FakePreprocessor:
    def process(self, patch):
        return torch.as_tensor(patch).permute(2, 0, 1).unsqueeze(0).float()

class SampleTargetSpy:
    def __init__(self):
        self.calls = []
        self.padding_by_size = {}

    def __call__(self, image, box, factor, output_sz):
        self.calls.append((tuple(box), factor, output_sz))
        patch = np.zeros((output_sz, output_sz, 6), dtype=np.uint8)
        padding = self.padding_by_size.get(
            output_sz, np.zeros((output_sz, output_sz), dtype=bool)
        )
        transform = CropTransform.from_target(box, factor, output_sz)
        return patch, transform.resize_factor, padding.copy()

class FakeDiagnostics:
    def __init__(self):
        self.as_dict_calls = 0
        self.payload = {"q_memory": torch.tensor(0.5)}

    def as_dict(self):
        self.as_dict_calls += 1
        return self.payload

class SpyController:
    def __init__(self):
        self.begin_calls = []
        self.anchor_calls = []
        self.finalize_calls = []
        self.before_calls = []
        self.after_calls = []
        self.anchor = SimpleNamespace(observer_enabled=True, routing_enabled=False)
        self.diagnostics = FakeDiagnostics()
        self.owner = None

    def begin_episode(self, reset_global=True):
        self.begin_calls.append(bool(reset_global))

    def anchor_context(self, **kwargs):
        self.anchor_calls.append(kwargs)
        return self.anchor

    def finalize_anchor(self, anchor_context):
        self.finalize_calls.append(anchor_context)

    def before_frame(self, frame_index, crop_transform, target_prior, valid_mask):
        snapshot = object()
        transaction = SimpleNamespace(snapshot=snapshot)
        context = SimpleNamespace(snapshot=snapshot)
        self.before_calls.append({
            "frame_index": frame_index,
            "crop_transform": crop_transform,
            "target_prior": target_prior.detach().clone(),
            "valid_mask": valid_mask.detach().clone(),
            "transaction": transaction,
            "context": context,
            "state_before": list(self.owner.state),
        })
        return transaction, context

    def after_prediction(self, transaction, outputs,
                         predicted_box_crop_xywh, committed_box_image_xywh):
        self.after_calls.append({
            "transaction": transaction,
            "outputs": outputs,
            "predicted": predicted_box_crop_xywh,
            "committed": committed_box_image_xywh,
        })
        if list(self.owner.state) == self.before_calls[-1]["state_before"]:
            raise AssertionError("prediction state was not committed first")
        return self.diagnostics

def make_tracker(controller=None):
    tracker = object.__new__(SEATrack)
    tracker.params = SimpleNamespace(
        template_factor=2.0, template_size=4,
        search_factor=4.0, search_size=8,
        save_all_boxes=False, debug=0,
    )
    tracker.cfg = SimpleNamespace(MODEL=SimpleNamespace(
        BACKBONE=SimpleNamespace(CE_LOC=[], STRIDE=2),
        NUM_OBJECT_QUERIES=1,
    ))
    tracker.network = FakeNetwork()
    tracker.preprocessor = FakePreprocessor()
    tracker.output_window = torch.ones(1, 1, 2, 2)
    tracker.template_grid_hw = (2, 2)
    tracker.search_grid_hw = (2, 2)
    tracker.box_mask_z = None
    tracker.state = None
    tracker.frame_id = 0
    tracker.debug = 0
    tracker.mode = None
    tracker.save_all_boxes = False
    tracker.spectral_controller = controller
    tracker._episode_pending_initialization = False
    tracker._last_output = None
    if controller is not None:
        controller.owner = tracker
    return tracker

def causal_info(frame_index=1, previous_output=None):
    return CausalFrameRecord.from_evaluator(
        frame_index, previous_output or {}
    ).as_tracker_info()

class Stage0TrackerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.sampler = SampleTargetSpy()
        patcher = mock.patch(
            "lib.test.tracker.seatrack.sample_target", side_effect=self.sampler
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.image = np.zeros((64, 64, 6), dtype=np.uint8)
        self.init_info = {"init_bbox": [10.0, 10.0, 8.0, 8.0]}

    def test_disabled_path_has_no_x0_forward_or_context(self):
        tracker = make_tracker(controller=None)
        tracker.begin_episode(reset_global=True)
        tracker.initialize(self.image, self.init_info)
        self.assertEqual([call[2] for call in self.sampler.calls], [4])
        self.assertEqual(tracker.network.calls, [])
        output = tracker.track(self.image, info=causal_info())
        self.assertEqual([call[2] for call in self.sampler.calls], [4, 8])
        self.assertEqual(len(tracker.network.calls), 1)
        self.assertNotIn("spectral_context", tracker.network.calls[0])
        self.assertEqual(output["target_bbox"], [10.0, 10.0, 10, 10])
        self.assertAlmostEqual(output["best_score"], 0.9, places=6)
        self.assertNotIn("target_spectral", output)

    def test_enabled_initialization_uses_distinct_crop_arguments(self):
        controller = SpyController()
        tracker = make_tracker(controller)
        tracker.begin_episode(reset_global=True)
        tracker.initialize(self.image, self.init_info)
        self.assertEqual(controller.begin_calls, [True])
        self.assertEqual([call[2] for call in self.sampler.calls], [4, 8])
        self.assertEqual(len(tracker.network.calls), 1)
        self.assertIs(tracker.network.calls[0]["spectral_context"], controller.anchor)
        call = controller.anchor_calls[0]
        self.assertEqual(call["template_transform"].origin_xy, (6, 6))
        self.assertEqual(call["search_transform"].origin_xy, (-2, -2))
        self.assertEqual(call["template_transform"].resize_factor, 0.25)
        self.assertEqual(call["search_transform"].resize_factor, 0.25)
        self.assertEqual(call["template_valid_mask"].shape, (1, 1, 2, 2))
        self.assertEqual(call["search_valid_mask"].shape, (1, 1, 2, 2))
        self.assertIs(controller.finalize_calls[0], controller.anchor)
        self.assertTrue(controller.anchor.observer_enabled)
        self.assertFalse(controller.anchor.routing_enabled)

    def test_enabled_track_predicts_then_commits_once(self):
        controller = SpyController()
        tracker = make_tracker(controller)
        tracker.begin_episode(True)
        tracker.initialize(self.image, self.init_info)
        tracker.network.calls.clear()
        before_parameters = {
            name: value.detach().clone()
            for name, value in tracker.network.state_dict().items()
        }
        output = tracker.track(self.image, info=causal_info())
        self.assertEqual(len(tracker.network.calls), 1)
        self.assertIs(
            tracker.network.calls[0]["spectral_context"],
            controller.before_calls[0]["context"],
        )
        self.assertIs(
            controller.before_calls[0]["transaction"].snapshot,
            controller.before_calls[0]["context"].snapshot,
        )
        self.assertEqual(len(controller.after_calls), 1)
        self.assertEqual(controller.after_calls[0]["predicted"].shape, (1, 4))
        self.assertEqual(controller.after_calls[0]["committed"].shape, (1, 4))
        self.assertEqual(controller.diagnostics.as_dict_calls, 1)
        self.assertIs(output["target_spectral"], controller.diagnostics.payload)
        self.assertNotIn("target_spectral", sanitize_previous_output(output))
        for name, before in before_parameters.items():
            self.assertTrue(torch.equal(before, tracker.network.state_dict()[name]))

    def test_all_padding_forces_zero_prior_and_validity(self):
        controller = SpyController()
        tracker = make_tracker(controller)
        tracker.begin_episode(True)
        tracker.initialize(self.image, self.init_info)
        self.sampler.padding_by_size[8] = np.ones((8, 8), dtype=bool)
        tracker.track(self.image, info=causal_info())
        self.assertEqual(
            int(torch.count_nonzero(controller.before_calls[0]["valid_mask"])), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(controller.before_calls[0]["target_prior"])), 0
        )

class FrameTransactionTests(unittest.TestCase):
    def test_commit_order_stale_version_and_double_commit(self):
        snapshot = SimpleNamespace(version=3)
        transaction = FrameTransaction(
            frame_index=4, snapshot=snapshot, observer=object()
        )
        with self.assertRaisesRegex(RuntimeError, "prediction"):
            transaction.mark_memory_committed(current_version=3)
        transaction.mark_prediction_committed()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            transaction.mark_memory_committed(current_version=4)
        transaction.mark_memory_committed(current_version=3)
        with self.assertRaisesRegex(RuntimeError, "already"):
            transaction.mark_memory_committed(current_version=3)

    def test_q_score_peak_ignores_padding_only_response_changes(self):
        valid = torch.tensor([[[[True, False], [True, False]]]])
        first = torch.tensor([[[[0.4, 0.2], [0.3, 0.1]]]])
        second = torch.tensor([[[[0.4, 0.99], [0.3, 0.98]]]])
        torch.testing.assert_close(
            valid_response_peak(first, valid),
            valid_response_peak(second, valid),
        )
        torch.testing.assert_close(
            valid_response_peak(first, valid), torch.tensor([0.4])
        )
~~~

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_stage0_tracker -v`

Expected: test discovery succeeds and fails only on the missing `Stage0Controller`/`decode_windowed_center_head`; no undefined fixture, CUDA allocation, or real checkpoint loading occurs.

- [ ] **Step 3: Implement immutable frame transactions**

~~~python
@dataclass
class FrameTransaction:
    frame_index: int
    snapshot: MemorySnapshot
    observer: PairedSpectralObserver
    prediction_committed: bool = False
    memory_committed: bool = False

    def mark_prediction_committed(self):
        if self.prediction_committed:
            raise RuntimeError("prediction already committed")
        self.prediction_committed = True

    def mark_memory_committed(self, current_version):
        if not self.prediction_committed:
            raise RuntimeError("prediction must commit before memory")
        if self.memory_committed:
            raise RuntimeError("memory already committed")
        if current_version != self.snapshot.version:
            raise RuntimeError(
                f"stale transaction: snapshot={self.snapshot.version}, "
                f"current={current_version}"
            )
        self.memory_committed = True

class Stage0Controller:
    def before_frame(self, frame_index, crop_transform, target_prior, valid_mask):
        snapshot = self.bank.snapshot()
        observer = PairedSpectralObserver(self.observer_config)
        transaction = FrameTransaction(frame_index, snapshot, observer)
        return transaction, FrameRouteContext(
            snapshot=snapshot,
            observer=observer,
            crop_transform=crop_transform,
            target_prior=target_prior.detach(),
            route_scopes=frozenset({"search"}),
        )
~~~

Construct `self.spectral_controller = build_stage0_controller(...)` only inside an `if cfg.MODEL.TARGET_SPECTRAL.ENABLED` branch; otherwise set it to `None` and do not allocate registry/state/observer objects. Task 6 extends the Task 1 lifecycle with `if self.spectral_controller is not None: self.spectral_controller.begin_episode(reset_global)`. For an enabled controller, `begin_episode(True)` replaces the bank, observer state, counters, and pending transaction; it sets initialization scalar state to `history_confidence=1` and `trusted_asymmetry=0`. It preserves the configured admission-source policy and immutable schedule/threshold payload, but calls `admission_source.begin_episode()` to reset only its cursor, bound sequence key, and audit state. `set_admission_source()` therefore survives repeated clip/sequence resets. Test two successive clips and two sealed sequences to prove that the configured source remains active while neither cursor leaks across episodes. `after_prediction` first marks `transaction.prediction_committed=True`, then computes observables and invokes the admission source; no caller evaluates a schedule/rule argument before that mark. It verifies the bank version equals the transaction snapshot version, builds all key/family chunks first, calls one frame-wide `prepare_frame(factor_chunks_by_key, proposed_trusted_means, ...)`, and performs exactly one `bank.commit(prepared_frame_write)`. Counters are distinct: `scheduled_admits`, `factor_rejections`, and `actual_commits`.

- [ ] **Step 4: Add initialization anchor capture**

In `SEATrack.initialize`, the disabled branch ends after the existing template/state setup and returns exactly as before. Only when `self.spectral_controller is not None`, retain/use the normal template crop's third return as `z_padding`, construct the two transforms from the same extraction boxes/factors, verify their resize factors against `sample_target`, and execute:

~~~python
if self.spectral_controller is not None:
    template_transform = CropTransform.from_target(
        info["init_bbox"], self.params.template_factor,
        self.params.template_size,
    )
    if resize_factor != template_transform.resize_factor:
        raise AssertionError("template crop transform mismatch")
    x0_patch, x0_resize, x0_padding = sample_target(
        image, info["init_bbox"], self.params.search_factor,
        output_sz=self.params.search_size,
    )
    search_transform = CropTransform.from_target(
        info["init_bbox"], self.params.search_factor,
        self.params.search_size,
    )
    if x0_resize != search_transform.resize_factor:
        raise AssertionError("search crop transform mismatch")
    x0_tensor = self.preprocessor.process(x0_patch)
    anchor_context = self.spectral_controller.anchor_context(
        init_box=info["init_bbox"],
        template_transform=template_transform,
        search_transform=search_transform,
        template_valid_mask=downsample_valid_mask(
            z_padding, self.template_grid_hw, device=self.z_tensor.device
        ),
        search_valid_mask=downsample_valid_mask(
            x0_padding, self.search_grid_hw, device=x0_tensor.device
        ),
    )
    with torch.no_grad():
        self.network(
            template=self.z_tensor,
            search=x0_tensor,
            ce_template_mask=self.box_mask_z,
            spectral_context=anchor_context,
        )
    self.spectral_controller.finalize_anchor(anchor_context)
~~~

Discard the enabled-only forward output. Never run active spectral routing during this call. With target spectral disabled, neither `sample_target(...search_factor...)` nor `self.network(...)` is called during initialization.

`anchor_context()` maps the legal image-space initialization box into template- and search-crop pixel `xywh` only through the two supplied `CropTransform` objects. It never reconstructs a continuous crop origin. It rasterizes each mask with Task 3's cell-centre rule, intersects its own valid mask, and assigns uniform L1-normalized weight inside that mask. Empty template or search anchor mass fails initialization. Build common identity factors separately for template and search, combine their normalized moments with fixed weights `0.5/0.5`, and never pool by token count. Set the initial trusted common mean from the search-anchor factor only; dynamic remains inactive until the first actual admitted streaming commit.

- [ ] **Step 5: Add GT-free modality asymmetry and confidence observables**

Do not run the Center Head separately on RGB and X. Consume only the fused response from the normal head and the detached template-to-search attention evidence captured in Task 5. On their common valid support, define:

~~~python
def normalized_concentration(probability, eps=1e-8):
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(eps)
    count = probability.shape[-1]
    if count < 2:
        return probability.new_zeros(probability.shape[:-1])
    return ((count * probability.square().sum(-1) - 1.0) / (count - 1)).clamp(0, 1)

def valid_response_peak(fused_response, valid_mask):
    if fused_response.shape != valid_mask.shape or fused_response.ndim != 4:
        raise ValueError("response/valid mask must share [B,1,H,W]")
    if not bool(torch.isfinite(fused_response).all()):
        raise ValueError("fused response must be finite")
    valid = valid_mask.to(torch.bool)
    valid_response = fused_response.clamp_min(0).masked_fill(~valid, 0)
    peak = valid_response.flatten(1).amax(-1).clamp(0, 1)
    has_valid_support = valid.flatten(1).any(-1)
    return torch.where(has_valid_support, peak, torch.zeros_like(peak))

# common_global_indices is the sorted block-5/block-9 RGB/X valid-support intersection.
fused_common = fused_full_probability.flatten(1).gather(1, common_global_indices)
fused_common = fused_common / fused_common.sum(-1, keepdim=True).clamp_min(eps)
c_fused = normalized_concentration(fused_common)
c_rgb = normalized_concentration(p_rgb)
c_x = normalized_concentration(p_x)
count = p_rgb.shape[-1]
agreement = ((count * (p_rgb * p_x).sum(-1) - 1.0) / max(count - 1, 1)).clamp(0, 1)
q_score = valid_response_peak(fused_response, valid_mask)
assert q_score.shape == c_fused.shape == (p_rgb.shape[0],)
q_fused = torch.sqrt(q_score * c_fused)
q_pair = agreement * torch.sqrt((c_rgb * c_x).clamp_min(0))
q_memory = torch.sqrt((q_fused * q_pair).clamp(0, 1))
asymmetry = (c_rgb - c_x) / (c_rgb + c_x + eps)
paired_valid = (torch.minimum(c_rgb, c_x) >= frozen_attention_floor) & (count >= 2)
~~~

Here `fused_response = outputs["fused_response"]` is the normal windowed Center-Head response and `valid_mask` is the current search crop's downsampled nonpadding mask. Reject nonfinite values or shapes other than `[B,1,H,W]`. Retain `outputs["spectral_best_score"]` only as a detached finite `[B]` diagnostic and keep the public `tracker_out["best_score"]` as the batch-zero Python float expected by existing evaluators. The calibration registry defines `attention_floor_quantile=0.10`. Pool finite `min(c_rgb,c_x)` values from all routing-disabled post-initialization calibration frames with common-support count at least two across all six strata, cast to canonical float64, and freeze `numpy.quantile(values,0.10,method="higher")`; frames with count below two remain `paired_valid=False` and do not enter the floor population. Record its ordered-population hash/count and reject an empty population. Detach all values after prediction. For an actual admitted, factor-valid commit only, update snapshot scalars with registry `history_ema_beta=0.90`:

~~~python
rho_next = 0.90 * snapshot.history_confidence + 0.10 * admission_record.q_memory
asymmetry_next = 0.90 * snapshot.trusted_asymmetry + 0.10 * admission_record.asymmetry
~~~

If `scheduled_admit=False`, `paired_valid=False`, target mass is empty, or all paired families reject, leave bank states, trusted mean, confidence, and asymmetry bitwise unchanged. The frame-`t` values become visible only in snapshot `t+1`. A synthetic smoke must show at least 20% scheduled write coverage and make every family cross its activation mass; otherwise calibration configuration fails.

- [ ] **Step 6a: Run the decoder and disabled-path tests RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_stage0_tracker -v`

Expected: FAIL on the missing decoder/legacy helper with every test fixture importing successfully.

- [ ] **Step 6b: Implement the batched active decoder and exact legacy branch**

Add the active decoder to `stage0.py`:

~~~python
@dataclass(frozen=True)
class WindowedCenterHeadDecode:
    query_crop_cxcywh: torch.Tensor
    crop_cxcywh: torch.Tensor
    crop_xywh: torch.Tensor
    best_score: torch.Tensor
    fused_response: torch.Tensor

def decode_windowed_center_head(box_head, outputs, output_window, search_size):
    response = output_window * outputs["score_map"]
    if response.ndim != 4 or response.shape[1] != 1:
        raise ValueError("fused response must have shape [B,1,H,W]")
    boxes, best_score = box_head.cal_bbox(
        response, outputs["size_map"], outputs["offset_map"], return_score=True
    )
    batch = response.shape[0]
    query_boxes = boxes.reshape(batch, -1, 4) * float(search_size)
    crop_cxcywh = query_boxes.mean(dim=1)
    best_score = best_score.reshape(batch, -1)
    if best_score.shape[1] != 1:
        raise ValueError("center head must return one best score per batch")
    best_score = best_score[:, 0]
    cx, cy, width, height = crop_cxcywh.unbind(-1)
    crop_xywh = torch.stack(
        (cx - 0.5 * width, cy - 0.5 * height, width, height), dim=-1
    )
    return WindowedCenterHeadDecode(
        query_crop_cxcywh=query_boxes.detach(),
        crop_cxcywh=crop_cxcywh.detach(),
        crop_xywh=crop_xywh.detach(),
        best_score=best_score.detach(),
        fused_response=response.detach(),
    )
~~~

Set both grid sizes once in the tracker constructor:

~~~python
stride = self.cfg.MODEL.BACKBONE.STRIDE
self.template_grid_hw = (
    self.params.template_size // stride, self.params.template_size // stride,
)
self.search_grid_hw = (
    self.params.search_size // stride, self.params.search_size // stride,
)
~~~

Keep disabled arithmetic in the exact pre-change order; do not call the new decoder on this bitwise-golden path:

~~~python
def _track_legacy_no_context(self, search, resize_factor, image_hw):
    image_height, image_width = image_hw
    with torch.no_grad():
        outputs = self.network.forward(
            template=self.z_tensor,
            search=search,
            ce_template_mask=self.box_mask_z,
        )
    pred_score_map = outputs["score_map"]
    response = self.output_window * pred_score_map
    pred_boxes, best_score = self.network.box_head.cal_bbox(
        response, outputs["size_map"], outputs["offset_map"], return_score=True
    )
    max_score = best_score[0][0].item()
    pred_boxes = pred_boxes.view(-1, 4)
    pred_box = (
        pred_boxes.mean(dim=0) * self.params.search_size / resize_factor
    ).tolist()
    self.state = clip_box(
        self.map_box_back(pred_box, resize_factor),
        image_height, image_width, margin=10,
    )
    self.debug = 0
    result = {"target_bbox": self.state, "best_score": max_score}
    if self.save_all_boxes:
        all_boxes = self.map_box_back_batch(
            pred_boxes * self.params.search_size / resize_factor,
            resize_factor,
        )
        result["all_boxes"] = all_boxes.view(-1).tolist()
    return result
~~~

Run: `.venv/bin/python -m unittest tests.test_spectral_stage0_tracker.Stage0TrackerLifecycleTests.test_disabled_path_has_no_x0_forward_or_context -v`

Expected: PASS with one search forward, no `spectral_context`, and no initialization x0 forward.

- [ ] **Step 6c: Implement enabled prior, prediction, and post-prediction commit**

~~~python
def track(self, image, info=None):
    if info is None:
        raise AssertionError("causal tracker info is required")
    assert_causal_tracker_info(info)
    image_height, image_width, _ = image.shape
    self.frame_id += 1
    if info["frame_index"] != self.frame_id:
        raise AssertionError(
            f"frame index mismatch: tracker={self.frame_id}, "
            f"evaluator={info['frame_index']}"
        )
    patch, resize_factor, padding_mask = sample_target(
        image, self.state, self.params.search_factor,
        output_sz=self.params.search_size,
    )
    search = self.preprocessor.process(patch)
    if self.spectral_controller is None:
        return self._track_legacy_no_context(
            search=search, resize_factor=resize_factor,
            image_hw=(image_height, image_width),
        )

    crop_transform = CropTransform.from_target(
        self.state, self.params.search_factor, self.params.search_size
    )
    if resize_factor != crop_transform.resize_factor:
        raise AssertionError("online crop transform mismatch")
    valid_mask = downsample_valid_mask(
        padding_mask, self.search_grid_hw, device=search.device
    )
    previous_box_crop = crop_transform.image_xywh_to_crop(
        self.state, device=search.device, dtype=search.dtype
    ).unsqueeze(0)
    target_prior = rasterize_box_crop_xywh(
        previous_box_crop, self.search_grid_hw,
        (self.params.search_size, self.params.search_size),
    ) & valid_mask
    transaction, spectral_context = self.spectral_controller.before_frame(
        frame_index=info["frame_index"],
        crop_transform=crop_transform,
        target_prior=target_prior,
        valid_mask=valid_mask,
    )
    with torch.no_grad():
        outputs = self.network.forward(
            template=self.z_tensor, search=search,
            ce_template_mask=self.box_mask_z,
            spectral_context=spectral_context,
        )
    decoded = decode_windowed_center_head(
        self.network.box_head, outputs, self.output_window,
        self.params.search_size,
    )
    if decoded.crop_xywh.shape != (1, 4):
        raise AssertionError("tracker evaluation requires batch one")
    relative_box = decoded.crop_cxcywh[0] / resize_factor
    self.state = clip_box(
        self.map_box_back(relative_box.tolist(), resize_factor),
        image_height, image_width, margin=10,
    )
    max_score = decoded.best_score[0].item()
    outputs["best_score"] = max_score
    outputs["spectral_best_score"] = decoded.best_score
    outputs["fused_response"] = decoded.fused_response
    committed_box = decoded.crop_xywh.new_tensor(self.state).unsqueeze(0)
    diagnostics = self.spectral_controller.after_prediction(
        transaction=transaction,
        outputs=outputs,
        predicted_box_crop_xywh=decoded.crop_xywh,
        committed_box_image_xywh=committed_box.detach(),
    )
    diagnostics_dict = diagnostics.as_dict()
    result = {
        "target_bbox": self.state,
        "best_score": max_score,
        "target_spectral": diagnostics_dict,
    }
    if self.save_all_boxes:
        all_boxes = self.map_box_back_batch(
            decoded.query_crop_cxcywh[0] / resize_factor, resize_factor
        )
        result["all_boxes"] = all_boxes.reshape(-1).tolist()
    return result
~~~

The binary prior is neither dilated, blurred, nor renormalized; padding remains zero. `CropTransform` is used only for anchor/prior/evidence coordinate alignment. Public prediction commit continues through the exact legacy `map_box_back`/`clip_box` arithmetic shown above, preserving routing-disabled and zero-strength identity. Both admission sources run only inside `after_prediction()` after `self.state` is committed. `CausalFrameRecord` removes diagnostics from next-frame history. OPE, video, and every VOT restart call the lifecycle before initialization.

Run: `.venv/bin/python -m unittest tests.test_spectral_stage0_tracker tests.test_spectral_causality -v`

Expected: PASS; state commits before `after_prediction`, diagnostics emit exactly once and disappear from causal history, and no current response changes its own prediction.

- [ ] **Step 7: Verify controller and tracker behavior**

~~~bash
.venv/bin/python -m unittest tests.test_spectral_stage0_tracker \
  tests.test_spectral_integration tests.test_spectral_causality -v
~~~

Expected: tests PASS; no current response affects its own first prediction.

- [ ] **Step 8: Commit the Stage 0 controller**

~~~bash
git add lib/models/target_spectral/__init__.py \
  lib/models/target_spectral/stage0.py lib/models/seatrack/vit_ci.py \
  lib/models/seatrack/seatrack.py lib/test/tracker/seatrack.py \
  tests/test_spectral_stage0_tracker.py
git commit -m "feat: add causal stage zero spectral controller"
~~~

### Task 7: Frozen Manifests, Registry Schema, and Experiment Configs

**Files:**

- Modify: `.gitignore`
- Create: `tools/freeze_target_spectral_splits.py`
- Create: `tools/index_target_spectral_checkpoints.py`
- Modify: `lib/train/dataset/lasher.py`
- Modify: `lib/train/dataset/depthtrack.py`
- Modify: `lib/train/base_functions.py`
- Modify: `lib/train/trainers/base_trainer.py`
- Create: `lib/models/seatrack/checkpoint.py`
- Modify: `lib/test/tracker/seatrack.py`
- Create: `lib/train/data_specs/lasher_spectral_fit.txt`
- Create: `lib/train/data_specs/lasher_spectral_calibration.txt`
- Create: `lib/train/data_specs/lasher_spectral_gate_confirmation.txt`
- Create: `lib/train/data_specs/depthtrack_spectral_fit.txt`
- Create: `lib/train/data_specs/depthtrack_spectral_calibration.txt`
- Create: `lib/train/data_specs/depthtrack_spectral_gate_confirmation.txt`
- Create: `lib/train/data_specs/target_spectral_split_audit.json`
- Create: `experiments/seatrack/registries/spectral_s0_v1.calibration.yaml`
- Create: `experiments/seatrack/rgbt_spectral_s0.yaml`
- Create: `experiments/seatrack/rgbd_spectral_s0.yaml`
- Create: `experiments/seatrack/rgbt_spectral_s0_short.yaml`
- Create: `experiments/seatrack/rgbt_spectral_base.yaml`
- Create: `experiments/seatrack/rgbd_spectral_base.yaml`
- Create: `knowledge_base/Target-Spectral-S0-实验记录.md`
- Create: `knowledge_base/Target-Spectral-S0-base-checkpoints.json`
- Create: `tests/test_spectral_config_registry.py`

**Interfaces:**

- Stable SHA-256 split seed `20260713`, ratio 80/10/10, using LasHeR train+val and DepthTrack train+val only.
- Empty lines and duplicates are rejected.
- Fit/calibration/gate sets are pairwise disjoint and their union equals the source pool.
- Calibration registry is executable for smoke/calibration but gate runners reject it.
- The registry hash chain is acyclic: calibration registry -> global coefficient checkpoint -> frozen registry -> gate schedule manifest -> result rows.
- Every base checkpoint and index entry declares `lora_weight_state` as `merged` or `unmerged`; Workstream A loaders reject missing/conflicting provenance.

- **Registry-test implementation:** add these groups separately:

- [ ] **Step 1a: Add deterministic split and audit-schema tests**
- [ ] **Step 1b: Add S0/base YAML and frozen-status tests**
- [ ] **Step 1c: Add merged/unmerged nonzero-LoRA parity test**
- [ ] **Step 1d: Add root/`/tmp` CLI and Git-visibility tests**

~~~python
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
import yaml

from lib.models.layers.attn import MergedLinear
from lib.models.seatrack.checkpoint import load_seatrack_checkpoint
from lib.models.target_spectral.stage0 import validate_registry
from tools.freeze_target_spectral_splits import stable_partition

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_REGISTRY = (
    REPO_ROOT / "experiments/seatrack/registries/spectral_s0_v1.calibration.yaml"
)
S0_CONFIGS = (
    REPO_ROOT / "experiments/seatrack/rgbt_spectral_s0.yaml",
    REPO_ROOT / "experiments/seatrack/rgbd_spectral_s0.yaml",
    REPO_ROOT / "experiments/seatrack/rgbt_spectral_s0_short.yaml",
)
BASE_CONFIGS = {
    "rgbt": (
        REPO_ROOT / "experiments/seatrack/rgbt_spectral_base.yaml",
        "LasHeR_spectral_fit", 60, 60000,
    ),
    "rgbd": (
        REPO_ROOT / "experiments/seatrack/rgbd_spectral_base.yaml",
        "DepthTrack_spectral_fit", 25, 60000,
    ),
}

def _yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} is not a YAML mapping")
    return value

def _lf_sha256(names):
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()

def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).upper()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)

def _nonzero_lora_layer():
    torch.manual_seed(7)
    layer = MergedLinear(
        4, 6, r=2, lora_alpha=1, lora_dropout=0.0,
        enable_lora=[True, False, True], bias=True,
    )
    with torch.no_grad():
        layer.lora_A.copy_(
            torch.arange(layer.lora_A.numel()).reshape_as(layer.lora_A).float() / 17.0
        )
        layer.lora_B.copy_(
            torch.arange(layer.lora_B.numel()).reshape_as(layer.lora_B).float() / 19.0
        )
    layer.train()
    return layer

class SpectralConfigRegistryTests(unittest.TestCase):
    def test_partition_rejects_empty_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            stable_partition(["a", "", "b"], "lasher")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stable_partition(["a", "a", "b"], "lasher")

    def test_committed_manifests_are_exact_disjoint_union(self):
        audit = json.loads((
            REPO_ROOT / "lib/train/data_specs/target_spectral_split_audit.json"
        ).read_text(encoding="utf-8"))
        for dataset in ("lasher", "depthtrack"):
            split_names = {}
            for split in ("fit", "calibration", "gate_confirmation"):
                path = REPO_ROOT / f"lib/train/data_specs/{dataset}_spectral_{split}.txt"
                raw = path.read_bytes()
                self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
                self.assertNotIn(b"\r", raw)
                names = path.read_text(encoding="utf-8").splitlines()
                self.assertTrue(names)
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(all(
                    name and "visevent" not in name.lower() for name in names
                ))
                split_names[split] = set(names)
                recorded = audit["datasets"][dataset]["outputs"][split]
                self.assertEqual(recorded["count"], len(names))
                self.assertEqual(recorded["sha256"], hashlib.sha256(raw).hexdigest())
            fit = split_names["fit"]
            calibration = split_names["calibration"]
            gate = split_names["gate_confirmation"]
            self.assertFalse(fit & calibration or fit & gate or calibration & gate)
            union = fit | calibration | gate
            source = audit["datasets"][dataset]["source"]
            self.assertEqual(source["count"], len(union))
            self.assertEqual(source["sha256"], _lf_sha256(union))
            self.assertEqual(
                (len(fit), len(calibration), len(gate)),
                (
                    int(len(union) * 0.8),
                    int(len(union) * 0.9) - int(len(union) * 0.8),
                    len(union) - int(len(union) * 0.9),
                ),
            )

    def test_s0_configs_are_exact_calibration_only_overlays(self):
        forbidden = {"UPDATE", "REPLAY", "EMA", "ROLLBACK", "STAGE_R", "STAGE_E", "PROBALIGN"}
        for path in S0_CONFIGS:
            document = _yaml(path)
            target = document["MODEL"]["TARGET_SPECTRAL"]
            self.assertIs(target["ENABLED"], True)
            self.assertEqual(target["STAGE"], "s0")
            self.assertEqual(target["LAYERS"], [5, 9])
            self.assertEqual(target["MODULES"], ["attn", "ffn"])
            self.assertEqual(target["OBSERVE_SCOPES"], ["template", "search"])
            self.assertEqual(target["ROUTE_SCOPES"], ["search"])
            self.assertEqual(target["RANK"], 16)
            self.assertEqual(target["RANK_CANDIDATES"], [8, 16, 32])
            self.assertEqual(target["RANK_SOURCE"], "calibration_fallback")
            self.assertEqual(
                document["MODEL"]["BACKBONE"]["CE_KEEP_RATIO"], [1.0, 1.0, 1.0]
            )
            self.assertTrue(document["MODEL"]["HMOE_ENABLED"])
            self.assertFalse(document["MODEL"]["GRA"]["ENABLED"])
            self.assertFalse(document["MODEL"]["GRA"]["DIAGNOSTICS"])
            self.assertFalse(document["MODEL"]["BILIFT"]["ENABLED"])
            self.assertFalse(document["MODEL"]["BILIFT"]["DIAGNOSTICS"])
            self.assertFalse(forbidden & set(_walk_keys(target)))
            self.assertNotIn("TRAIN", document)

    def test_base_configs_preserve_fit_only_training_protocol(self):
        for _, (path, dataset, epochs, samples) in BASE_CONFIGS.items():
            document = _yaml(path)
            self.assertEqual(document["DATA"]["TRAIN"]["DATASETS_NAME"], [dataset])
            self.assertEqual(document["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"], samples)
            self.assertEqual(document["DATA"]["VAL"]["DATASETS_NAME"], [None])
            self.assertEqual(document["TRAIN"]["EPOCH"], epochs)
            self.assertEqual(document["TRAIN"]["OPTIMIZER"], "ADAMW")
            self.assertEqual(document["TRAIN"]["LR"], 0.0004)
            self.assertIs(document["MODEL"]["DETERMINISTIC_LORA_INIT"], True)
            self.assertIs(document["MODEL"]["TARGET_SPECTRAL"]["ENABLED"], False)
            self.assertIs(document["MODEL"]["GRA"]["ENABLED"], False)
            self.assertIs(document["MODEL"]["BILIFT"]["ENABLED"], False)
            self.assertEqual(document["MODEL"]["BACKBONE"]["CE_KEEP_RATIO"], [1, 1, 1])
            self.assertEqual(document["DATA"]["SEARCH"]["SIZE"], 256)
            self.assertEqual(document["DATA"]["TEMPLATE"]["SIZE"], 128)
            self.assertEqual(document["TRAIN"]["SAVE_EPOCH_INTERVAL"], epochs)

    def test_gate_registry_fails_closed_until_frozen(self):
        registry = _yaml(CALIBRATION_REGISTRY)
        validate_registry(registry, purpose="calibration")
        with self.assertRaisesRegex(ValueError, "frozen registry required"):
            validate_registry(registry, purpose="gate_confirmation")

    def test_merged_and_unmerged_nonzero_lora_checkpoints_are_equivalent(self):
        x = torch.tensor([[0.1, -0.2, 0.3, 0.4]])
        source = _nonzero_lora_layer()
        reference = source(x).detach()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            unmerged = directory / "unmerged.pt"
            merged = directory / "merged.pt"
            torch.save({"net": source.state_dict(), "lora_weight_state": "unmerged"}, unmerged)
            merged_source = copy.deepcopy(source).eval()
            torch.save({"net": merged_source.state_dict(), "lora_weight_state": "merged"}, merged)
            for path, state in ((unmerged, "unmerged"), (merged, "merged")):
                loaded = _nonzero_lora_layer()
                load_seatrack_checkpoint(
                    loaded, path, explicit_state=state, require_metadata=True
                )
                torch.testing.assert_close(loaded(x), reference)
            legacy = _nonzero_lora_layer()
            legacy.load_state_dict(torch.load(unmerged, weights_only=False)["net"])
            legacy.merged = True
            legacy.train()
            legacy.eval()
            self.assertFalse(torch.allclose(legacy(x), reference))

    def test_created_clis_are_cwd_independent_and_paths_git_visible(self):
        scripts = (
            "tools/freeze_target_spectral_splits.py",
            "tools/index_target_spectral_checkpoints.py",
        )
        for script in scripts:
            for cwd in (REPO_ROOT, Path("/tmp")):
                completed = subprocess.run(
                    [str(REPO_ROOT / ".venv/bin/python"), str(REPO_ROOT / script), "--help"],
                    cwd=cwd, capture_output=True, text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout.lower())
        intended = (
            *scripts,
            *(str(path.relative_to(REPO_ROOT)) for path in S0_CONFIGS),
            *(str(value[0].relative_to(REPO_ROOT)) for value in BASE_CONFIGS.values()),
            "experiments/seatrack/registries/spectral_s0_v1.calibration.yaml",
            "knowledge_base/Target-Spectral-S0-实验记录.md",
        )
        for path in intended:
            completed = subprocess.run(
                ["git", "check-ignore", "-q", path], cwd=REPO_ROOT
            )
            self.assertNotEqual(completed.returncode, 0, path)
~~~

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_config_registry -v`

Expected: FAIL because manifests, configs, and registry do not exist.

- [ ] **Step 3: Implement deterministic manifests**

~~~python
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def stable_partition(names, dataset, seed=20260713):
    names = list(names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("source names contain an empty entry")
    if len(names) != len(set(names)):
        raise ValueError("source names contain a duplicate")
    ordered = sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"{seed}:{dataset}:{name}".encode("utf-8")
        ).hexdigest(),
    )
    fit_end = int(len(ordered) * 0.8)
    calibration_end = int(len(ordered) * 0.9)
    return {
        "fit": ordered[:fit_end],
        "calibration": ordered[fit_end:calibration_end],
        "gate_confirmation": ordered[calibration_end:],
    }
~~~

Use the same bootstrap preamble in every new `tools/*.py`. The split tool writes LF-normalized manifests and a JSON audit with source/output SHA-256 values plus deterministic hashes of the non-test sequence/frame/annotation trees actually used by each dataset loader. It refuses to overwrite a differing committed output.

Every new `tracking/*.py` starts with `import _init_paths` before importing `lib`; do not duplicate the tools preamble inside tracking scripts.

Run:

~~~bash
.venv/bin/python tools/freeze_target_spectral_splits.py \
  --seed 20260713 \
  --output-dir lib/train/data_specs
~~~

Expected: LasHeR and DepthTrack each produce three nonempty, disjoint manifests and an audit JSON.

- [ ] **Step 4: Expose named dataset splits and explicit LoRA checkpoint state**

Allow `spectral_fit`, `spectral_calibration`, and `spectral_gate_confirmation` in LasHeR and DepthTrack. Add corresponding explicit names to `names2datasets()`. Keep `build_dataloaders()` unchanged; add a separate `build_rollout_datasets()`.

Before serializing, `BaseTrainer.save_checkpoint()` inspects every `MergedLinear`, requires one common `merged` flag, and writes `lora_weight_state="merged"` or `"unmerged"` beside `net`. Use one loader everywhere:

~~~python
def load_seatrack_checkpoint(model, checkpoint_path, explicit_state=None, require_metadata=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    recorded = checkpoint.get("lora_weight_state")
    if recorded is None:
        if require_metadata or explicit_state is None:
            raise ValueError("LoRA checkpoint merge state is required")
        recorded = explicit_state
    if recorded not in {"merged", "unmerged"}:
        raise ValueError(f"invalid LoRA weight state: {recorded}")
    if explicit_state is not None and explicit_state != recorded:
        raise ValueError("LoRA checkpoint/index merge-state mismatch")
    model.load_state_dict(checkpoint["net"], strict=True)
    for module in model.modules():
        if isinstance(module, MergedLinear):
            module.merged = recorded == "merged"
    model.eval()  # merges exactly once only when the loaded weight was unmerged
    return checkpoint
~~~

Replace `SEATrack`'s current `merged=True -> train() -> eval()` heuristic with this helper. Workstream A tracker/fitter/runners pass `require_metadata=True` and the index state. Legacy released-checkpoint configs may pass an explicit `merged` state, but no Workstream A run may infer it. A synthetic parity test saves the same nonzero-LoRA layer once merged and once unmerged, reloads both, and requires identical eval output to the pre-save reference; it also proves the old heuristic fails the unmerged fixture.

- [ ] **Step 5: Create calibration registry with concrete engineering defaults and search domains**

Use:

~~~yaml
schema_version: 1
registry_id: spectral-s0-v1
status: calibration
design_sha256: 36eaf659a0b6550aefee1db9548a67caca875b6dd839857baacde099ee9049de
split_seed: 20260713

memory:
  allowed_ranks: [8, 16, 32]
  provisional_rank: 16
  trace_energy_threshold: 0.90
  beta: 0.95
  identity_anchor_weight: 0.50
  anchor_scope_weights: [0.50, 0.50]
  minimum_effective_mass: 16.0
  eigengap_relative: 0.01
  epsilon: 0.000001
  history_ema_beta: 0.90

routing:
  layers: [5, 9]
  modules: [attn, ffn]
  observe_scopes: [template, search]
  route_scopes: [search]
  alpha_budget: 0.25
  operator_norm_cap: 0.50
  dispatch_normalized_budget: 0.20
  combine_normalized_budget: 0.10
  temperature_floor: 0.0001
  active_strength: 1.0

weights:
  response_temperature: 1.0
  target_epsilon: 0.00000001
  background_box_scale_multiplier: 1.5
  background_top_fraction: 0.10

admission:
  coverage_candidates: [0.20, 0.30, 0.40, 0.50]
  minimum_write_coverage: 0.20
  localization_iou_positive: 0.50
  attention_floor_quantile: 0.10
  pair_alignment_min_lcb: 0.02
  pair_alignment_lcb_confidence: 0.975
  pair_alignment_scopes: [template, search]

statistics:
  bootstrap_seed: 20260713
  bootstrap_replicates: 10000
  bootstrap_design: crossed_seed_sequence_paired
  sequence_draw_shared_across_seed_slots: true
  seed_slots_resampled: true
  inference_scope: conditional_on_frozen_shared_u_and_checkpoint_set_0_1_2
  one_sided_confidence: 0.975
  adapter_metric_scale: unit_interval
  raw_delta_to_percentage_points: 100.0
  clean_noninferiority_margin_pp: -0.3

controls:
  shuffle_seed: 20260713
  random_orthogonal_seed: 20260714
  temporal_ring_length: 8
  coefficient_owner: full_four_spectrum
  coefficient_checkpoint_shared_from_full: true
  independent_control_fitting: false
  table_role: frozen_coefficient_mechanism_isolation
  required_strength_matched_loo: true

efficiency:
  profile_sequence_rule: sha256_min_clean_eligible
  warmup_frames: 10
  measured_frames: 50
  complete_episode_search_frames: 60
  one_method_per_child_process: true
  binding_fps_field: episode_fps
  binding_memory_fields: [absolute_peak_allocated_bytes, absolute_peak_reserved_bytes]
  minimum_active_fps_ratio: 0.80
  maximum_active_peak_memory_ratio: 1.25

benchmarks:
  lasher:
    evaluator: lasher_author_artifact
    evaluation_scope: heldout_non_test_gate
    primary_field: success_auc
    metric_scale: unit_interval
    success_thresholds: [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    aggregation: sequence_macro
  depthtrack:
    evaluator: depthtrack_paper_faithful_v1
    evaluation_scope: heldout_non_test_gate
    primary_field: f_score_sequence
    metric_scale: unit_interval
    threshold_scan: unique_confidences_plus_infinite_sentinels
    threshold_tie_break: higher_threshold

coefficient_fit:
  u_initial: [0.0, 0.0, 0.0, 0.0]
  fit_seed: 20260713
  attempt_manifest_seed: 20260713
  attempted_clips_per_stratum_per_superstep: 2
  feasibility_attempts_per_stratum: 100
  minimum_family_active_coverage: 0.20
  minimum_effective_route_coverage: 0.20
  minimum_effective_route_delta_l2: 0.000000000001
  tangent_gradient_sum_tolerance: 0.00000001
  minimum_tangent_rms_singular_value: 0.000001
  minimum_tangent_relative_singular_value: 0.01
  max_peak_device_fraction: 0.90
  required_successful_optimizer_steps: 1000
  maximum_attempted_supersteps: 2000
  checkpoint_successful_steps: [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  optimizer: Adam
  learning_rate: 0.01
  weight_decay: 0.0

recovery:
  corruption: x_modality_blackout
  severity_search: [0.50, 0.75, 1.00]
  burst_start_fraction: 0.40
  burst_length_frames: 20
  horizon_frames: 100
  failure_iou_below: 0.10
  failure_consecutive_frames: 5
  recovery_iou_at_least: 0.50
  recovery_consecutive_frames: 5
~~~

Also record both normative source paths, the base-design hash, the addendum path whose hash is computed at execution, the displayed coefficient-fit budget, exact control seeds/ring length, raw-metric/percentage-point scales, frozen efficiency protocol, one shared `u` across all six bases, and earliest-step tie breaking. Decision 4 permits only the one `full_four_spectrum` coefficient fit. Random/pooled/LOO/shuffle rows therefore share that frozen checkpoint to isolate mechanism; they are not independently tuned competitors and cannot support a “best tuned baseline” superiority claim. Registry validation rejects missing or renamed control, scale, or efficiency fields and requires both benchmark adapters to emit unit-interval raw metrics with the sole raw-to-pp multiplier exactly `100.0`. First freeze the attention floor at the registered calibration quantile.

Interpret each `coverage_candidates` value `c` as a proposed global scheduled-coverage fraction. Form one canonical float64 array of finite routing-disabled `q_memory` from every post-initialization calibration frame across all six `(base_seed,benchmark)` strata, including frames whose later `paired_valid` is false, ordered by `(benchmark,base_seed,sequence,frame_index)` only for reproducible serialization. Define `threshold(c) = numpy.quantile(q_values, 1.0-c, method="higher")`; use the observed threshold exactly, with no interpolation or rounding. Then materialize `scheduled_admit = (q_memory >= threshold(c)) & paired_valid`, so exact-threshold ties are admitted. The quantile population is frame-pooled by preregistration; the subsequent per-stratum constraint prevents a large stratum from hiding poor coverage elsewhere. Coverage denominator is every post-initialization frame in that manifest. Reject a candidate unless realized scheduled-admit coverage is at least `0.20` both overall and separately in each of the six strata. Among survivors choose the lowest false-admission rate, where false admission is evaluated on GT-valid scheduled frames and means evaluator IoU below `0.50`; ties choose higher realized overall coverage, then the lower numerical threshold, then the smaller registered `c` if thresholds are identical. This use of GT is calibration-only and freezes a GT-free runtime threshold. Record candidate `c`, quantile probability/method/population hash, threshold, all denominators/counts, and fail if any stratum has no GT-valid scheduled frame. No schedule or future gate-result hash appears in the calibration registry.

- **Configuration materialization:** create one complete YAML per atomic action:

- [ ] **Step 6a: Create `rgbt_spectral_s0.yaml`**

`experiments/seatrack/rgbt_spectral_s0.yaml`:

~~~yaml
DATA:
  MAX_SAMPLE_INTERVAL: 200
  MEAN: [0.485, 0.456, 0.406]
  STD: [0.229, 0.224, 0.225]
  SEARCH: {CENTER_JITTER: 3, FACTOR: 4.0, SCALE_JITTER: 0.25, SIZE: 256, NUMBER: 1}
  TEMPLATE: {CENTER_JITTER: 0, FACTOR: 2.0, SCALE_JITTER: 0, SIZE: 128, NUMBER: 1}
  TRAIN:
    DATASETS_NAME: [LasHeR_spectral_fit]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 60000
  VAL:
    DATASETS_NAME: [null]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 60000
MODEL:
  PRETRAIN_FILE: ""
  EXTRA_MERGER: false
  RETURN_INTER: false
  AMGLORA_RANK: 8
  HMOE_RANK: 4
  DETERMINISTIC_LORA_INIT: true
  AMG_ENABLED: true
  HMOE_ENABLED: true
  CHECKPOINT_LORA_WEIGHT_STATE: unmerged
  BILIFT: {ENABLED: false, DIAGNOSTICS: false}
  GRA: {ENABLED: false, DIAGNOSTICS: false}
  TARGET_SPECTRAL:
    ENABLED: true
    STAGE: s0
    LAYERS: [5, 9]
    MODULES: [attn, ffn]
    OBSERVE_SCOPES: [template, search]
    ROUTE_SCOPES: [search]
    RANK: 16
    RANK_CANDIDATES: [8, 16, 32]
    RANK_SOURCE: calibration_fallback
    CONTROL: full_four_spectrum
    STRENGTH: 1.0
    COEFFICIENT_CHECKPOINT: ""
    REGISTRY: experiments/seatrack/registries/spectral_s0_v1.calibration.yaml
  BACKBONE:
    TYPE: vit_base_patch16_224_ce
    STRIDE: 16
    CE_LOC: [3, 6, 9]
    CE_KEEP_RATIO: [1.0, 1.0, 1.0]
    CE_TEMPLATE_RANGE: CTR_POINT
  HEAD: {TYPE: CENTER, NUM_CHANNELS: 256}
TEST: {EPOCH: 60, SEARCH_FACTOR: 4.0, SEARCH_SIZE: 256, TEMPLATE_FACTOR: 2.0, TEMPLATE_SIZE: 128}
~~~

- [ ] **Step 6b: Create `rgbd_spectral_s0.yaml`**

`experiments/seatrack/rgbd_spectral_s0.yaml`:

~~~yaml
DATA:
  MAX_SAMPLE_INTERVAL: 200
  MEAN: [0.485, 0.456, 0.406]
  STD: [0.229, 0.224, 0.225]
  SEARCH: {CENTER_JITTER: 3, FACTOR: 4.0, SCALE_JITTER: 0.25, SIZE: 256, NUMBER: 1}
  TEMPLATE: {CENTER_JITTER: 0, FACTOR: 2.0, SCALE_JITTER: 0, SIZE: 128, NUMBER: 1}
  TRAIN:
    DATASETS_NAME: [DepthTrack_spectral_fit]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 60000
  VAL:
    DATASETS_NAME: [null]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 10000
MODEL:
  PRETRAIN_FILE: ""
  EXTRA_MERGER: false
  RETURN_INTER: false
  AMGLORA_RANK: 8
  HMOE_RANK: 4
  DETERMINISTIC_LORA_INIT: true
  AMG_ENABLED: true
  HMOE_ENABLED: true
  CHECKPOINT_LORA_WEIGHT_STATE: unmerged
  BILIFT: {ENABLED: false, DIAGNOSTICS: false}
  GRA: {ENABLED: false, DIAGNOSTICS: false}
  TARGET_SPECTRAL:
    ENABLED: true
    STAGE: s0
    LAYERS: [5, 9]
    MODULES: [attn, ffn]
    OBSERVE_SCOPES: [template, search]
    ROUTE_SCOPES: [search]
    RANK: 16
    RANK_CANDIDATES: [8, 16, 32]
    RANK_SOURCE: calibration_fallback
    CONTROL: full_four_spectrum
    STRENGTH: 1.0
    COEFFICIENT_CHECKPOINT: ""
    REGISTRY: experiments/seatrack/registries/spectral_s0_v1.calibration.yaml
  BACKBONE:
    TYPE: vit_base_patch16_224_ce
    STRIDE: 16
    CE_LOC: [3, 6, 9]
    CE_KEEP_RATIO: [1.0, 1.0, 1.0]
    CE_TEMPLATE_RANGE: CTR_POINT
  HEAD: {TYPE: CENTER, NUM_CHANNELS: 256}
TEST: {EPOCH: 60, SEARCH_FACTOR: 4.0, SEARCH_SIZE: 256, TEMPLATE_FACTOR: 2.0, TEMPLATE_SIZE: 128}
~~~

- [ ] **Step 6c: Create `rgbt_spectral_s0_short.yaml`**

`experiments/seatrack/rgbt_spectral_s0_short.yaml`:

~~~yaml
DATA:
  MAX_SAMPLE_INTERVAL: 200
  MEAN: [0.485, 0.456, 0.406]
  STD: [0.229, 0.224, 0.225]
  SEARCH: {CENTER_JITTER: 0, FACTOR: 4.0, SCALE_JITTER: 0, SIZE: 256, NUMBER: 1}
  TEMPLATE: {CENTER_JITTER: 0, FACTOR: 2.0, SCALE_JITTER: 0, SIZE: 128, NUMBER: 1}
  TRAIN:
    DATASETS_NAME: [LasHeR_smoke]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 1
  VAL:
    DATASETS_NAME: [null]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 1
MODEL:
  PRETRAIN_FILE: ""
  EXTRA_MERGER: false
  RETURN_INTER: false
  AMGLORA_RANK: 8
  HMOE_RANK: 4
  DETERMINISTIC_LORA_INIT: true
  AMG_ENABLED: true
  HMOE_ENABLED: true
  CHECKPOINT_LORA_WEIGHT_STATE: unmerged
  BILIFT: {ENABLED: false, DIAGNOSTICS: false}
  GRA: {ENABLED: false, DIAGNOSTICS: false}
  TARGET_SPECTRAL:
    ENABLED: true
    STAGE: s0
    LAYERS: [5, 9]
    MODULES: [attn, ffn]
    OBSERVE_SCOPES: [template, search]
    ROUTE_SCOPES: [search]
    RANK: 16
    RANK_CANDIDATES: [8, 16, 32]
    RANK_SOURCE: calibration_fallback
    CONTROL: full_four_spectrum
    STRENGTH: 1.0
    COEFFICIENT_CHECKPOINT: ""
    REGISTRY: experiments/seatrack/registries/spectral_s0_v1.calibration.yaml
  BACKBONE:
    TYPE: vit_base_patch16_224_ce
    STRIDE: 16
    CE_LOC: [3, 6, 9]
    CE_KEEP_RATIO: [1.0, 1.0, 1.0]
    CE_TEMPLATE_RANGE: CTR_POINT
  HEAD: {TYPE: CENTER, NUM_CHANNELS: 256}
TEST: {EPOCH: 60, SEARCH_FACTOR: 4.0, SEARCH_SIZE: 256, TEMPLATE_FACTOR: 2.0, TEMPLATE_SIZE: 128}
~~~

- [ ] **Step 6d: Create `rgbt_spectral_base.yaml`**

`experiments/seatrack/rgbt_spectral_base.yaml`:

~~~yaml
DATA:
  MAX_SAMPLE_INTERVAL: 200
  MEAN: [0.485, 0.456, 0.406]
  STD: [0.229, 0.224, 0.225]
  SEARCH: {CENTER_JITTER: 3, FACTOR: 4.0, SCALE_JITTER: 0.25, SIZE: 256, NUMBER: 1}
  TEMPLATE: {CENTER_JITTER: 0, FACTOR: 2.0, SCALE_JITTER: 0, SIZE: 128, NUMBER: 1}
  TRAIN:
    DATASETS_NAME: [LasHeR_spectral_fit]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 60000
  VAL:
    DATASETS_NAME: [null]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 60000
MODEL:
  PRETRAIN_FILE: ./pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar
  EXTRA_MERGER: false
  RETURN_INTER: false
  AMGLORA_RANK: 8
  HMOE_RANK: 4
  DETERMINISTIC_LORA_INIT: true
  AMG_ENABLED: true
  HMOE_ENABLED: true
  CHECKPOINT_LORA_WEIGHT_STATE: merged
  BILIFT: {ENABLED: false, DIAGNOSTICS: false}
  GRA: {ENABLED: false, DIAGNOSTICS: false}
  TARGET_SPECTRAL: {ENABLED: false, STAGE: disabled}
  BACKBONE:
    TYPE: vit_base_patch16_224_ce
    STRIDE: 16
    CE_LOC: [3, 6, 9]
    CE_KEEP_RATIO: [1, 1, 1]
    CE_TEMPLATE_RANGE: CTR_POINT
  HEAD: {TYPE: CENTER, NUM_CHANNELS: 256}
TRAIN:
  BACKBONE_MULTIPLIER: 0.1
  DROP_PATH_RATE: 0.1
  CE_START_EPOCH: 4
  CE_WARM_EPOCH: 16
  BATCH_SIZE: 32
  EPOCH: 60
  GIOU_WEIGHT: 2.0
  L1_WEIGHT: 5.0
  GRAD_CLIP_NORM: 0.1
  LR: 0.0004
  LR_DROP_EPOCH: 48
  NUM_WORKER: 10
  OPTIMIZER: ADAMW
  PRINT_INTERVAL: 50
  SCHEDULER: {TYPE: step, DECAY_RATE: 0.1}
  VAL_EPOCH_INTERVAL: 5
  WEIGHT_DECAY: 0.0001
  PEFT: true
  AMP: false
  FIX_BN: true
  SAVE_EPOCH_INTERVAL: 60
  SAVE_LAST_N_EPOCH: 1
TEST: {EPOCH: 60, SEARCH_FACTOR: 4.0, SEARCH_SIZE: 256, TEMPLATE_FACTOR: 2.0, TEMPLATE_SIZE: 128}
~~~

- [ ] **Step 6e: Create `rgbd_spectral_base.yaml`**

`experiments/seatrack/rgbd_spectral_base.yaml`:

~~~yaml
DATA:
  MAX_SAMPLE_INTERVAL: 200
  MEAN: [0.485, 0.456, 0.406]
  STD: [0.229, 0.224, 0.225]
  SEARCH: {CENTER_JITTER: 3, FACTOR: 4.0, SCALE_JITTER: 0.25, SIZE: 256, NUMBER: 1}
  TEMPLATE: {CENTER_JITTER: 0, FACTOR: 2.0, SCALE_JITTER: 0, SIZE: 128, NUMBER: 1}
  TRAIN:
    DATASETS_NAME: [DepthTrack_spectral_fit]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 60000
  VAL:
    DATASETS_NAME: [null]
    DATASETS_RATIO: [1]
    SAMPLE_PER_EPOCH: 10000
MODEL:
  PRETRAIN_FILE: ./pretrained/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar
  EXTRA_MERGER: false
  RETURN_INTER: false
  AMGLORA_RANK: 8
  HMOE_RANK: 4
  DETERMINISTIC_LORA_INIT: true
  AMG_ENABLED: true
  HMOE_ENABLED: true
  CHECKPOINT_LORA_WEIGHT_STATE: merged
  BILIFT: {ENABLED: false, DIAGNOSTICS: false}
  GRA: {ENABLED: false, DIAGNOSTICS: false}
  TARGET_SPECTRAL: {ENABLED: false, STAGE: disabled}
  BACKBONE:
    TYPE: vit_base_patch16_224_ce
    STRIDE: 16
    CE_LOC: [3, 6, 9]
    CE_KEEP_RATIO: [1, 1, 1]
    CE_TEMPLATE_RANGE: CTR_POINT
  HEAD: {TYPE: CENTER, NUM_CHANNELS: 256}
TRAIN:
  BACKBONE_MULTIPLIER: 0.1
  DROP_PATH_RATE: 0.1
  CE_START_EPOCH: 4
  CE_WARM_EPOCH: 16
  BATCH_SIZE: 32
  EPOCH: 25
  GIOU_WEIGHT: 2.0
  L1_WEIGHT: 5.0
  GRAD_CLIP_NORM: 0.1
  LR: 0.0004
  LR_DROP_EPOCH: 48
  NUM_WORKER: 10
  OPTIMIZER: ADAMW
  PRINT_INTERVAL: 50
  SCHEDULER: {TYPE: step, DECAY_RATE: 0.1}
  VAL_EPOCH_INTERVAL: 5
  WEIGHT_DECAY: 0.0001
  PEFT: true
  AMP: false
  FIX_BN: true
  SAVE_EPOCH_INTERVAL: 25
  SAVE_LAST_N_EPOCH: 1
TEST: {EPOCH: 60, SEARCH_FACTOR: 4.0, SEARCH_SIZE: 256, TEMPLATE_FACTOR: 2.0, TEMPLATE_SIZE: 128}
~~~

- [ ] **Step 6f: Verify the gate overlay and checkpoint-state interpretation**

Gate runners deepcopy the committed S0 config, overlay only `RANK_SOURCE: registry` and the frozen selected rank, then hash the effective canonical config. `CHECKPOINT_LORA_WEIGHT_STATE: merged` in the base files describes their released pretrain input; the saved fit checkpoints record their observed final unmerged state beside `net` and S0 loads those indexed checkpoints as `unmerged`.

Run: `.venv/bin/python -m unittest tests.test_spectral_config_registry.SpectralConfigRegistryTests.test_s0_configs_are_exact_calibration_only_overlays tests.test_spectral_config_registry.SpectralConfigRegistryTests.test_base_configs_preserve_fit_only_training_protocol -v`

Expected: both tests PASS; all five YAML files parse and expose exactly the registered rank, routing, split, LoRA-state, and disabled-method fields.

- [ ] **Step 7: Add precise Git visibility rules**

Append these rules at the end of `.gitignore`, after its final `experiments` rule:

~~~gitignore
!experiments/
experiments/*
!experiments/seatrack/
experiments/seatrack/*
!experiments/seatrack/rgbt_spectral_s0.yaml
!experiments/seatrack/rgbd_spectral_s0.yaml
!experiments/seatrack/rgbt_spectral_s0_short.yaml
!experiments/seatrack/rgbt_spectral_base.yaml
!experiments/seatrack/rgbd_spectral_base.yaml
!experiments/seatrack/registries/
experiments/seatrack/registries/*
!experiments/seatrack/registries/spectral_s0_v1.calibration.yaml
!experiments/seatrack/registries/spectral_s0_v1.frozen.yaml
!experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json
!tools/
tools/*
!tools/freeze_target_spectral_splits.py
!tools/index_target_spectral_checkpoints.py
!tools/freeze_spectral_s0_registry.py
!tools/profile_spectral_s0.py
!tools/validate_benchmark_evaluators.py
!knowledge_base/
knowledge_base/*
!knowledge_base/Target-Spectral-S0-实验记录.md
!knowledge_base/Target-Spectral-S0-base-checkpoints.json
!knowledge_base/Target-Spectral-S0-gate.json
~~~

- [ ] **Step 8: Verify registry inputs and Git visibility**

Run:

~~~bash
.venv/bin/python -m unittest tests.test_spectral_config_registry -v
for path in \
  tools/freeze_target_spectral_splits.py \
  experiments/seatrack/rgbt_spectral_s0.yaml \
  experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  knowledge_base/Target-Spectral-S0-实验记录.md; do
  if git check-ignore -q "$path"; then echo "still ignored: $path"; exit 1; fi
done
repo="$PWD"
for script in tools/freeze_target_spectral_splits.py tools/index_target_spectral_checkpoints.py; do
  .venv/bin/python "$script" --help >/dev/null
  (cd /tmp && "$repo/.venv/bin/python" "$repo/$script" --help >/dev/null)
done
~~~

Expected: validation PASS, every visibility check is silent, all four root/`/tmp` CLI invocations exit zero, and gate purpose rejects the calibration registry.

- [ ] **Step 9: Commit registry inputs exactly**

~~~bash
git add .gitignore tools/freeze_target_spectral_splits.py \
  tools/index_target_spectral_checkpoints.py \
  lib/train/data_specs/lasher_spectral_fit.txt \
  lib/train/data_specs/lasher_spectral_calibration.txt \
  lib/train/data_specs/lasher_spectral_gate_confirmation.txt \
  lib/train/data_specs/depthtrack_spectral_fit.txt \
  lib/train/data_specs/depthtrack_spectral_calibration.txt \
  lib/train/data_specs/depthtrack_spectral_gate_confirmation.txt \
  lib/train/data_specs/target_spectral_split_audit.json \
  lib/train/dataset/lasher.py lib/train/dataset/depthtrack.py \
  lib/train/base_functions.py lib/train/trainers/base_trainer.py \
  lib/models/seatrack/checkpoint.py lib/test/tracker/seatrack.py \
  experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  experiments/seatrack/rgbt_spectral_s0.yaml \
  experiments/seatrack/rgbd_spectral_s0.yaml \
  experiments/seatrack/rgbt_spectral_s0_short.yaml \
  experiments/seatrack/rgbt_spectral_base.yaml \
  experiments/seatrack/rgbd_spectral_base.yaml \
  knowledge_base/Target-Spectral-S0-实验记录.md \
  tests/test_spectral_config_registry.py
git commit -m "test: lock workstream a manifests and registry"
~~~

- [ ] **Step 10: Train the three independently initialized fit-only base seeds**

~~~bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/train.py \
    --script seatrack --config rgbt_spectral_base --mode single \
    --seed "$seed" --save_dir "output/spectral_s0/base/seed_${seed}/rgbt"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/train.py \
    --script seatrack --config rgbd_spectral_base --mode single \
    --seed "$seed" --save_dir "output/spectral_s0/base/seed_${seed}/rgbd"
done
~~~

Expected: six finite final checkpoints, one RGB-T and one RGB-D checkpoint for each seed. Record their SHA-256 values, exact commands, hardware, epochs, and fit-manifest hashes. Do not call these six runs “online seeds.”

- [ ] **Step 11: Index and audit base parity before coefficient fitting**

Run:

~~~bash
.venv/bin/python tools/index_target_spectral_checkpoints.py \
  --root output/spectral_s0/base \
  --rgbt-relative checkpoints/rgbt_spectral_base/SEATrack_ep0060.pth.tar \
  --rgbd-relative checkpoints/rgbd_spectral_base/SEATrack_ep0025.pth.tar \
  --seeds 0,1,2 --validate-strict-load \
  --output output/spectral_s0/base/base_checkpoints.json \
  --provenance-copy knowledge_base/Target-Spectral-S0-base-checkpoints.json
~~~

The index contains exactly six absolute checkpoint paths, SHA-256 values, `lora_weight_state=unmerged`, modality, base seed, config hash, fit-manifest hash, and creation command. It rejects a checkpoint whose embedded merge state differs. For each seed, strict-load both checkpoints through `load_seatrack_checkpoint(..., require_metadata=True)`, run the nonzero-LoRA parity fixture, assert expected HMoE modules are present, target-spectral state is absent, and no calibration/gate-confirmation manifest is opened. Append the audit to `knowledge_base/Target-Spectral-S0-实验记录.md`.

- [ ] **Step 12: Commit the base-checkpoint provenance audit**

~~~bash
git add knowledge_base/Target-Spectral-S0-实验记录.md \
  knowledge_base/Target-Spectral-S0-base-checkpoints.json
git commit -m "chore: record spectral base checkpoint provenance"
~~~

### Task 8: Strict Chronological, Prediction-Centred Rollout

**Files:**

- Create: `lib/train/data/rollout_sampler.py`
- Create: `lib/train/data/rollout_processing.py`
- Modify: `lib/train/data/__init__.py`
- Modify: `lib/train/base_functions.py`
- Create: `tests/test_spectral_rollout.py`

**Interfaces:**

- `ChronologicalRolloutSampler.clip_frame_ids(num_frames, target_frame) -> (0,t-2,t-1,t,t+1)`.
- `RolloutClip` keeps those five optimization frames and a separate ordered image-only `warmup_prefix` for frames `1..t-3`; prefix annotations never enter tracker-facing objects.
- `PredictionCenteredRolloutProcessor.begin_clip(clip)`.
- `next_search(role)` crops around the previous committed prediction.
- `RolloutSearchStep` carries the exact `CropTransform` used for prior/evidence/outer-label mapping.
- `commit_prediction(pred_box_crop_cxcywh, resize_factor, image_hw)` updates detached crop state with the unchanged legacy tracker mapping.
- Batch size is one and network-side processing uses zero DataLoader workers.

- [ ] **Step 1: Write failing ordering and GT-sentinel tests**

Cover strict uniqueness/order, invisible retention, no visible filtering, no replacement, no gap expansion, all optimization/prefix frames from one sequence, reset at each clip, prefix IDs exactly `range(1,t-2)`, no prefix GT/visibility access, a second `next_search()` failing until the pending step is committed, and a deliberately offset predicted box controlling the next crop centre and scale. Frame 0 is the initialization anchor; prefix plus the four search frames form the uninterrupted chronological stream `1..t+1`, while only `[0,t-2,t-1,t,t+1]` is reported to the outer objective as the optimization clip.

~~~python
def test_clip_ids_are_exact_strict_and_distinct(self):
    ids = ChronologicalRolloutSampler.clip_frame_ids(20, target_frame=8)
    self.assertEqual(ids, (0, 6, 7, 8, 9))
    self.assertEqual(len(set(ids)), 5)

def test_tp1_crop_uses_t_prediction_not_tp1_ground_truth(self):
    processor = make_processor(search_size=100)
    processor.begin_clip(clip_with_init_state_and_far_tp1_gt([100., 80., 40., 30.]))
    processor.next_search("candidate_t")
    processor.commit_prediction(torch.tensor([30., 40., 20., 10.]), 1.0, (240, 320))
    step = processor.next_search("outer_tp1")
    # Previous centre=(120,95), crop origin=(70,45), so mapped image box is [90,80,20,10].
    torch.testing.assert_close(step.extraction_state_xywh, torch.tensor([90., 80., 20., 10.]))
~~~

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_rollout -v`

Expected: FAIL on missing rollout modules.

- [ ] **Step 3: Implement the dedicated sampler**

Implement the sampler without calling `TrackingSampler._sample_visible_ids`:

~~~python
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

def hash_frame_ids(frame_ids):
    payload = json.dumps(list(frame_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

@dataclass(frozen=True)
class RolloutClip:
    dataset_name: str
    sequence_name: str
    optimization_frame_ids: tuple[int, int, int, int, int]
    optimization_images: tuple[np.ndarray, ...]
    optimization_annotations: Mapping[str, torch.Tensor]
    warmup_prefix_frame_ids: tuple[int, ...]
    warmup_prefix: tuple[np.ndarray, ...]
    optimization_frame_ids_sha256: str
    warmup_prefix_frame_ids_sha256: str

class ChronologicalRolloutSampler:
    def __init__(self, generator):
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator")
        self.generator = generator

    @staticmethod
    def clip_frame_ids(num_frames, target_frame):
        t = int(target_frame)
        if num_frames < 5 or not 3 <= t <= num_frames - 2:
            raise ValueError("target_frame must lie in [3,num_frames-2]")
        return (0, t - 2, t - 1, t, t + 1)

    def sample(self, dataset, sequence_id, target_frame=None):
        info = dataset.get_sequence_info(sequence_id)
        num_frames = int(info["bbox"].shape[0])
        t = int(target_frame) if target_frame is not None else int(torch.randint(
            low=3,
            high=num_frames - 1,
            size=(),
            generator=self.generator,
        ))
        optimization_ids = self.clip_frame_ids(num_frames, t)
        warmup_ids = tuple(range(1, t - 2))
        optimization_frames, optimization_anno, _ = dataset.get_frames(
            sequence_id, list(optimization_ids), anno=info
        )
        # Deliberately request prefix images without passing its annotations onward.
        warmup_frames, warmup_anno, _ = dataset.get_frames(
            sequence_id, list(warmup_ids), anno={}
        )
        if warmup_anno:
            raise RuntimeError("warm-up prefix loader exposed annotations")
        if len(optimization_frames) != 5 or len(warmup_frames) != len(warmup_ids):
            raise RuntimeError("dataset returned a mismatched rollout frame count")
        annotations = {
            name: torch.stack([torch.as_tensor(value) for value in values])
            for name, values in optimization_anno.items()
            if name in {"bbox", "valid", "visible"}
        }
        if set(annotations) != {"bbox", "valid", "visible"}:
            raise RuntimeError("optimization annotations are incomplete")
        return RolloutClip(
            dataset_name=dataset.get_name(),
            sequence_name=dataset.sequence_list[sequence_id],
            optimization_frame_ids=optimization_ids,
            optimization_images=tuple(optimization_frames),
            optimization_annotations=MappingProxyType(annotations),
            warmup_prefix_frame_ids=warmup_ids,
            warmup_prefix=tuple(warmup_frames),
            optimization_frame_ids_sha256=hash_frame_ids(optimization_ids),
            warmup_prefix_frame_ids_sha256=hash_frame_ids(warmup_ids),
        )
~~~

Reject a loader that returns a different frame count. Hash and log `warmup_prefix_frame_ids` separately from `optimization_frame_ids`; never resample based on visibility, validity, family activation, or outer-label usability.

- [ ] **Step 4: Implement inference-faithful processing**

~~~python
def next_search(self, role):
    assert self._pending_step is None, "commit the previous search before advancing"
    frame = self._frames[self.ROLE_INDEX[role]]
    extraction = self._committed_state.detach().clone()
    crop, resize_factor, padding = sample_target(
        frame, extraction.tolist(), self.search_factor, self.search_size
    )
    crop_transform = CropTransform.from_target(
        extraction, self.search_factor, self.search_size
    )
    if resize_factor != crop_transform.resize_factor:
        raise AssertionError("rollout crop transform mismatch")
    step = RolloutSearchStep(
        role=role,
        frame_id=self._frame_ids[self.ROLE_INDEX[role]],
        search_tensor=self.transform(image=crop),
        extraction_state_xywh=extraction,
        resize_factor=float(resize_factor),
        crop_transform=crop_transform,
        image_hw=frame.shape[:2],
        padding_mask=padding,
    )
    self._pending_step = step
    return step
~~~

`padding_mask` remains `True=padding` until Task 3's explicit downsample-and-invert operation. A second `next_search()` before `commit_prediction()` fails rather than overwriting the pending extraction geometry. The causal prior and `outer_target()` use `self._pending_step.crop_transform`; the latter may read `gt_tp1` only after the outer crop and forward exist. Implement prediction commit with the unchanged legacy continuous-centre mapping using the extraction state saved by the immediately preceding `next_search`, not any annotation:

~~~python
def commit_prediction(self, pred_box_crop_cxcywh, resize_factor, image_hw):
    assert self._pending_step is not None
    previous = self._pending_step.extraction_state_xywh
    previous_cx = previous[0] + 0.5 * previous[2]
    previous_cy = previous[1] + 0.5 * previous[3]
    crop_side_image = self.search_size / float(resize_factor)
    crop_origin = pred_box_crop_cxcywh.new_tensor([
        previous_cx - 0.5 * crop_side_image,
        previous_cy - 0.5 * crop_side_image,
    ])
    cxcywh_image = pred_box_crop_cxcywh / float(resize_factor)
    cxcywh_image[:2] += crop_origin
    xywh_image = box_cxcywh_to_xywh(cxcywh_image)
    height, width = image_hw
    self._committed_state = torch.tensor(
        clip_box(xywh_image.tolist(), height, width, margin=10),
        dtype=xywh_image.dtype,
    )
    self._pending_step = None
~~~

Add a half-integer fixture where the legacy continuous origin differs from `CropTransform.origin_xy`: committed tracker state must still match legacy `map_box_back`, while prior rasterization and outer-label crop coordinates must match the exact rounded transform. This separates output compatibility from evidence/label geometry instead of silently using one transform for both contracts.

After reset and initialization, process every `warmup_prefix` image with the same prediction-centred `next_search()`/windowed-decode/commit path under `torch.no_grad()`, followed by `t-2,t-1,t`; do not jump tracker state across an omitted interval. Only after the `outer_tp1` crop object has been constructed may `outer_target()` read `gt_tp1` and transform it for the outer loss. Prefix/history GT, visibility, and validity are never read by the processor.

- [ ] **Step 5: Verify rollout behavior**

~~~bash
.venv/bin/python -m unittest tests.test_spectral_rollout -v
~~~

Expected: all rollout tests PASS; deliberately offset predictions determine later crop geometry.

- [ ] **Step 6: Commit chronological rollout**

~~~bash
git add lib/train/data/rollout_sampler.py lib/train/data/rollout_processing.py \
  lib/train/data/__init__.py lib/train/base_functions.py tests/test_spectral_rollout.py
git commit -m "feat: add prediction centred spectral rollout"
~~~

### Task 9: Offline Shared-Coefficient Fitting

**Files:**

- Create: `lib/train/spectral/__init__.py`
- Create: `lib/train/spectral/coefficient_fit.py`
- Create: `tracking/fit_spectral_coefficients.py`
- Create: `tests/test_spectral_coefficients.py`

**Interfaces:**

- `SpectralCoefficientFitter` owns the only trainable `spectral_coefficients.u`.
- Same-sequence warm-up frames `1..t-3` and history frames `t-2,t-1,t` predict and commit detached state in one uninterrupted causal stream.
- Only `t+1` outer tracking loss backpropagates.
- One coefficient vector is jointly fitted across all six frozen base checkpoints, both RGB-T/RGB-D fit streams, blocks 5/9, attn/FFN, and RGB/X. Every S0 seed loads that same frozen vector.
- `FitAttemptManifest` fixes all clip attempts through the 2,000-attempted-superstep ceiling before any label, activity, confidence, loss, or coefficient-dependent value is inspected.
- `FitFeasibilityReport` evaluates the fixed first 100 attempts per stratum before optimization and again at the calibration-selected checkpoint; failure stops instead of resampling or selecting the next checkpoint.
- Checkpoints are indexed by `successful_step`; `attempted_superstep` is separately monotone and may advance without an optimizer update.
- Task 9 implements fitting and a provisional smoke only. Task 12 selects the optimizer step on calibration after confidence threshold and rank are locked.
- Fit clips use `FrozenRuleAdmissionSource` with the locked calibration threshold/floor; they never require a pre-recorded fit schedule or GT-derived admission.

- **Coefficient-fitter test implementation:** add these fixtures separately:

- [ ] **Step 1a: Add the six-frozen-base equal-gradient fixture**
- [ ] **Step 1b: Add the causal prefix/outer-target ordering fixture**
- [ ] **Step 1c: Add four-coordinate sensitivity and alpha-budget fixtures**
- [ ] **Step 1d: Add fixed-attempt, partial-activity, key/family feasibility, and successful-step fixtures**

Assert all six models stay in evaluation mode, no model gradients, exactly one shared `u` changes, finite gradient, exact alpha budget, candidate memory visible at `t+1`, RGB-T/RGB-D and three base seeds contribute equally, no prefix/history GT read before outer-target construction, checkpoint provenance completeness, isolated dual-config loading, and no gate manifest access. A partially active synthetic causal prefix must still construct the `t+1` outer loss. Perturbing each of the four independent leaf-alpha coordinates changes the outer output/loss; feasibility never treats softmax-`u` coordinates as four independent effects because `u` has a global-shift null direction.

~~~python
from collections import OrderedDict
import hashlib
import unittest

import torch
from torch import nn

from lib.models.target_spectral.routing import SharedRoutingCoefficients
from lib.train.spectral.coefficient_fit import SpectralCoefficientFitter

BASE_KEYS = tuple(
    f"seed{seed}:{modality}"
    for seed in range(3) for modality in ("rgbd", "rgbt")
)

def _parameter_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

class _FrozenModel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(value)]))

class _GradientFixtureFitter(SpectralCoefficientFitter):
    def __init__(self):
        self.components = OrderedDict()
        self.calls = {key: 0 for key in BASE_KEYS}
        self.features = {}
        for index, key in enumerate(BASE_KEYS):
            model = _FrozenModel(index + 1)
            model.requires_grad_(False)
            model.eval()
            self.components[key] = (model, object(), object())
            self.features[key] = torch.tensor(
                [1.0 + index, 2.0 + index, 4.0 + index, 8.0 + index]
            )
        self.coefficients = SharedRoutingCoefficients(
            alpha_budget=0.25, initial_u=torch.zeros(4)
        )
        self.optimizer = torch.optim.Adam(
            [self.coefficients.u], lr=0.01, weight_decay=0.0
        )
        self.attempted_supersteps = 0
        self.successful_steps = 0
        self.maximum_attempted_supersteps = 2

    def validate_manifested_superstep(
        self, clip_attempts_by_base, attempted_superstep,
        expected_base_keys, expected_attempts_per_base,
    ):
        self.assert_manifest_fixture = (
            attempted_superstep == 0
            and set(clip_attempts_by_base) == set(expected_base_keys)
            and all(tuple(value) == (0, 1) for value in clip_attempts_by_base.values())
            and expected_attempts_per_base == 2
        )
        if not self.assert_manifest_fixture:
            raise ValueError("synthetic manifested superstep mismatch")

    def clip_loss(self, base_key, clip):
        self.calls[base_key] += 1
        offset = torch.tensor([0.03, -0.02, 0.01, -0.04]) * float(clip)
        return torch.dot(
            self.coefficients.alpha, self.features[base_key] + offset
        )

    def report_skipped_superstep(self, base_key, reason):
        raise AssertionError((base_key, reason))

class _Snapshot:
    def __init__(self):
        self.active = set()
        self.candidate_committed = False

class _CausalController:
    def __init__(self):
        self.state = _Snapshot()

    def begin_episode(self, reset_global=True):
        if reset_global is not True:
            raise AssertionError("clip reset must be global")
        self.state = _Snapshot()

    def snapshot(self):
        return self.state

class _CausalProcessor:
    def __init__(self, events):
        self.events = events

    def begin_clip(self, clip):
        self.events.append("begin")

    def warmup_prefix_steps(self):
        return ("identity", "dynamic")

    def outer_target(self):
        if self.events[-1] != "outer_forward":
            raise AssertionError("GT was read before outer crop/forward")
        self.events.append("outer_target")
        return {
            "valid": torch.tensor([True]),
            "search_anno": torch.zeros(1, 4),
        }

class _CausalFixtureFitter(SpectralCoefficientFitter):
    def __init__(self):
        self.events = []
        controller = _CausalController()
        processor = _CausalProcessor(self.events)
        self.components = {
            "seed0:rgbt": (_FrozenModel(1), controller, processor)
        }
        self.coefficients = SharedRoutingCoefficients(
            alpha_budget=0.25, initial_u=torch.zeros(4)
        )

    def _initialize_anchor(self, base_key):
        self.events.append("anchor")

    def _predict_and_commit_step(self, base_key, family):
        self.components[base_key][1].state.active.add(family)
        self.events.append(f"prefix:{family}")

    def _predict_and_commit(self, base_key, role):
        self.events.append(role)
        if role == "candidate_t":
            self.components[base_key][1].state.candidate_committed = True

    def _forward_outer_with_coefficient_grad(self, base_key, role):
        state = self.components[base_key][1].state
        if role != "outer_tp1" or not state.candidate_committed:
            raise AssertionError("t candidate memory is not visible at t+1")
        self.events.append("outer_forward")
        return {"feature": torch.tensor([1.0, 2.0, 4.0, 8.0])}

    def outer_loss(self, outer, target):
        return torch.dot(self.coefficients.alpha, outer["feature"])

class SpectralCoefficientContractTests(unittest.TestCase):
    def test_superstep_is_exact_equal_six_stratum_gradient(self):
        fitter = _GradientFixtureFitter()
        before_u = fitter.coefficients.u.detach().clone()
        model_hashes = {
            key: _parameter_sha256(model)
            for key, (model, _, _) in fitter.components.items()
        }
        manual_u = before_u.clone().requires_grad_(True)
        alpha = 0.25 * torch.softmax(manual_u, dim=0)
        stratum_means = []
        for key in BASE_KEYS:
            losses = [
                torch.dot(
                    alpha,
                    fitter.features[key]
                    + torch.tensor([0.03, -0.02, 0.01, -0.04]) * clip,
                )
                for clip in (0, 1)
            ]
            stratum_means.append(torch.stack(losses).mean())
        expected_gradient, = torch.autograd.grad(
            torch.stack(stratum_means).mean(), manual_u
        )

        result = fitter.fit_superstep(
            {key: (0, 1) for key in BASE_KEYS}, attempted_superstep=0
        )
        self.assertEqual(result.successful_step, 1)
        self.assertTrue(fitter.assert_manifest_fixture)
        self.assertTrue(torch.isfinite(result.loss))
        torch.testing.assert_close(
            fitter.coefficients.u.grad, expected_gradient, atol=1e-7, rtol=1e-6
        )
        self.assertFalse(torch.equal(before_u, fitter.coefficients.u.detach()))
        torch.testing.assert_close(
            fitter.coefficients.alpha.sum(), torch.tensor(0.25)
        )
        self.assertEqual(set(fitter.calls.values()), {2})
        for key, (model, _, _) in fitter.components.items():
            self.assertFalse(model.training)
            self.assertTrue(all(
                not parameter.requires_grad and parameter.grad is None
                for parameter in model.parameters()
            ))
            self.assertEqual(_parameter_sha256(model), model_hashes[key])

    def test_partial_family_activity_still_contributes_before_outer_gt(self):
        fitter = _CausalFixtureFitter()
        loss = fitter.clip_loss("seed0:rgbt", object())
        self.assertEqual(
            fitter.components["seed0:rgbt"][1].snapshot().active,
            {"identity", "dynamic"},
        )
        gradient, = torch.autograd.grad(loss, fitter.coefficients.u)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient != 0).all())
        self.assertEqual(fitter.events[-2:], ["outer_forward", "outer_target"])

    def test_each_frozen_alpha_coordinate_changes_outer_loss(self):
        alpha = torch.tensor([0.04, 0.05, 0.07, 0.09])
        feature = torch.tensor([1.0, 2.0, 4.0, 8.0])
        baseline = torch.dot(alpha, feature)
        for index in range(4):
            perturbed = alpha.clone()
            perturbed[index] += 1e-3
            self.assertNotEqual(torch.dot(perturbed, feature).item(), baseline.item())
~~~

Add the fixed-attempt and real-data feasibility core:

~~~python
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch

FIT_KEYS = ((5, "attn"), (5, "ffn"), (9, "attn"), (9, "ffn"))
FIT_FAMILIES = ("identity", "dynamic", "private", "background")
FIT_BASE_KEYS = tuple(
    f"seed{seed}:{modality}"
    for seed in range(3) for modality in ("rgbd", "rgbt")
)

@dataclass(frozen=True)
class FitAttempt:
    attempted_superstep: int
    base_key: str
    ordinal_within_stratum: int
    sequence_id: str
    target_frame: int

def build_fit_attempt_manifest(
    candidates_by_base, seed, attempts_per_stratum, maximum_supersteps,
):
    if (
        set(candidates_by_base) != set(FIT_BASE_KEYS)
        or int(attempts_per_stratum) < 1
        or int(maximum_supersteps) < 1
    ):
        raise ValueError("fit manifest requires six bases and positive budgets")
    attempts = []
    needed = int(attempts_per_stratum) * int(maximum_supersteps)
    for base_key in sorted(candidates_by_base):
        candidates = tuple(sorted(
            (str(sequence), int(frame))
            for sequence, frame in candidates_by_base[base_key]
        ))
        if not candidates:
            raise ValueError(f"empty fit candidate set: {base_key}")
        generator = torch.Generator(device="cpu")
        digest = hashlib.sha256(
            f"{int(seed)}:{base_key}:fit-attempt-manifest".encode("utf-8")
        ).digest()
        generator.manual_seed(int.from_bytes(digest[:8], "big"))
        ordered = []
        while len(ordered) < needed:
            permutation = torch.randperm(len(candidates), generator=generator)
            ordered.extend(candidates[index] for index in permutation.tolist())
        for ordinal, (sequence_id, target_frame) in enumerate(ordered[:needed]):
            attempts.append(FitAttempt(
                attempted_superstep=ordinal // int(attempts_per_stratum),
                base_key=base_key,
                ordinal_within_stratum=ordinal,
                sequence_id=sequence_id,
                target_frame=target_frame,
            ))
    payload = [asdict(attempt) for attempt in sorted(
        attempts,
        key=lambda item: (
            item.attempted_superstep, item.base_key,
            item.ordinal_within_stratum,
        ),
    )]
    artifact = {
        "schema_version": 1,
        "manifest_seed": int(seed),
        "attempts_per_stratum_per_superstep": int(attempts_per_stratum),
        "maximum_attempted_supersteps": int(maximum_supersteps),
        "attempt_count": len(payload),
        "attempts": payload,
    }
    encoded = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return tuple(attempts), artifact, encoded, hashlib.sha256(encoded).hexdigest()

def audit_fit_feasibility(
    records, expected_attempt_ids_by_base, frozen_alpha,
    minimum_coverage=0.20, minimum_route_delta_l2=1e-12,
    gradient_sum_tolerance=1e-8,
    minimum_tangent_rms_singular_value=1e-6,
    minimum_tangent_relative_singular_value=0.01,
):
    alpha = np.asarray(frozen_alpha, dtype=np.float64)
    if (
        set(expected_attempt_ids_by_base) != set(FIT_BASE_KEYS)
        or not 0.0 <= float(minimum_coverage) <= 1.0
        or alpha.shape != (4,)
        or not np.isfinite(alpha).all()
        or (alpha <= 0.0).any()
        or not all(np.isfinite(value) and value > 0.0 for value in (
            minimum_route_delta_l2, gradient_sum_tolerance,
            minimum_tangent_rms_singular_value,
            minimum_tangent_relative_singular_value,
        ))
        or minimum_tangent_relative_singular_value > 1.0
    ):
        raise ValueError("invalid feasibility registry or six-base support")
    alpha_budget = float(alpha.sum())
    softmax_jacobian = np.diag(alpha) - np.outer(alpha, alpha) / alpha_budget
    helmert = np.asarray([
        [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
        [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
        [0.0, -2 / np.sqrt(6), 1 / np.sqrt(12)],
        [0.0, 0.0, -3 / np.sqrt(12)],
    ], dtype=np.float64)
    np.testing.assert_allclose(helmert.T @ helmert, np.eye(3), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(helmert.sum(axis=0), 0.0, atol=1e-12, rtol=0.0)
    records = tuple(records)
    indexed = {}
    for record in records:
        key = (record["base_key"], record["attempt_id"])
        if key in indexed:
            raise ValueError(f"duplicate feasibility attempt: {key}")
        indexed[key] = record
    expected = {
        (base_key, attempt_id)
        for base_key, attempt_ids in expected_attempt_ids_by_base.items()
        for attempt_id in attempt_ids
    }
    if set(indexed) != expected:
        raise ValueError("feasibility attempts differ from frozen manifest")

    active_coverage = {}
    effective_coverage = {}
    leaf_alpha_abs_mean = {}
    tangent_singular_values = {}
    tangent_relative_min = {}
    tangent_valid_attempts = {}
    signed_leaf_alpha_gradients = {}
    for base_key, attempt_ids in sorted(expected_attempt_ids_by_base.items()):
        selected = [indexed[(base_key, attempt_id)] for attempt_id in attempt_ids]
        if len(selected) != 100:
            raise ValueError("feasibility requires exactly 100 attempts per stratum")
        for block, site in FIT_KEYS:
            for family in FIT_FAMILIES:
                cell = f"{block}:{site}:{family}"
                # `identity` here means adaptive identity only; the immutable
                # anchor is recorded separately and cannot satisfy this bit.
                active = [bool(row["adaptive_active"][cell]) for row in selected]
                delta = np.asarray(
                    [row["unit_family_raw_logit_delta_l2"][cell] for row in selected],
                    dtype=np.float64,
                )
                if not np.isfinite(delta).all():
                    raise ValueError(f"nonfinite feasibility route delta: {base_key}/{cell}")
                report_cell = f"{base_key}|{cell}"
                active_coverage[report_cell] = float(np.mean(active))
                effective_coverage[report_cell] = float(
                    np.mean(delta > minimum_route_delta_l2)
                )
        signed_leaf_rows = []
        for row in selected:
            if type(row["outer_valid"]) is not bool:
                raise TypeError("outer_valid must be an exact bool")
            if row["outer_valid"]:
                signed_leaf_rows.append(row["signed_leaf_alpha_gradient"])
        signed_leaf = np.asarray(signed_leaf_rows, dtype=np.float64)
        if signed_leaf.ndim != 2 or signed_leaf.shape[1] != 4 or signed_leaf.shape[0] < 3:
            raise ValueError(f"insufficient signed alpha gradients for {base_key}")
        u_gradients = signed_leaf @ softmax_jacobian
        if (
            not np.isfinite(u_gradients).all()
            or np.abs(u_gradients.sum(axis=1)).max() > gradient_sum_tolerance
        ):
            raise ValueError(f"invalid simplex-tangent gradients for {base_key}")
        tangent = u_gradients @ helmert
        singular = np.linalg.svd(
            tangent / np.sqrt(tangent.shape[0]), compute_uv=False
        )
        relative_min = float(singular[-1] / max(singular[0], np.finfo(np.float64).tiny))
        if (
            singular.shape != (3,)
            or not np.isfinite(singular).all()
            or float(singular[-1]) < minimum_tangent_rms_singular_value
            or relative_min < minimum_tangent_relative_singular_value
        ):
            raise ValueError(f"rank-deficient coefficient tangent design for {base_key}")
        leaf_alpha_abs_mean[base_key] = [
            float(value) for value in np.abs(signed_leaf).mean(axis=0)
        ]
        tangent_singular_values[base_key] = [float(value) for value in singular]
        tangent_relative_min[base_key] = relative_min
        tangent_valid_attempts[base_key] = int(signed_leaf.shape[0])
        signed_leaf_alpha_gradients[base_key] = signed_leaf.tolist()

    if min(active_coverage.values()) < minimum_coverage:
        raise ValueError("adaptive key/family activation coverage failed")
    if min(effective_coverage.values()) < minimum_coverage:
        raise ValueError("effective key/family route coverage failed")
    return {
        "active_coverage": active_coverage,
        "effective_route_coverage": effective_coverage,
        "leaf_alpha_abs_mean_diagnostic": leaf_alpha_abs_mean,
        "tangent_rms_singular_values": tangent_singular_values,
        "tangent_relative_min": tangent_relative_min,
        "tangent_valid_attempts": tangent_valid_attempts,
        "signed_leaf_alpha_gradients": signed_leaf_alpha_gradients,
        "frozen_alpha": [float(value) for value in alpha],
        "attempts_per_stratum": 100,
        "base_keys": list(FIT_BASE_KEYS),
    }
~~~

The manifest writer writes the returned `encoded` bytes exactly, with no newline or self-hash field; therefore `attempt_manifest_sha256` everywhere is both the builder hash and the file-byte SHA-256. Each feasibility record is collected for the fixed attempt even when the outer label is invalid. `adaptive_active` is read from the pre-outer snapshot at every registered key; immutable anchor activity is a separate diagnostic. `unit_family_raw_logit_delta_l2` is the larger RGB/X pre-clipping raw-logit L2 change obtained by applying that family alone with unit coefficient, so a state that is technically active but has zero prior/background support does not pass. For each post-forward outer-valid attempt, compute and retain the signed gradient with respect to an independent leaf `alpha.shape==(4,)`; never discard its sign. At the current frozen alpha, multiply by the exact fixed-budget softmax Jacobian, verify each resulting `u` gradient sums to zero, project through the displayed orthonormal Helmert basis, and require the RMS design's three singular values to pass both the frozen absolute and relative minima in every stratum. Equal, nonzero leaf gradients therefore fail because their tangent projection is zero. Leaf-wise absolute means remain diagnostic only. The report also stores outer-valid count, actual writes, prefix-forward count, wall-clock time, every attempt/prefix hash, current-alpha hash, and all registry/checkpoint/manifest parents. Tests prove cells may be satisfied by different clips, activity at `(5,attn)` cannot satisfy `(9,ffn)`, an active zero-effect operator fails, equal nonzero leaf gradients fail tangent rank, three independent tangent directions pass, and any missing fixed attempt fails rather than triggering replacement sampling.

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_coefficients -v`

Expected: FAIL because coefficient fitter is absent.

- [ ] **Step 3a: Freeze the attempt manifest and implement the two feasibility audits**

Materialize all 2,000 attempted supersteps before the first model rollout. The candidate list contains only `(base_key, sequence_id, target_frame)` derived from the committed fit manifests and sequence lengths. Commit `fit_attempt_manifest.json` plus its hash to coefficient provenance. Run `audit_fit_feasibility()` on the first 100 manifested attempts in each stratum at `u_initial`; fitting cannot begin unless it passes. Calibration later reruns those exact attempts using the selected coefficient checkpoint; failure stops instead of selecting another checkpoint. Neither audit opens gate-confirmation data.

- [ ] **Step 3b: Implement frozen model guard and sequential fit step**

~~~python
@dataclass(frozen=True)
class FitStepResult:
    attempted_superstep: int
    successful_step: int | None
    loss: torch.Tensor | None = None
    reason: str | None = None

for model in models_by_base.values():
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
fit_devices = {next(model.parameters()).device for model in models_by_base.values()}
if len(fit_devices) != 1:
    raise ValueError("all six base models must share one fit device")
self.fit_device = next(iter(fit_devices))
self.coefficients = SharedRoutingCoefficients(
    alpha_budget=registry.alpha_budget,
    initial_u=torch.tensor(registry.coefficient_fit.u_initial),
).to(device=self.fit_device, dtype=torch.float32)
self.optimizer = torch.optim.Adam(
    [self.coefficients.u], lr=0.01, weight_decay=0.0
)
self.focal_loss = FocalLoss()
for _, controller, _ in self.components.values():
    controller.set_admission_source(FrozenRuleAdmissionSource(locked_calibration))

def clip_loss(self, base_key, clip):
    model, controller, processor = self.components[base_key]
    controller.begin_episode(reset_global=True)
    processor.begin_clip(clip)
    self._initialize_anchor(base_key)
    with torch.no_grad():
        for prefix_step in processor.warmup_prefix_steps():
            self._predict_and_commit_step(base_key, prefix_step)
        for role in ("history_tm2", "history_tm1", "candidate_t"):
            self._predict_and_commit(base_key, role)
    outer = self._forward_outer_with_coefficient_grad(base_key, "outer_tp1")
    self.record_pre_outer_activity(base_key, controller.snapshot())
    outer_target = processor.outer_target()
    if not bool(outer_target["valid"].reshape(-1)[0]):
        self.report_skipped_clip(base_key, reason="invalid t+1 outer label")
        return None
    return self.outer_loss(outer, outer_target)

def fit_superstep(self, clip_attempts_by_base, attempted_superstep):
    if attempted_superstep != self.attempted_supersteps:
        raise ValueError("attempted superstep is not the next manifested step")
    if attempted_superstep >= self.maximum_attempted_supersteps:
        raise RuntimeError("coefficient fitting exhausted attempted-superstep budget")
    self.validate_manifested_superstep(
        clip_attempts_by_base, attempted_superstep,
        expected_base_keys=FIT_BASE_KEYS,
        expected_attempts_per_base=2,
    )
    self.attempted_supersteps += 1
    self.optimizer.zero_grad(set_to_none=True)
    total_gradient = torch.zeros_like(self.coefficients.u)
    detached_stratum_means = []
    zero_valid_strata = []
    stratum_count = len(self.components)
    for base_key in sorted(self.components):  # seed 0..2, then modality name
        gradient_sum = torch.zeros_like(self.coefficients.u)
        detached_loss_sum = self.coefficients.u.new_zeros(())
        valid_count = 0
        for clip in clip_attempts_by_base[base_key]:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.clip_loss(base_key, clip)
            if loss is None:
                continue
            loss.backward()  # release this ViT graph before constructing the next one
            if self.coefficients.u.grad is None:
                raise RuntimeError("coefficient gradient is missing")
            gradient_sum.add_(self.coefficients.u.grad.detach())
            detached_loss_sum.add_(loss.detach())
            valid_count += 1
            del loss
        if valid_count == 0:
            self.report_skipped_superstep(base_key, reason="no valid outer labels")
            zero_valid_strata.append(base_key)
            continue
        total_gradient.add_(gradient_sum / float(stratum_count * valid_count))
        detached_stratum_means.append(detached_loss_sum / float(valid_count))
    self.optimizer.zero_grad(set_to_none=True)
    if zero_valid_strata:
        return FitStepResult(
            attempted_superstep=attempted_superstep,
            successful_step=None,
            reason="no valid outer labels for " + ",".join(zero_valid_strata),
        )
    self.coefficients.u.grad = total_gradient
    if not bool(torch.isfinite(total_gradient).all()):
        self.optimizer.zero_grad(set_to_none=True)
        return FitStepResult(
            attempted_superstep=attempted_superstep,
            successful_step=None,
            reason="nonfinite six-stratum aggregate gradient",
        )
    self.optimizer.step()
    self.successful_steps += 1
    return FitStepResult(
        attempted_superstep=attempted_superstep,
        successful_step=self.successful_steps,
        loss=torch.stack(detached_stratum_means).mean(),
    )
~~~

Define `FitStepResult` as an immutable record containing `attempted_superstep`, optional `successful_step`, optional detached `loss`, and a reason. `validate_manifested_superstep()` requires exactly the six `FIT_BASE_KEYS`, exactly two attempts per key, no duplicate, extra, or missing attempt, and byte-for-byte equality of each ordered `(attempted_superstep,base_key,ordinal_within_stratum,sequence_id,target_frame)` record with the frozen manifest slice before any rollout begins. The six base models must also have one parameter dtype policy compatible with the registered autocast policy; reject mixed devices or an unregistered mixed-dtype setup before the first clip. The fixed manifest is built only from fit sequence IDs/lengths, seed, and budgets, then committed before the feasibility pass. The fit iterator cycles deterministically through `(seed0 RGB-D, seed0 RGB-T, seed1 RGB-D, seed1 RGB-T, seed2 RGB-D, seed2 RGB-T)`. Each attempted superstep always consumes and records all twelve manifested clips, averages post-forward outer-valid gradients/losses inside each stratum, then gives each of the six strata weight `1/6`. Family inactivity is recorded but never rejects a clip. If one or more strata have zero valid outer labels, finish all later strata first, then discard all accumulated gradients and advance only `attempted_superstep`; an aggregate nonfinite gradient is handled the same way after all six strata. Never return before the final manifested clip and never replacement-sample. Reach 1,000 successful optimizer steps within 2,000 attempted supersteps or fail. Backpropagate and release each outer ViT graph immediately; retain only a four-scalar gradient sum per stratum, never a list of live loss graphs. Unit tests instrument graph finalizers and require at most one live outer graph, no model parameter gradient, and exact equality to the explicit six-stratum mean gradient. A zero-valid fixture places the failure in the first and middle strata and still requires all twelve manifest identities, forwards, and audit rows before an unsuccessful return with no optimizer update. Missing/extra/reordered/wrong-base/wrong-frame attempts fail before the first forward. Do not backpropagate through crop/state commits or stored factors.

`_predict_and_commit()` must call the shared `decode_windowed_center_head()` from Task 6 and assert batch one. It passes `decoded.crop_cxcywh[0]` (shape `[4]`) to `PredictionCenteredRolloutProcessor.commit_prediction()`, while preserving `decoded.crop_xywh` (shape `[B,4]`), `best_score [B]`, and Hann-fused response for `Stage0Controller.after_prediction()`. A shape-spy test fails if the processor receives `[B,4]` or the controller receives `[4]`. It must never commit `outputs["pred_boxes"]`. `_forward_outer_with_coefficient_grad()` uses the same windowed decode only to determine the reported prediction/crop state, while `outer_loss()` continues to consume the existing raw `pred_boxes/score_map` head outputs exactly as the training objective does.

- [ ] **Step 4: Reuse the existing tracking objective exactly**

Use the same normalized-box and heatmap path as `SEATrackActor.compute_losses`; only valid `t+1` outer labels contribute:

~~~python
import torch
import torch.nn.functional as F
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap

def outer_loss(self, pred_dict, outer_target):
    assert bool(outer_target["valid"].reshape(-1)[0])
    gt_xywh = outer_target["search_anno"].reshape(1, 4)
    gt_maps = generate_heatmap(
        gt_xywh.unsqueeze(0),
        self.cfg.DATA.SEARCH.SIZE,
        self.cfg.MODEL.BACKBONE.STRIDE,
    )[-1].unsqueeze(1)
    pred_boxes = pred_dict["pred_boxes"]
    if not torch.isfinite(pred_boxes).all():
        raise ValueError("nonfinite outer prediction")
    query_count = pred_boxes.shape[1]
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes).reshape(-1, 4)
    gt_xyxy = box_xywh_to_xyxy(gt_xywh)[:, None, :].repeat(
        1, query_count, 1
    ).reshape(-1, 4).clamp(0.0, 1.0)
    loss_giou, _ = giou_loss(pred_xyxy, gt_xyxy)
    loss_l1 = F.l1_loss(pred_xyxy, gt_xyxy)
    loss_focal = self.focal_loss(pred_dict["score_map"], gt_maps)
    return 2.0 * loss_giou + 5.0 * loss_l1 + loss_focal
~~~

Import `giou_loss`, `box_cxcywh_to_xyxy`, `box_xywh_to_xyxy`, `generate_heatmap`, and `FocalLoss` from the same repository modules used by `SEATrackActor`; instantiate `self.focal_loss = FocalLoss()` once. Do not replacement-sample an invalid outer label.

- [ ] **Step 5: Save complete coefficient provenance**

Checkpoint keys:

~~~python
{
    "u": coefficients.u.detach().cpu(),
    "alpha": coefficients.alpha.detach().cpu(),
    "alpha_budget": coefficients.alpha_budget,
    "successful_step": current_successful_step,
    "attempted_supersteps": attempted_supersteps,
    "fit_seed": fit_seed,
    "attempt_manifest_sha256": attempt_manifest_sha256,
    "resource_preflight_sha256": resource_preflight_sha256,
    "initial_feasibility_sha256": initial_feasibility_sha256,
    "base_checkpoint_sha256": base_hashes,
    "manifest_sha256": manifest_hashes,
    "calibration_registry_sha256": calibration_registry_hash,
    "locked_calibration_sha256": locked_calibration_hash,
    "code_commit": git_commit,
}
~~~

Each candidate checkpoint requires one registered `successful_step` in the candidate set, `attempted_supersteps <= 2000`, exactly six sorted base hashes, both modality manifest hashes, and the frozen attempt/preflight/initial-feasibility hashes. It cannot contain `selected_successful_step` or any later selection/result hash. Refuse to load it if the calibration-registry parent or locked rank/threshold artifact differs. Coefficient selection writes `selected_successful_step`, `selected_checkpoint_attempted_supersteps`, the selected candidate SHA-256, `fit_completed_successful_steps=1000`, and the actual `fit_completed_attempted_supersteps <= 2000` to its own JSON; the completed counters describe the whole fit, not merely the selected checkpoint. It copies the selected candidate byte-for-byte to `global_coefficients.pt`. The subsequent repeated selected-checkpoint feasibility audit is a child that records the coefficient-selection file SHA-256 and selected candidate SHA-256. The frozen registry then stores the selection, candidate, attempt-manifest, and both feasibility file hashes plus both selected/completed counters. A failed repeated audit stops; it never falls back to another checkpoint. No child hash is written back into a coefficient checkpoint or its selection record.

The CLI loads configs without mutating the module-global default:

~~~python
rgbt_cfg = copy.deepcopy(default_cfg)
rgbd_cfg = copy.deepcopy(default_cfg)
update_config_from_file(rgbt_yaml, base_cfg=rgbt_cfg)
update_config_from_file(rgbd_yaml, base_cfg=rgbd_cfg)
~~~

- [ ] **Step 6: Run graph-lifetime and registered 12-clip peak-memory smokes**

Run:

~~~bash
.venv/bin/python tracking/fit_spectral_coefficients.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --write-engineering-lock \
  --engineering-rank 16 --engineering-q-threshold 0.20 \
  --engineering-attention-floor 0.00 \
  --output output/spectral_s0/smoke/locked_calibration.json

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/fit_spectral_coefficients.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --locked-calibration output/spectral_s0/smoke/locked_calibration.json \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 \
  --split spectral_fit --attempts-per-stratum 2 \
  --maximum-attempted-supersteps 1 --checkpoint-successful-steps 0,1 \
  --fit-seed 20260713 \
  --engineering-smoke --profile-fit-memory \
  --max-peak-device-fraction 0.90 \
  --audit-output output/spectral_s0/smoke/fit_engineering_smoke.json \
  --output output/spectral_s0/smoke/memory_profile_coefficients.pt

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/fit_spectral_coefficients.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --locked-calibration output/spectral_s0/smoke/locked_calibration.json \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 \
  --split spectral_fit --attempts-per-stratum 1 \
  --maximum-attempted-supersteps 2 --checkpoint-successful-steps 0,1,2 \
  --fit-seed 20260713 \
  --engineering-smoke \
  --output output/spectral_s0/smoke/global_coefficients.pt
~~~

The first command writes `locked_calibration.json` with explicit engineering-only rank 16, numeric `q_memory` threshold `0.20`, and numeric attention floor `0.00`; it does not misinterpret a target coverage quantile as a threshold. The file is marked `engineering_smoke=true`, hashes its parent registry, and is rejected by calibration/gate modes. The second command constructs the full registered peak of 12 attempted clips per superstep (two in each of six strata), resets CUDA peak statistics immediately before the step, and fails on OOM or if `max_memory_allocated / total_device_memory > 0.90`; it records both byte values, elapsed time, causal-prefix forward count, and emitted audit bytes in `fit_engineering_smoke.json`. The unit graph-lifetime spy must also report `max_live_outer_graphs=1`. Expected fit JSON: `model_mode="eval"`, `trainable_names=["spectral_coefficients.u"]`, `base_models=6`, equal contribution counts, finite loss/alpha, exact five optimization-frame IDs plus exact causal prefix IDs, per-clip adaptive activity masks without activity-conditioned filtering, `prediction_centered_crops=true`, `max_live_outer_graphs=1`, and `base_parameter_hash_unchanged=true`.

- [ ] **Step 7: Verify the global fitter**

~~~bash
.venv/bin/python -m unittest tests.test_spectral_coefficients \
  tests.test_spectral_rollout tests.test_spectral_routing -v
repo="$PWD"
.venv/bin/python tracking/fit_spectral_coefficients.py --help >/dev/null
(cd /tmp && "$repo/.venv/bin/python" \
  "$repo/tracking/fit_spectral_coefficients.py" --help >/dev/null)
~~~

Expected: all three test modules PASS; both root and `/tmp` help invocations exit zero; the test report confirms one trainable name (`spectral_coefficients.u`), six frozen base models, `max_live_outer_graphs=1`, and unchanged base-parameter hashes.

- [ ] **Step 8: Commit the global fitter**

~~~bash
git add lib/train/spectral/__init__.py \
  lib/train/spectral/coefficient_fit.py tracking/fit_spectral_coefficients.py \
  tests/test_spectral_coefficients.py
git commit -m "feat: fit shared spectral routing coefficients"
~~~

### Task 10: Recorded GT-Free Schedule, Matched Controls, and S0 Runner

**Files:**

- Create: `lib/test/evaluation/spectral_s0.py`
- Create: `lib/test/evaluation/spectral_corruption.py`
- Create: `tracking/record_spectral_schedule.py`
- Create: `tracking/run_spectral_s0.py`
- Create: `tests/test_spectral_s0_evaluation.py`
- Modify: `knowledge_base/Target-Spectral-S0-实验记录.md`

**Interfaces:**

- `record_spectral_schedule` accepts only routing-disabled frozen legacy outputs.
- JSONL key is `(condition, dataset, sequence, frame_index)` and records `q_memory`, asymmetry, `paired_valid`, scheduled admit, actual-commit outcome, checkpoint/config/registry hashes.
- `ControlSpec` enumerates every required S0 row.
- `CorruptionBurstSpec` creates exactly one evaluator-side X-blackout burst per eligible sequence.
- `assert_matched_control_budget()` compares state bytes, coefficient capacity, residual budgets, layers/modules, and schedule hash.
- Engineering smoke may use a calibration registry but returns `gate_decision=null`.

- [ ] **Step 1: Write failing schedule and control tests**

Require:

~~~python
REQUIRED_CORE_CONTROLS = {
    "routing_disabled_legacy", "zero_strength_instrumented",
    "confidence_only_scalar_history", "random_orthogonal", "pooled_same",
    "target_balanced_identity", "full_four_spectrum",
    "rgbx_pair_shuffle", "temporal_order_shuffle",
    "target_background_mask_shuffle",
}
REQUIRED_BRANCH_CONTROLS = {
    "identity_only", "private_only", "dynamic_only", "background_only",
    "full_minus_identity", "full_minus_private",
    "full_minus_dynamic", "full_minus_background",
    "full_minus_identity_strength_matched",
    "full_minus_private_strength_matched",
    "full_minus_dynamic_strength_matched",
    "full_minus_background_strength_matched",
}
REQUIRED_CUMULATIVE_CONTROLS = {
    "identity_plus_dynamic",
    "identity_plus_dynamic_plus_private",
    "full_four_spectrum",
}
~~~

Tests fail if pair shuffle is absent, a control changes legacy image/HMoE input, schedule hashes differ, inactive state is unreported, or a schedule is recorded from any adaptive row. Add tests that all rows replay identical `(q_memory, asymmetry, paired_valid, scheduled_admit)`, a low-confidence modality skips all four paired families, raw LOO masks never renormalize alpha while mandatory strength-matched LOO exactly restores the frozen total budget, every active/core/branch/cumulative/shuffle row has strength exactly `1.0`, only `zero_strength_instrumented` has `0.0`, and temporal shuffle never consumes a factor with source frame `> t-1`. A runner fixture must persist the returned `target_spectral` payload into the current condition's `frames.jsonl`, then prove `sanitize_previous_output()` drops it before the next tracker call. A dual-condition fixture requires disjoint `<method>/clean` and `<method>/registered_corruption` files/hashes and fails on a mixed path. A clean full-row fixture requires separate locked-rank cells for immutable anchor, adaptive identity, dynamic, private, and background sources and verifies that diagnostic accumulation cannot mutate runtime bank bytes.

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_spectral_s0_evaluation -v`

Expected: FAIL on missing `spectral_s0.py`.

- [ ] **Step 3: Record the common schedule from legacy outputs**

Implement canonical row construction and make any adaptive source fail closed:

~~~python
@dataclass(frozen=True, order=True)
class ScheduleKey:
    condition: str
    dataset: str
    sequence: str
    frame_index: int

@dataclass(frozen=True)
class RecordedAdmission:
    key: ScheduleKey
    q_memory: float
    asymmetry: float
    paired_valid: bool
    scheduled_admit: bool
    proposed_writes: int
    factor_rejections: int
    actual_committed: bool
    checkpoint_sha256: str
    config_sha256: str
    registry_sha256: str

    def as_json(self):
        return {
            "condition": self.key.condition,
            "dataset": self.key.dataset,
            "sequence": self.key.sequence,
            "frame_index": self.key.frame_index,
            "q_memory": self.q_memory,
            "asymmetry": self.asymmetry,
            "paired_valid": self.paired_valid,
            "scheduled_admit": self.scheduled_admit,
            "proposed_writes": self.proposed_writes,
            "factor_rejections": self.factor_rejections,
            "actual_committed": self.actual_committed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
        }

def make_recorded_admission(key, diagnostics, locked, hashes):
    if key.condition not in {"clean", "registered_corruption"}:
        raise ValueError("invalid schedule condition")
    if type(key.frame_index) is not int or key.frame_index < 1:
        raise ValueError("schedule frame_index must be a positive int")
    if diagnostics.get("prediction_committed") is not True:
        raise RuntimeError("schedule observables require a committed prediction")
    if set(hashes) != {"checkpoint", "config", "registry"}:
        raise ValueError("schedule requires exactly three parent hashes")
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes.values()):
        raise ValueError("schedule parents require lowercase SHA-256 values")
    q_memory = float(diagnostics["q_memory"])
    asymmetry = float(diagnostics["asymmetry"])
    paired_valid = diagnostics["paired_valid"]
    if type(paired_valid) is not bool:
        raise TypeError("paired_valid must be bool")
    if not math.isfinite(q_memory) or not math.isfinite(asymmetry):
        raise ValueError("schedule observables must be finite")
    scheduled = q_memory >= float(locked["q_memory_threshold"]) and paired_valid
    proposed_writes = int(diagnostics["proposed_writes"])
    factor_rejections = int(diagnostics["factor_rejections"])
    if proposed_writes < 0 or factor_rejections < 0:
        raise ValueError("schedule counts must be nonnegative")
    actual_committed = diagnostics["actual_committed"]
    if type(actual_committed) is not bool:
        raise TypeError("actual_committed must be bool")
    if actual_committed and not scheduled:
        raise ValueError("an actual commit requires scheduled admission")
    return RecordedAdmission(
        key=key,
        q_memory=q_memory,
        asymmetry=asymmetry,
        paired_valid=paired_valid,
        scheduled_admit=scheduled,
        proposed_writes=proposed_writes,
        factor_rejections=factor_rejections,
        actual_committed=actual_committed,
        checkpoint_sha256=hashes["checkpoint"],
        config_sha256=hashes["config"],
        registry_sha256=hashes["registry"],
    )

def write_canonical_schedule(path, rows):
    ordered = sorted(rows, key=lambda row: row.key)
    if len({row.key for row in ordered}) != len(ordered):
        raise ValueError("duplicate schedule key")
    payload = "".join(
        json.dumps(
            row.as_json(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n"
        for row in ordered
    ).encode("utf-8")
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

def hash_model_parameters(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

class LegacyScheduleRecorder:
    def __init__(self, tracker, locked, hashes, control_name):
        if control_name != "routing_disabled_legacy":
            raise ValueError("schedule source must be routing-disabled frozen legacy")
        self.tracker = tracker
        self.locked = locked
        self.hashes = hashes
        self.parameter_hash = hash_model_parameters(tracker.network)

    def record_frame(self, image, tracker_info, key):
        output = self.tracker.track(image, info=tracker_info)
        diagnostics = output["target_spectral"]
        row = make_recorded_admission(key, diagnostics, self.locked, self.hashes)
        if hash_model_parameters(self.tracker.network) != self.parameter_hash:
            raise RuntimeError("schedule recording changed model parameters")
        return row
~~~

The recorder initializes the observer-instrumented, routing-disabled row, calls `record_frame()` only after the tracker has committed its prediction, and stores its canonical JSONL SHA-256 plus proposed/scheduled/factor-rejection/actual-commit audit counts. Every replay row consumes the recorded tuple instead of replacing it with recomputed observables; recomputation is diagnostic-only and tolerance checked.

- **Matched-control implementation:** execute these independently reviewable actions:

- [ ] **Step 4a: Add the closed `ControlSpec` table and frozen-alpha masks**
- [ ] **Step 4b: Add the shared canonical SHA-256 CPU RNG helper**
- [ ] **Step 4c: Add confidence-only and random-orthogonal route operators**
- [ ] **Step 4d: Add pooled-same and target-balanced-identity factor builders**
- [ ] **Step 4e: Add token-preserving pair shuffle**
- [ ] **Step 4f: Add past-only temporal-ring shuffle**
- [ ] **Step 4g: Add common-valid target/background mask shuffle**
- [ ] **Step 4h: Add exact matched-budget validation and focused fixtures**

~~~python
import hashlib
import math
from dataclasses import dataclass
import torch
from lib.models.target_spectral.memory import factor_columns, operator_from_thin_svd
from lib.models.target_spectral.types import WeightedFactors

FAMILY_ORDER = ("identity", "dynamic", "private", "background")
ATTRIBUTION_MASKS = {
    "identity_only": (1, 0, 0, 0),
    "dynamic_only": (0, 1, 0, 0),
    "private_only": (0, 0, 1, 0),
    "background_only": (0, 0, 0, 1),
    "full_minus_identity": (0, 1, 1, 1),
    "full_minus_dynamic": (1, 0, 1, 1),
    "full_minus_private": (1, 1, 0, 1),
    "full_minus_background": (1, 1, 1, 0),
    "identity_plus_dynamic": (1, 1, 0, 0),
    "identity_plus_dynamic_plus_private": (1, 1, 1, 0),
    "full_four_spectrum": (1, 1, 1, 1),
}
STRENGTH_MATCHED_LOO = {
    "full_minus_identity_strength_matched": (0, 1, 1, 1),
    "full_minus_dynamic_strength_matched": (1, 0, 1, 1),
    "full_minus_private_strength_matched": (1, 1, 0, 1),
    "full_minus_background_strength_matched": (1, 1, 1, 0),
}
SPECIAL_GEOMETRIES = {
    "confidence_only_scalar_history": "confidence_only",
    "random_orthogonal": "random_orthogonal",
    "pooled_same": "pooled_same",
    "target_balanced_identity": "target_balanced_identity",
}
SHUFFLES = {
    "rgbx_pair_shuffle": "pair",
    "temporal_order_shuffle": "temporal",
    "target_background_mask_shuffle": "mask",
}
KNOWN_CONTROLS = frozenset({
    "routing_disabled_legacy", "zero_strength_instrumented",
    *ATTRIBUTION_MASKS, *STRENGTH_MATCHED_LOO,
    *SPECIAL_GEOMETRIES, *SHUFFLES,
})

@dataclass(frozen=True)
class ControlSpec:
    name: str
    geometry: str
    strength: float | None
    attribution_mask: tuple[int, int, int, int] | None = None
    shuffle: str | None = None

def control_spec(name):
    if name not in KNOWN_CONTROLS:
        raise ValueError(f"unknown registered control: {name}")
    if name == "routing_disabled_legacy":
        return ControlSpec(name, "disabled", None)
    if name == "zero_strength_instrumented":
        return ControlSpec(name, "registered", 0.0)
    return ControlSpec(
        name=name,
        geometry=SPECIAL_GEOMETRIES.get(name, "registered"),
        strength=1.0,
        attribution_mask=(
            ATTRIBUTION_MASKS.get(name) or STRENGTH_MATCHED_LOO.get(name)
        ),
        shuffle=SHUFFLES.get(name),
    )

def masked_frozen_alpha(spec, frozen_alpha):
    alpha = torch.as_tensor(frozen_alpha)
    if alpha.shape != (4,) or not torch.isfinite(alpha).all():
        raise ValueError("frozen alpha must be four finite values")
    if spec.geometry == "disabled":
        return torch.zeros_like(alpha)
    if spec.attribution_mask is None:
        return alpha.clone()
    masked = alpha * alpha.new_tensor(spec.attribution_mask)
    if spec.name in STRENGTH_MATCHED_LOO:
        if float(masked.sum()) <= 0.0:
            raise ValueError("strength-matched LOO has no surviving coefficient")
        masked = masked * (alpha.sum() / masked.sum())
    return masked

def canonical_seed64(*parts):
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")

def canonical_randperm(length, *parts):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(canonical_seed64(*parts))
    return torch.randperm(length, generator=generator, device="cpu")

def canonical_random_basis(dimension, rank, *parts):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(canonical_seed64(*parts))
    gaussian = torch.randn(dimension, rank, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(gaussian, mode="reduced")
    pivot = basis.abs().argmax(dim=0)
    columns = torch.arange(rank)
    signs = torch.where(basis[pivot, columns] < 0, -1.0, 1.0)
    return basis * signs.unsqueeze(0)

def confidence_only_route(rows, prior, rho, operator_scale,
                          alpha_budget, strength=1.0):
    if rows.ndim != 2 or prior.shape not in {(rows.shape[0],), (rows.shape[0], 1)}:
        raise ValueError("rows/prior shape mismatch")
    prior = prior.reshape(-1, 1).to(device=rows.device, dtype=rows.dtype)
    scale = rows.new_tensor(float(strength * rho * operator_scale * alpha_budget))
    return rows + scale * prior * rows

def random_orthogonal_apply(rows, shrinkage, *seed_parts):
    shrinkage = torch.as_tensor(shrinkage)
    basis = canonical_random_basis(
        rows.shape[-1], shrinkage.numel(), *seed_parts
    ).to(rows)
    values = shrinkage.to(rows)
    return ((rows @ basis) * values) @ basis.transpose(0, 1)

def pooled_same_families(template_rows, search_rows):
    template_rows = torch.as_tensor(template_rows)
    search_rows = torch.as_tensor(
        search_rows, device=template_rows.device, dtype=template_rows.dtype
    )
    if template_rows.ndim != 2 or search_rows.ndim != 2:
        raise ValueError("pooled rows must be [M,D]")
    if not template_rows.shape[0] or not search_rows.shape[0]:
        raise ValueError("both scopes require at least one valid row")
    vectors = torch.cat((template_rows, search_rows), dim=0)
    weights = torch.cat((
        vectors.new_full((template_rows.shape[0],), 0.5 / template_rows.shape[0]),
        vectors.new_full((search_rows.shape[0],), 0.5 / search_rows.shape[0]),
    ))
    chunk = WeightedFactors(vectors=vectors, weights=weights)
    return {family: (chunk,) for family in FAMILY_ORDER}

def target_balanced_identity_operator(
    anchor_state, adaptive_identity_state, locked_rank, anchor_weight=0.5,
):
    weight = float(anchor_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("anchor_weight must lie in [0,1]")
    columns = [math.sqrt(weight) * factor_columns(anchor_state)]
    if adaptive_identity_state is not None:
        columns.append(
            math.sqrt(1.0 - weight) * factor_columns(adaptive_identity_state)
        )
    return operator_from_thin_svd(
        torch.cat(columns, dim=1),
        max_rank=2 * int(locked_rank),
        truncate_nonzero=False,
    )

def target_balanced_identity_route(
    rows, prior, identity_operator, rho, operator_scale,
    alpha_budget, strength=1.0,
):
    prior = prior.reshape(-1, 1).to(rows)
    update = prior * identity_operator.apply_rows(rows)
    scale = rows.new_tensor(
        float(strength * rho * operator_scale * alpha_budget)
    )
    return rows + scale * update

def shuffle_x_token_partners(x_rows, sorted_tokens, *seed_parts):
    if x_rows.ndim != 3:
        raise ValueError("x_rows must be [common_tokens,slots,dimension]")
    tokens = torch.as_tensor(sorted_tokens, dtype=torch.long, device="cpu")
    if tokens.numel() != x_rows.shape[0] or not torch.equal(
        tokens, tokens.sort().values
    ):
        raise ValueError("x_rows must align with sorted common global tokens")
    permutation = canonical_randperm(tokens.numel(), *seed_parts)
    return x_rows[permutation.to(x_rows.device)].clone(), permutation

def shuffle_mask_on_common_valid(mask, common_valid, *seed_parts):
    if mask.shape != common_valid.shape:
        raise ValueError("mask/common-valid shape mismatch")
    indices = torch.nonzero(
        common_valid.detach().cpu().reshape(-1), as_tuple=False
    ).flatten()
    permutation = canonical_randperm(indices.numel(), *seed_parts)
    output = torch.zeros_like(mask).reshape(-1)
    device_indices = indices.to(mask.device)
    output[device_indices] = mask.reshape(-1)[indices[permutation].to(mask.device)]
    return output.reshape_as(mask), permutation

def permute_past_ring_entries(entries, frame_index, *seed_parts):
    ordered = sorted(entries, key=lambda entry: (
        entry.source_frame, entry.block, entry.site, entry.scope, entry.family,
    ))
    if any(entry.source_frame >= frame_index for entry in ordered):
        raise ValueError("temporal shuffle cannot consume current/future factors")
    permutation = canonical_randperm(len(ordered), *seed_parts)
    return [ordered[index] for index in permutation.tolist()], permutation

@dataclass(frozen=True)
class ControlBudget:
    persistent_state_bytes: int
    peak_temporary_bytes_limit: int
    coefficient_capacity: int
    coefficient_owner_sha256: str
    calibration_exposure_sha256: str
    locked_rank: int
    operator_norm_cap: float
    residual_budget: float
    layers: tuple[int, ...]
    modules: tuple[str, ...]
    schedule_sha256: str
    state_update_semantics: str

def assert_matched_control_budget(reference, candidate):
    fields = tuple(ControlBudget.__dataclass_fields__)
    mismatches = [
        field for field in fields
        if getattr(reference, field) != getattr(candidate, field)
    ]
    if mismatches:
        raise ValueError(f"unmatched control budget: {mismatches}")
~~~

Add these focused fixtures to `tests/test_spectral_s0_evaluation.py`:

~~~python
@dataclass(frozen=True)
class _RingEntry:
    source_frame: int
    block: int
    site: str
    scope: str
    family: str

class ControlImplementationTests(unittest.TestCase):
    def test_control_table_is_closed_and_masks_never_renormalize(self):
        with self.assertRaisesRegex(ValueError, "unknown registered control"):
            control_spec("unregistered")
        frozen = torch.tensor([0.04, 0.05, 0.07, 0.09])
        masked = masked_frozen_alpha(
            control_spec("full_minus_private"), frozen
        )
        torch.testing.assert_close(
            masked, torch.tensor([0.04, 0.05, 0.00, 0.09])
        )
        self.assertAlmostEqual(float(masked.sum()), 0.18, places=7)
        strength_matched = masked_frozen_alpha(
            control_spec("full_minus_private_strength_matched"), frozen
        )
        self.assertEqual(float(strength_matched[2]), 0.0)
        torch.testing.assert_close(strength_matched.sum(), frozen.sum())
        self.assertEqual(
            control_spec("zero_strength_instrumented").strength, 0.0
        )
        self.assertTrue(all(
            control_spec(name).strength == 1.0
            for name in KNOWN_CONTROLS
            if name not in {
                "routing_disabled_legacy", "zero_strength_instrumented"
            }
        ))

    def test_pair_shuffle_uses_compact_positions_not_global_ids(self):
        rows = torch.tensor([[[30.0]], [[110.0]], [[290.0]]])
        shuffled, permutation = shuffle_x_token_partners(
            rows, [3, 11, 29], 20260713, "lasher", "seqA", 5,
            5, "attn", "search", "pair_shuffle",
        )
        self.assertEqual(permutation.tolist(), [0, 2, 1])
        torch.testing.assert_close(shuffled, rows[[0, 2, 1]])

    def test_registered_mask_and_temporal_permutations_are_exact(self):
        mask_permutation = canonical_randperm(
            5, 20260713, "lasher", "seqA", 5, 5, "attn", "search",
            "target", "mask_shuffle",
        )
        self.assertEqual(mask_permutation.tolist(), [2, 4, 3, 1, 0])
        entries = [
            _RingEntry(frame, 5, "attn", "search", "identity")
            for frame in range(1, 9)
        ]
        shuffled, permutation = permute_past_ring_entries(
            entries, 9, 20260713, "lasher", "seqA", 9, 5, "attn",
            "search", "identity", "temporal_shuffle",
        )
        self.assertEqual(permutation.tolist(), [5, 0, 1, 2, 7, 4, 3, 6])
        self.assertEqual(
            [entry.source_frame for entry in shuffled], [6, 1, 2, 3, 8, 5, 4, 7]
        )
        with self.assertRaisesRegex(ValueError, "current/future"):
            permute_past_ring_entries(entries, 8, 20260713)

    def test_random_basis_has_canonical_column_signs(self):
        basis = canonical_random_basis(
            12, 4, 20260714, 5, "attn", "search", "identity",
            "random_orthogonal",
        )
        pivots = basis.abs().argmax(dim=0)
        columns = torch.arange(basis.shape[1])
        self.assertTrue((basis[pivots, columns] >= 0).all())
        torch.testing.assert_close(
            basis.transpose(0, 1) @ basis,
            torch.eye(4, dtype=torch.float64), atol=1e-12, rtol=1e-12,
        )
~~~

`target_balanced_identity` does not obtain its alpha through a branch mask: its route multiplies the full `alpha_budget` by the once-shrunk operator returned from `target_balanced_identity_operator()`. Pass the adaptive state only after Task 2's activity gate; otherwise pass `None`. The three other families use counted inert buffers. `pooled_same_families()` receives only observer-RMS-normalized, common-valid RGB/X rows after the caller concatenates modalities within each scope.

- The original branch-only, cumulative, and raw leave-one-out controls apply a binary mask directly to frozen `alpha`: `alpha_control = mask * alpha_frozen`; they remain descriptive mechanism diagnostics and do not renormalize. The four mandatory `*_strength_matched` LOO rows multiply the surviving masked coefficients by `alpha_frozen.sum()/masked.sum()`, retain the same total alpha budget/operator cap/residual bounds, and are the binding family-attribution controls. The historical suffix means **alpha-budget-matched only**, not realized operator- or logit-norm matched: every such row must report unclipped/clipped route-delta L2 distributions beside the full row, and no claim may call their realized routing strengths equal.
- Cumulative rows use masks `[1,0,0,0]`, `[1,1,0,0]`, `[1,1,1,0]`, and `[1,1,1,1]` in frozen order identity, dynamic, private, background.
- `confidence_only_scalar_history` is the registered confidence-only comparator: its only retained history is the bounded committed EMA scalar `rho`; it uses no eigenspace or learned geometry. Its update is `U = prior * rows` and `routed = rows + strength * rho * operator_scale * alpha_budget * U`. It allocates one identity-sized inert operator buffer plus padding to the exact full-bank byte count. Never describe it as an eigenspace-memory baseline.
- `random_orthogonal` creates, for every key/family, a fixed Gaussian matrix by colon-encoding `(controls.random_orthogonal_seed,block,site,scope,family,"random_orthogonal")`, using the first eight SHA-256 bytes as an unsigned big-endian seed for a CPU `torch.Generator`, and drawing `torch.randn` in canonical `[D,locked_rank]` shape. It obtains `Q` with QR and makes each column's largest-absolute, lowest-index-tie entry nonnegative, then applies `Q diag(shrinkage_full) Q^T`. Its matched state update retains the observed full basis/eigenvalues solely to reproduce eigenvalue history and identical state bytes, but routing never reads that basis. `Q` is regenerated statelessly into counted temporary memory, so there is no extra persistent buffer.
- `pooled_same` forms `V_pool` from all valid observer-RMS-normalized RGB/X template/search rows without target/background masks or common/private decomposition. Template and search are separately unit-mass normalized and weighted `0.5/0.5`; identical pooled chunks update all four family buffers, so frozen alpha still sums to the same budget.
- `target_balanced_identity` uses only the separately normalized immutable anchor and target-common adaptive identity factors, with `alpha_budget` on that operator and three inert family buffers.
- `full_four_spectrum` uses the four registered factor families and frozen alpha unchanged.
- Inactive branches allocate inert padding counted under `padding_state_bytes`.
- Pair shuffle permutes X observation partners only after raw HMoE computation. Canonically encode `(controls.shuffle_seed,dataset,sequence,frame_index,block,site,scope,"pair_shuffle")` as colon-separated UTF-8, take the first eight bytes of SHA-256 as an unsigned big-endian seed for a CPU `torch.Generator`, and apply `torch.randperm` to the sorted common global-token partners. Apply the same token permutation to every slot so slots within a token are never separated; RGB rows, images, routing inputs, weights, and global indices stay fixed.
- Freeze the pair-alignment audit exactly. For every eligible frame and key `(block in {5,9}, site in {attn,ffn}, scope in {template,search})`, let `r_rgb_i,r_x_i` be the Task 3 observer-RMS-normalized paired rows and let `w_i` be the unit-mass anchor weight for template or response-target row weight for search, divided by slots. Define `A_match(t,k)=sum_i w_i*cos(r_rgb_i,r_x_i)` and `A_shuffle(t,k)=sum_i w_i*cos(r_rgb_i,r_x_perm(i))`, with cosine denominator clamped by `1e-8`; the frame delta is their difference. A frame/key is eligible only when it has at least two positive-weight paired rows and finite terms. Average eligible frame deltas within each sequence, then retain the exact sorted `{sequence_id: delta}` map separately for every `(base_seed,benchmark,key)` cell. For each cell, derive an independent CPU RNG seed from SHA-256 of `(statistics.bootstrap_seed,cell_name,"pair-alignment")`, run exactly 10,000 paired sequence-bootstrap replicates, and take `numpy.quantile(replicates,0.025,method="linear")`. Every one of the `3*2*8=48` cell LCBs must be strictly greater than `pair_alignment_min_lcb=0.02`; fail if a cell has no eligible sequence and record all eligible/excluded counts. The shuffled partners themselves always use `controls.shuffle_seed`.

  The pair artifact stores the raw per-sequence deltas, derived seeds, recomputed LCBs, exact sequence support, and explicit `calibration_registry_sha256`, `locked_admission_sha256`, `rank_sketches_sha256`, `base_checkpoint_index_sha256`, `calibration_manifest_sha256_by_modality` with exactly the `rgbd` and `rgbt` hashes, `statistics.bootstrap_seed`, and `controls.shuffle_seed`. The analyzer writes and hashes this parent-only artifact first, then writes `locked_calibration.json` with `pair_alignment_audit_sha256`; the pair artifact never points back to locked calibration. Freeze and final audit rehash the file, require all parents/seeds/support to match, and independently rebuild every LCB from the raw sequence map. This audit is only a common/private paired-space sanity check under anchor/target weights. It is not background-weighted family evidence and does not by itself prove any of the four family claims; those rely on the joint endpoint attribution gates.
- Temporal shuffle maintains a fixed `K=controls.temporal_ring_length=8` ring per key/family of already observed, admitted factor summaries plus the exact boundary snapshot preceding that ring. Store ring entries in canonical `(source_frame, block, site, scope, family)` order. Before frame `t`, assert every eligible source frame is `<t`, encode `(controls.shuffle_seed,dataset,sequence,t,block,site,scope,family,"temporal_shuffle")` as colon-separated UTF-8, seed a CPU `torch.Generator` from the first eight SHA-256 bytes interpreted unsigned big-endian, and apply `torch.randperm` to exactly those canonical entries. Reconstruct from the boundary snapshot and the permuted entries; the set/count is unchanged and no future factor is visible. The full row allocates the same ring bytes and replays the same entries chronologically. Tests freeze exact permutations for a fixture, including wraparound and equal-frame tie ordering.
- Target/background shuffle uses independent canonical permutations. Encode `(controls.shuffle_seed,dataset,sequence,frame_index,block,site,scope,mask_kind,"mask_shuffle")`, where `mask_kind` is `target` or `background`, exactly as above; seed a CPU `torch.Generator` from the first eight SHA-256 bytes and apply `torch.randperm` only over sorted common-valid cells. Scatter back to the original grid, leaving invalid cells zero and preserving each mask's cardinality and L1 mass. Use the same permuted mask for RGB/X and all slots; images, raw HMoE rows, prior, and schedule remain unchanged.
- Every row receives the same schedule cursor and residual budgets.

- [ ] **Step 5: Implement the registered evaluator-side corruption**

~~~python
@dataclass(frozen=True)
class CorruptionBurstSpec:
    severity: float
    start: int
    length: int

    @classmethod
    def for_sequence(cls, num_frames, severity):
        search_frames = num_frames - 1
        start = max(1, math.floor(0.4 * search_frames))
        length = 20
        if start + length + 5 > num_frames:
            return None
        return cls(float(severity), start, length)

def apply_x_blackout(image, spec, frame_index):
    corrupted = image.copy()
    if spec.start <= frame_index < spec.start + spec.length:
        x = corrupted[..., 3:]
        scaled = (1.0 - spec.severity) * x.astype(np.float64)
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            scaled = np.rint(scaled).clip(info.min, info.max)
        corrupted[..., 3:] = scaled.astype(x.dtype)
    return corrupted
~~~

There is exactly one burst per eligible sequence. RGB channels remain bitwise unchanged; X is rounded/clipped back to its original dtype. Ineligible short sequences are marked, not moved or shortened. The recorder seals the same `start/end/severity` for every row. Corruption metadata stays in the evaluator and the causal sentinel fails if any of it reaches `track()`.

- [ ] **Step 6: Implement fail-closed S0 execution**

Implement one writer per method and condition; never share an open writer across conditions:

~~~python
def write_strict_json(path, value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    Path(path).write_text(payload, encoding="utf-8", newline="\n")

def write_strict_jsonl(path, rows):
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )
    Path(path).write_text(payload, encoding="utf-8", newline="\n")

class ConditionResultWriter:
    CONDITIONS = frozenset({"clean", "registered_corruption"})

    def __init__(self, root, dataset, method, condition, frozen_hashes):
        if condition not in self.CONDITIONS:
            raise ValueError(f"invalid condition: {condition}")
        if frozen_hashes["registry_status"] != "frozen":
            raise ValueError("gate execution requires a frozen registry")
        self.condition = condition
        self.schedule_sha256 = frozen_hashes["schedule_sha256"]
        self.directory = Path(root) / dataset / method / condition
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.directory / "frames.jsonl"
        self.sequences_path = self.directory / "sequences.json"
        self.metrics_path = self.directory / "benchmark_metrics.json"
        if any(path.exists() for path in (
            self.frames_path, self.sequences_path, self.metrics_path
        )):
            raise FileExistsError(f"refusing to overwrite result row: {self.directory}")
        self._frames = []
        self._sequences = []
        self._hashes = dict(frozen_hashes)

    def add_frame(self, row):
        if row["condition"] != self.condition:
            raise ValueError("frame condition/path mismatch")
        if row["schedule_sha256"] != self.schedule_sha256:
            raise ValueError("frame schedule hash mismatch")
        if type(row["actual_committed"]) is not bool:
            raise TypeError("actual_committed must be one frame-level bool")
        self._frames.append(dict(row))

    def add_sequence(self, row):
        if row["condition"] != self.condition:
            raise ValueError("sequence condition/path mismatch")
        self._sequences.append(dict(row))

    def close(self):
        write_strict_jsonl(self.frames_path, self._frames)
        write_strict_json(self.sequences_path, {
            "condition": self.condition,
            "schedule_sha256": self.schedule_sha256,
            "hashes": self._hashes,
            "sequences": self._sequences,
        })

def locked_rank_cell(state, rank):
    lambda_r = state.eigenvalues[rank - 1]
    lambda_next = boundary_eigenvalue(state, rank)
    return {
        "rank": int(rank),
        "trace_energy": float(trace_energy(state, rank)),
        "lambda_r": float(lambda_r),
        "lambda_r_plus_1": float(lambda_next),
        "relative_gap": float(rank_gap(state, rank)),
    }
~~~

`write_strict_json()` and `write_strict_jsonl()` use sorted keys, compact separators, UTF-8/LF, and `allow_nan=False`. Each per-frame row contains predictions, evaluator input path, all frozen parent hashes, detached block/site/scope/modality/partition Combine diagnostics, JSD, top-1/top-2 overlap, logit drift, proposed/scheduled/rejected/committed counts, p50/p95 latency, byte counts, and forbidden-key accesses. Set `actual_committed=True` only when the one atomic bank commit advances the frame version with at least one valid family; never sum family/key writes.

For clean `full_four_spectrum`, call `locked_rank_cell()` on detached clones of every immutable anchor and admitted adaptive identity/dynamic/private/background source, keyed by `(base_seed,benchmark,block,site,scope,source_family)`. Do not pass the temporary anchor+adaptive concatenation and do not write any diagnostic tensor back to the bank. Gate mode rejects an unfrozen registry, unsealed schedule manifest, missing/mismatched hash, mixed condition path, or pre-existing output file.

- [ ] **Step 7: Run a one-sequence engineering smoke**

~~~bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/record_spectral_schedule.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --config rgbt_spectral_s0 \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --base-seed 0 \
  --coefficient-checkpoint output/spectral_s0/smoke/global_coefficients.pt \
  --locked-calibration output/spectral_s0/smoke/locked_calibration.json \
  --dataset lasher --split spectral_calibration --max-sequences 1 \
  --engineering-smoke \
  --output output/spectral_s0/smoke/schedule.jsonl

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/run_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --config rgbt_spectral_s0 \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --base-seed 0 \
  --coefficient-checkpoint output/spectral_s0/smoke/global_coefficients.pt \
  --locked-calibration output/spectral_s0/smoke/locked_calibration.json \
  --dataset lasher --split spectral_calibration \
  --methods routing_disabled_legacy,confidence_only_scalar_history,full_four_spectrum,rgbx_pair_shuffle \
  --schedule output/spectral_s0/smoke/schedule.jsonl \
  --max-sequences 1 --engineering-smoke \
  --output output/spectral_s0/smoke
~~~

Expected: four identical schedule hashes and recorded tuples, zero forbidden accesses, legacy and disabled outputs bitwise identical, matched mechanism budgets, coverage at least 0.20 on the synthetic family-activation fixture, all four families activated there, and `gate_decision=null`.

- [ ] **Step 8: Verify schedules, controls, and corruption**

~~~bash
.venv/bin/python -m unittest tests.test_spectral_s0_evaluation \
  tests.test_spectral_stage0_tracker tests.test_spectral_causality -v
repo="$PWD"
for script in tracking/record_spectral_schedule.py tracking/run_spectral_s0.py; do
  .venv/bin/python "$script" --help >/dev/null
  (cd /tmp && "$repo/.venv/bin/python" "$repo/$script" --help >/dev/null)
done
~~~

Expected: all three test modules PASS and all four root/`/tmp` CLI help invocations exit zero; no schedule/control test reports a hash, strength, causal-key, or condition-directory mismatch.

- [ ] **Step 9: Commit the matched runner**

~~~bash
git add lib/test/evaluation/spectral_s0.py \
  lib/test/evaluation/spectral_corruption.py \
  tracking/record_spectral_schedule.py tracking/run_spectral_s0.py \
  tests/test_spectral_s0_evaluation.py \
  knowledge_base/Target-Spectral-S0-实验记录.md
git commit -m "feat: add matched frozen stage zero runner"
~~~

### Task 11: Preregistered Statistics, Recovery Estimand, and Gate Analysis

**Files:**

- Create: `lib/test/evaluation/spectral_statistics.py`
- Create: `lib/test/evaluation/benchmark_metrics.py`
- Create: `tracking/analyze_spectral_s0.py`
- Create: `tracking/evaluate_benchmark_metrics.py`
- Create: `tools/freeze_spectral_s0_registry.py`
- Create: `tools/validate_benchmark_evaluators.py`
- Modify: `tests/test_spectral_s0_evaluation.py`
- Modify: `tests/test_spectral_config_registry.py`

**Interfaces:**

- `paired_hierarchical_bootstrap()` resamples base-checkpoint seed, then sequence separately inside every benchmark, with identical indices for method/control.
- `recovery_endpoints()` uses the fixed treatment-independent burst risk set below.
- `evaluate_s0_gate()` implements exact effect, noninferiority, and shuffle-attribution gates.
- `select_admission()`, `select_common_rank()`, `select_coefficient_step()`, and `select_corruption_severity()` implement the preregistered calibration-only CLI modes later invoked in Task 12.
- Registry freeze resolves and records concrete benchmark-evaluator provenance, selected rank/threshold/coefficients, manifests, checkpoints, code hash, hardware, and freeze time.

- **Statistical-test implementation:** add these independently reviewable groups:

- [ ] **Step 1a: Add raw-unit conversion and admission-tie fixtures**
- [ ] **Step 1b: Add recovery, censoring, and treatment-independent risk fixtures**
- [ ] **Step 1c: Add deterministic seed-slot nonlinear aggregation fixture**
- [ ] **Step 1d: Add strict noninferiority, metric-schema, and shuffle fixtures**

Add executable fixtures rather than a checklist:

~~~python
def recompute_seed_slot_contrast(
    *, cells, sampled_seed_slots, benchmark, sampled_indices, aggregate,
):
    sampled_seed_slots = tuple(sampled_seed_slots)
    sampled_indices = tuple(sampled_indices)
    if not sampled_seed_slots:
        raise ValueError("at least one sampled seed slot is required")
    if len(sampled_seed_slots) != len(sampled_indices):
        raise ValueError("each sampled seed slot needs its own sequence indices")
    slot_deltas = []
    for seed, indices in zip(sampled_seed_slots, sampled_indices):
        cell = cells[(seed, benchmark)]
        method = np.asarray(cell["method"], dtype=np.float64)
        control = np.asarray(cell["control"], dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        if method.shape != control.shape or method.ndim != 1:
            raise ValueError("paired method/control cells must be equal vectors")
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("sampled sequence indices must be nonempty vectors")
        if (indices < 0).any() or (indices >= method.size).any():
            raise IndexError("sampled sequence index is out of range")
        method_value = float(aggregate(method[indices]))
        control_value = float(aggregate(control[indices]))
        if not np.isfinite(method_value) or not np.isfinite(control_value):
            raise ValueError("registered aggregate returned a nonfinite value")
        slot_deltas.append(method_value - control_value)
    return float(np.mean(np.asarray(slot_deltas, dtype=np.float64)))

class SpectralStatisticsTests(unittest.TestCase):
    def test_raw_contrast_is_converted_to_pp_once(self):
        self.assertAlmostEqual(
            float(raw_contrast_to_percentage_points(0.503 - 0.500)), 0.3
        )
        self.assertAlmostEqual(
            float(raw_contrast_to_percentage_points(0.10)), 10.0
        )
        with self.assertRaisesRegex(ValueError, "raw benchmark contrast"):
            raw_contrast_to_percentage_points(10.0)

    def test_admission_quantile_includes_later_paired_invalid_rows_and_ties(self):
        q = np.asarray([0.10, 0.20, 0.20, 0.90], dtype=np.float64)
        paired_valid = np.asarray([True, True, True, False])
        threshold, scheduled = threshold_and_schedule(q, paired_valid, coverage=0.50)
        self.assertEqual(threshold, np.quantile(q, 0.50, method="higher"))
        self.assertEqual(threshold, 0.20)
        np.testing.assert_array_equal(scheduled, [False, True, True, False])

    def test_recovery_event_is_fifth_qualifying_calendar_frame(self):
        observation = burst_aligned_time_to_stable_success(
            iou=[0.6, 0.7, 0.8, 0.9, 1.0],
            valid=[True, True, True, True, True],
            horizon=100,
        )
        self.assertEqual((observation.time, observation.event), (5, True))

    def test_invalid_frame_advances_time_and_breaks_recovery_run(self):
        observation = burst_aligned_time_to_stable_success(
            iou=[0.6, 0.7, 0.8, 0.9, 0.0, 0.6, 0.7, 0.8, 0.9, 1.0],
            valid=[True, True, True, True, False, True, True, True, True, True],
            horizon=100,
        )
        self.assertEqual((observation.time, observation.event), (10, True))

    def test_no_event_is_right_censored_not_dropped(self):
        observation = burst_aligned_time_to_stable_success(
            iou=[0.2] * 8, valid=[True] * 8, horizon=6
        )
        self.assertEqual(observation, RecoveryObservation(time=6, event=False, horizon=6))

    def test_noninferiority_is_strict(self):
        self.assertFalse(clean_noninferiority_passes(-0.3))
        self.assertTrue(clean_noninferiority_passes(np.nextafter(-0.3, np.inf)))

    def test_depthtrack_threshold_schema_is_separate_from_unit_metrics(self):
        validate_metric_payload({
            "f_score_sequence": 0.75,
            "precision_sequence": 0.80,
            "recall_sequence": 0.70,
            "threshold_sequence": {"kind": "pos_inf"},
            "metric_scale": "unit_interval",
        })
        with self.assertRaisesRegex(ValueError, "unit_interval"):
            validate_metric_payload({"f_score_sequence": 75.0, "metric_scale": "pp"})

    def test_seed_slot_contrast_precedes_nonlinear_pooling(self):
        cells = {
            (0, "depthtrack"): {
                "method": np.array([0.90, 0.00], dtype=np.float64),
                "control": np.array([0.00, 0.00], dtype=np.float64),
            },
            (1, "depthtrack"): {
                "method": np.array([0.20, 0.20], dtype=np.float64),
                "control": np.array([0.10, 0.10], dtype=np.float64),
            },
        }
        slot_local = recompute_seed_slot_contrast(
            cells=cells,
            sampled_seed_slots=(0, 1),
            benchmark="depthtrack",
            sampled_indices=((0, 1), (0, 1)),
            aggregate=lambda values: float(np.max(values)),
        )
        self.assertAlmostEqual(slot_local, 0.50, places=12)
        pooled_method = np.concatenate([
            cells[(0, "depthtrack")]["method"],
            cells[(1, "depthtrack")]["method"],
        ])
        pooled_control = np.concatenate([
            cells[(0, "depthtrack")]["control"],
            cells[(1, "depthtrack")]["control"],
        ])
        pooled_wrong = float(np.max(pooled_method) - np.max(pooled_control))
        self.assertAlmostEqual(pooled_wrong, 0.80, places=12)
        self.assertNotEqual(slot_local, pooled_wrong)

    def test_shared_risk_set_is_prediction_independent(self):
        eligible = recovery_risk_eligible(
            sequence_length=140, burst_start=40, burst_length=20,
            evaluator_valid_after_burst=[True] * 5,
        )
        self.assertTrue(eligible)
        self.assertFalse(recovery_risk_eligible(
            sequence_length=63, burst_start=40, burst_length=20,
            evaluator_valid_after_burst=[True, True, True],
        ))

    def test_never_failed_method_can_recover_at_time_five(self):
        endpoint = recovery_endpoints(
            iou=[0.8] * 5, valid=[True] * 5, horizon=100
        )
        self.assertFalse(endpoint.failure_observed)
        self.assertEqual(endpoint.stable_success.time, 5)

    def test_relative_rmst_reduction_has_improvement_positive_sign(self):
        full = [RecoveryObservation(5, True, 20)]
        control = [RecoveryObservation(10, True, 20)]
        self.assertGreater(relative_rmst_reduction(full, control), 0.0)

    def test_both_endpoints_share_one_twenty_contrast_attribution_family(self):
        gains = {}
        for endpoint in ("j_core", "relative_rmst_reduction"):
            gains[endpoint] = {
                method: np.full(100, 0.4)
                for method in ATTRIBUTION_ENDPOINT_ROWS
            }
            gains[endpoint]["full_four_spectrum"] = np.ones(100)
        attribution = {
            "crossed_bootstrap_plan_sha256": "a" * 64,
            "gains": gains,
        }
        passed, selected, all_bounds = joint_endpoint_attribution_gate(
            attribution, "j_core", "a" * 64
        )
        self.assertEqual(len(all_bounds), 20)
        self.assertEqual({name: len(value) for name, value in selected.items()}, {
            "geometry": 3, "loo": 4, "shuffle": 3,
        })
        self.assertTrue(all(passed.values()))
        gains["j_core"]["temporal_order_shuffle"] = np.full(100, 0.6)
        passed, _, _ = joint_endpoint_attribution_gate(
            attribution, "j_core", "a" * 64
        )
        self.assertFalse(passed["shuffle"])

    def test_co_primary_selection_is_internal_and_jcore_has_priority(self):
        base = {
            "crossed_bootstrap_plan_sha256": "a" * 64,
            "jcore_lcb_pp": 0.31,
            "rmst_relative_reduction_lcb": 0.11,
        }
        self.assertEqual(select_passing_co_primary(base), "j_core")
        base["jcore_lcb_pp"] = 0.29
        self.assertEqual(
            select_passing_co_primary(base), "relative_rmst_reduction"
        )
        base["rmst_relative_reduction_lcb"] = 0.09
        self.assertIsNone(select_passing_co_primary(base))
~~~

Add the target-absent calendar-break case to `test_invalid_frame_advances_time_and_breaks_recovery_run`; do not create any frame-level bootstrap fixture. The implementation steps below define every helper invoked here.

- [ ] **Step 2: Implement the exact recovery estimand**

Use these registry semantics:

- There is exactly one scheduled burst per corruption-eligible sequence. The shared risk set requires the registered burst to fit and at least five evaluator-valid post-burst frames inside the horizon; it never depends on a compared method's predictions or failure.
- Time origin is the first frame after the registered corruption burst ends.
- Horizon is 100 calendar frames or sequence end, whichever occurs first.
- Recovery event is the fifth consecutive evaluable frame with IoU at least 0.5; event time is that fifth qualifying frame.
- Target-absent/invalid frames advance calendar time, do not qualify, and break the consecutive run.
- A method that never met the five-frame failure definition remains in the shared risk set, is flagged `failure_observed=false`, and may achieve stable-success event time 5.
- A method already above threshold at time origin still requires five consecutive qualifying frames, so its earliest event time is 5.
- No event by the horizon is right-censored; it is never dropped from RMST.
- Also report failure count separately: five consecutive IoU values below 0.1.

Implement the endpoint without conditioning on failure:

~~~python
@dataclass(frozen=True)
class RecoveryObservation:
    time: int
    event: bool
    horizon: int

@dataclass(frozen=True)
class RecoveryEndpoints:
    stable_success: RecoveryObservation
    failure_observed: bool

def recovery_risk_eligible(
    sequence_length, burst_start, burst_length, evaluator_valid_after_burst,
    horizon=100,
):
    origin = burst_start + burst_length
    if origin >= sequence_length:
        return False
    available = min(horizon, sequence_length - origin)
    valid = tuple(bool(value) for value in evaluator_valid_after_burst[:available])
    return available >= 5 and sum(valid) >= 5

def burst_aligned_time_to_stable_success(iou, valid, horizon):
    run = 0
    for time, (value, is_valid) in enumerate(zip(iou[:horizon], valid[:horizon]), 1):
        run = run + 1 if is_valid and value >= 0.50 else 0
        if run == 5:
            return RecoveryObservation(time=time, event=True, horizon=horizon)
    return RecoveryObservation(time=min(len(iou), horizon), event=False, horizon=horizon)

def recovery_endpoints(iou, valid, horizon):
    low_run = 0
    failure_observed = False
    for value, is_valid in zip(iou[:horizon], valid[:horizon]):
        low_run = low_run + 1 if is_valid and value < 0.10 else 0
        failure_observed = failure_observed or low_run >= 5
    return RecoveryEndpoints(
        stable_success=burst_aligned_time_to_stable_success(iou, valid, horizon),
        failure_observed=failure_observed,
    )

def rmst(observations):
    observations = tuple(observations)
    if not observations:
        raise ValueError("RMST requires at least one recovery observation")
    tau = max(item.horizon for item in observations)
    survival = 1.0
    area = 0.0
    for time in range(1, tau + 1):
        at_risk = sum(item.time >= time and item.horizon >= time for item in observations)
        if at_risk == 0:
            break
        area += survival
        events = sum(item.event and item.time == time for item in observations)
        survival *= 1.0 - events / at_risk
    return area

def relative_rmst_reduction(full, confidence):
    return (rmst(confidence) - rmst(full)) / max(rmst(confidence), 1e-12)
~~~

Within every bootstrap replicate and benchmark, draw the sequence IDs once and reuse that exact draw across all sampled seed-slots, then compute method/control RMST independently inside each slot and equal-weight the three slot values. Equal-weight those benchmark values as `RMST_core = 0.5 * (RMST_LasHeR + RMST_DepthTrack)`, then apply the displayed relative-reduction formula to the two methods' `RMST_core` values. A duplicated sampled seed occupies a distinct slot but shares the benchmark's crossed sequence draw. Never pool sequences across seed-slots or let the benchmark with more eligible sequences dominate.

- [ ] **Step 3: Implement paired hierarchical bootstrap**

~~~python
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

import numpy as np

@dataclass(frozen=True)
class CrossedBootstrapPlan:
    seed_slots: np.ndarray
    sequence_indices: Mapping[str, np.ndarray]
    sequence_ids: Mapping[str, tuple[str, ...]]
    bootstrap_seed: int
    frozen_u_sha256: str
    checkpoint_index_sha256: str
    inference_scope: str = "conditional_on_frozen_shared_u_and_checkpoint_set_0_1_2"

BOOTSTRAP_POOL_CONDITIONS = {
    "clean_metric": "clean",
    "corruption_metric": "registered_corruption",
    "recovery": "registered_corruption",
}

def bootstrap_pool_key(pool, benchmark):
    if pool not in BOOTSTRAP_POOL_CONDITIONS or benchmark not in {"lasher", "depthtrack"}:
        raise ValueError("unknown bootstrap pool/benchmark")
    return f"{pool}|{benchmark}"

def validate_crossed_cells(rows, methods, pool):
    expected_condition = BOOTSTRAP_POOL_CONDITIONS[pool]
    indexed = {}
    for row in rows:
        if row["condition"] != expected_condition:
            raise ValueError(f"wrong condition in {pool}: {row['condition']}")
        if pool == "recovery" and type(row["shared_risk_eligible"]) is not bool:
            raise TypeError("shared_risk_eligible must be an exact bool")
        if pool == "recovery" and not row["shared_risk_eligible"]:
            raise ValueError("ineligible sequence leaked into recovery pool")
        key = (
            int(row["base_seed"]), str(row["benchmark"]),
            str(row["method"]), str(row["sequence_id"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate crossed cell: {key}")
        indexed[key] = row
    ids_by_benchmark = {}
    for benchmark in ("lasher", "depthtrack"):
        reference = None
        for seed in (0, 1, 2):
            for method in methods:
                ids = {
                    key[3] for key in indexed
                    if key[:3] == (seed, benchmark, method)
                }
                if not ids:
                    raise ValueError(f"empty crossed cell: {(seed, benchmark, method)}")
                if reference is None:
                    reference = ids
                elif ids != reference:
                    raise ValueError("sequence support differs across seed/method")
        ids_by_benchmark[benchmark] = tuple(sorted(reference))
    return indexed, ids_by_benchmark

def make_crossed_bootstrap_plan(
    sequence_ids_by_pool, replicates, seed,
    frozen_u_sha256, checkpoint_index_sha256,
):
    if int(replicates) != replicates or int(replicates) < 1:
        raise ValueError("bootstrap replicate count must be a positive integer")
    replicates = int(replicates)
    required_pool_keys = {
        bootstrap_pool_key(pool, benchmark)
        for pool in BOOTSTRAP_POOL_CONDITIONS
        for benchmark in ("lasher", "depthtrack")
    }
    if set(sequence_ids_by_pool) != required_pool_keys:
        raise ValueError("bootstrap plan requires all six pool/benchmark supports")
    ids_by_pool = {}
    for pool_key, values in sorted(sequence_ids_by_pool.items()):
        ids = tuple(str(value) for value in values)
        if not ids or ids != tuple(sorted(set(ids))):
            raise ValueError(f"bootstrap support must be sorted/unique: {pool_key}")
        ids_by_pool[pool_key] = ids
    rng = np.random.default_rng(seed)
    seed_slots = rng.choice(
        np.asarray((0, 1, 2), dtype=np.int64),
        size=(replicates, 3), replace=True,
    )
    sequence_indices = {
        pool_key: rng.integers(
            0, len(ids), size=(replicates, len(ids)), endpoint=False,
        )
        for pool_key, ids in sorted(ids_by_pool.items())
    }
    return CrossedBootstrapPlan(
        seed_slots=seed_slots,
        sequence_indices=sequence_indices,
        sequence_ids=ids_by_pool,
        bootstrap_seed=int(seed),
        frozen_u_sha256=frozen_u_sha256,
        checkpoint_index_sha256=checkpoint_index_sha256,
    )

def canonical_crossed_plan_bytes(plan):
    artifact = {
        "schema_version": 1,
        "bootstrap_seed": int(plan.bootstrap_seed),
        "frozen_u_sha256": plan.frozen_u_sha256,
        "checkpoint_index_sha256": plan.checkpoint_index_sha256,
        "inference_scope": plan.inference_scope,
        "seed_slots": plan.seed_slots.tolist(),
        "sequence_ids": {
            key: list(value) for key, value in sorted(plan.sequence_ids.items())
        },
        "sequence_indices": {
            key: value.tolist()
            for key, value in sorted(plan.sequence_indices.items())
        },
    }
    return json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

def crossed_plan_sha256(plan):
    return hashlib.sha256(canonical_crossed_plan_bytes(plan)).hexdigest()

def apply_crossed_paired_plan(rows, plan, method, control, pool):
    indexed, ids_by_benchmark = validate_crossed_cells(
        rows, (method, control), pool
    )
    replicates = plan.seed_slots.shape[0]
    output = {
        benchmark: np.empty(replicates, dtype=np.float64)
        for benchmark in ("lasher", "depthtrack")
    }
    for replicate in range(replicates):
        for benchmark in ("lasher", "depthtrack"):
            pool_key = bootstrap_pool_key(pool, benchmark)
            ids = plan.sequence_ids[pool_key]
            if tuple(ids_by_benchmark[benchmark]) != tuple(ids):
                raise ValueError("bootstrap-plan sequence support mismatch")
            sampled_ids = tuple(
                ids[index]
                for index in plan.sequence_indices[pool_key][replicate]
            )
            slot_deltas = []
            for sampled_seed in plan.seed_slots[replicate]:
                seed = int(sampled_seed)
                method_rows = [indexed[(seed, benchmark, method, item)] for item in sampled_ids]
                control_rows = [indexed[(seed, benchmark, control, item)] for item in sampled_ids]
                slot_deltas.append(
                    recompute_registered_benchmark_aggregate(method_rows)
                    - recompute_registered_benchmark_aggregate(control_rows)
                )
            output[benchmark][replicate] = float(np.mean(slot_deltas))
    output["j_core"] = 0.5 * (output["lasher"] + output["depthtrack"])
    return output

def recovery_observation_from_row(row):
    value = row["stable_success"]
    if type(value["event"]) is not bool:
        raise TypeError("recovery event must be an exact bool")
    time_value, horizon = int(value["time"]), int(value["horizon"])
    if not 1 <= time_value <= horizon:
        raise ValueError("invalid recovery time/horizon")
    return RecoveryObservation(
        time=time_value, event=value["event"], horizon=horizon,
    )

def apply_crossed_rmst_plan(rows, plan, method, control):
    pool = "recovery"
    indexed, ids_by_benchmark = validate_crossed_cells(
        rows, (method, control), pool
    )
    replicates = plan.seed_slots.shape[0]
    method_core = np.empty(replicates, dtype=np.float64)
    control_core = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        method_benchmarks, control_benchmarks = [], []
        for benchmark in ("lasher", "depthtrack"):
            pool_key = bootstrap_pool_key(pool, benchmark)
            ids = plan.sequence_ids[pool_key]
            if tuple(ids_by_benchmark[benchmark]) != tuple(ids):
                raise ValueError("bootstrap-plan sequence support mismatch")
            sampled_ids = tuple(
                ids[index]
                for index in plan.sequence_indices[pool_key][replicate]
            )
            method_slots, control_slots = [], []
            for sampled_seed in plan.seed_slots[replicate]:
                seed = int(sampled_seed)
                method_slots.append(rmst(
                    recovery_observation_from_row(
                        indexed[(seed, benchmark, method, sequence_id)]
                    )
                    for sequence_id in sampled_ids
                ))
                control_slots.append(rmst(
                    recovery_observation_from_row(
                        indexed[(seed, benchmark, control, sequence_id)]
                    )
                    for sequence_id in sampled_ids
                ))
            method_benchmarks.append(float(np.mean(method_slots)))
            control_benchmarks.append(float(np.mean(control_slots)))
        method_core[replicate] = float(np.mean(method_benchmarks))
        control_core[replicate] = float(np.mean(control_benchmarks))
    if (control_core <= 0.0).any() or not (
        np.isfinite(method_core).all() and np.isfinite(control_core).all()
    ):
        raise ValueError("invalid crossed RMST bootstrap")
    return (control_core - method_core) / control_core
~~~

Build one `CrossedBootstrapPlan` and reuse that exact object unchanged for every effect, noninferiority, geometry, LOO, and shuffle contrast. The object contains three preregistered pools per benchmark: full clean metric support, full registered-corruption metric support, and the smaller treatment-independent recovery-risk support. Build those supports from the committed manifests/evaluator-validity masks before reading any method outcome; do not force the recovery subset onto `J_core` or clean noninferiority. Within each pool and benchmark, the same sequence draw is shared across every sampled seed-slot and method; the one seed-slot array is shared across all pools and both benchmarks. Write `canonical_crossed_plan_bytes(plan)` byte-for-byte to `output/spectral_s0/gate/audits/crossed_bootstrap_plan.json`, and record its file SHA-256 in both final artifacts. A test rebuilds it from the same manifest/registry and requires byte identity; every contrast writer records the one plan hash and rejects a second plan. Recompute the complete registered aggregate separately inside every sampled seed-slot, form the within-slot contrast, then equal-weight seed-slot contrasts before equal-weighting benchmarks. For recovery, compute method and comparator RMST separately inside each sampled seed-slot, then equal-weight slots and benchmarks before forming the relative reduction; never average already-formed per-seed ratios. DepthTrack threshold maximization remains inside each sampled slot. State the inference scope exactly as conditional on the selected shared `u` and checkpoint set `{0,1,2}`; do not generalize the intervals to arbitrary retraining seeds or a refitted `u`.

- [ ] **Step 4: Implement exact S0 decision**

Adapters, per-sequence summaries, and `recompute_registered_benchmark_aggregate()` emit only raw unit-interval values. The one and only unit conversion lives at the gate-analysis boundary:

~~~python
def raw_contrast_to_percentage_points(delta_raw):
    delta_raw = np.asarray(delta_raw, dtype=np.float64)
    if not np.isfinite(delta_raw).all() or (np.abs(delta_raw) > 1.0).any():
        raise ValueError("raw benchmark contrast must lie in [-1,1]")
    return 100.0 * delta_raw

def bootstrap_contrasts_to_percentage_points(bootstrap_raw):
    return raw_contrast_to_percentage_points(bootstrap_raw)

def threshold_and_schedule(q_values, paired_valid, coverage):
    q_values = np.asarray(q_values, dtype=np.float64)
    paired_valid = np.asarray(paired_valid, dtype=np.bool_)
    if q_values.ndim != 1 or paired_valid.shape != q_values.shape:
        raise ValueError("admission arrays must be aligned one-dimensional vectors")
    if not np.isfinite(q_values).all() or not 0.0 <= coverage <= 1.0:
        raise ValueError("invalid admission quantile input")
    threshold = float(np.quantile(q_values, 1.0 - coverage, method="higher"))
    return threshold, (q_values >= threshold) & paired_valid

def clean_noninferiority_passes(lower_bound_pp):
    return float(lower_bound_pp) > -0.3

@dataclass(frozen=True, order=True)
class CleanNIKey:
    benchmark: str
    comparator: str

    @property
    def artifact_key(self):
        return f"{self.benchmark}|{self.comparator}"

REQUIRED_CLEAN_NI_KEYS = frozenset({
    CleanNIKey("lasher", "confidence_only_scalar_history"),
    CleanNIKey("depthtrack", "confidence_only_scalar_history"),
    CleanNIKey("lasher", "routing_disabled_legacy"),
    CleanNIKey("depthtrack", "routing_disabled_legacy"),
})
REQUIRED_CLEAN_NI_ARTIFACT_KEYS = frozenset(
    key.artifact_key for key in REQUIRED_CLEAN_NI_KEYS
)

def evaluate_keyed_clean_noninferiority(lcbs_pp, margin_pp=-0.3):
    if (
        not isinstance(lcbs_pp, Mapping)
        or set(lcbs_pp) != REQUIRED_CLEAN_NI_ARTIFACT_KEYS
    ):
        raise ValueError("clean NI benchmark/comparator keys mismatch")
    passed = {}
    for key in sorted(REQUIRED_CLEAN_NI_KEYS):
        bound = float(lcbs_pp[key.artifact_key])
        if not math.isfinite(bound):
            raise ValueError(f"nonfinite clean NI bound: {key}")
        passed[key.artifact_key] = bound > float(margin_pp)
    return passed

def simultaneous_lower_bounds(samples_by_name, familywise_alpha=0.025):
    if not samples_by_name:
        raise ValueError("simultaneous bounds require at least one contrast")
    if not math.isfinite(float(familywise_alpha)) or not 0.0 < familywise_alpha < 1.0:
        raise ValueError("invalid familywise alpha")
    count = len(samples_by_name)
    output = {}
    expected_shape = None
    for name, samples in samples_by_name.items():
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"invalid bootstrap contrast: {name}")
        expected_shape = values.shape if expected_shape is None else expected_shape
        if values.shape != expected_shape:
            raise ValueError("aligned contrasts must share replicate count")
        output[name] = float(np.quantile(
            values, familywise_alpha / count, method="linear"
        ))
    return output

GEOMETRY_CONTROLS = (
    "random_orthogonal", "pooled_same", "target_balanced_identity",
)
STRENGTH_MATCHED_LOO_ROWS = {
    "identity": "full_minus_identity_strength_matched",
    "dynamic": "full_minus_dynamic_strength_matched",
    "private": "full_minus_private_strength_matched",
    "background": "full_minus_background_strength_matched",
}
SHUFFLE_ROWS = (
    "rgbx_pair_shuffle", "temporal_order_shuffle",
    "target_background_mask_shuffle",
)

ATTRIBUTION_ENDPOINT_ROWS = (
    "full_four_spectrum", *GEOMETRY_CONTROLS,
    *STRENGTH_MATCHED_LOO_ROWS.values(), *SHUFFLE_ROWS,
)

def compute_co_primary_lcbs(corruption_metric_rows, recovery_rows, plan):
    jcore_raw = apply_crossed_paired_plan(
        corruption_metric_rows, plan,
        "full_four_spectrum", "confidence_only_scalar_history",
        pool="corruption_metric",
    )["j_core"]
    jcore_pp = bootstrap_contrasts_to_percentage_points(jcore_raw)
    rmst_gain = apply_crossed_rmst_plan(
        recovery_rows, plan,
        "full_four_spectrum", "confidence_only_scalar_history"
    )
    return {
        "crossed_bootstrap_plan_sha256": crossed_plan_sha256(plan),
        "jcore_lcb_pp": float(np.quantile(jcore_pp, 0.025 / 2, method="linear")),
        "rmst_relative_reduction_lcb": float(np.quantile(
            rmst_gain, 0.025 / 2, method="linear"
        )),
    }

def compute_clean_noninferiority_lcbs(rows, plan):
    samples = {}
    for comparator in (
        "confidence_only_scalar_history", "routing_disabled_legacy",
    ):
        contrasts = apply_crossed_paired_plan(
            rows, plan, "full_four_spectrum", comparator,
            pool="clean_metric",
        )
        for benchmark in ("lasher", "depthtrack"):
            key = CleanNIKey(benchmark, comparator)
            samples[key.artifact_key] = (
                bootstrap_contrasts_to_percentage_points(contrasts[benchmark])
            )
    return {
        "crossed_bootstrap_plan_sha256": crossed_plan_sha256(plan),
        "lcbs_pp": simultaneous_lower_bounds(samples, familywise_alpha=0.025),
    }

def build_both_endpoint_gains(corruption_metric_rows, recovery_rows, plan):
    gains = {"j_core": {}, "relative_rmst_reduction": {}}
    for method in ATTRIBUTION_ENDPOINT_ROWS:
        raw = apply_crossed_paired_plan(
            corruption_metric_rows, plan,
            method, "confidence_only_scalar_history",
            pool="corruption_metric",
        )["j_core"]
        gains["j_core"][method] = bootstrap_contrasts_to_percentage_points(raw)
        gains["relative_rmst_reduction"][method] = apply_crossed_rmst_plan(
            recovery_rows, plan,
            method, "confidence_only_scalar_history",
        )
    return {
        "crossed_bootstrap_plan_sha256": crossed_plan_sha256(plan),
        "gains": gains,
    }

def select_passing_co_primary(co_primary):
    required = {
        "crossed_bootstrap_plan_sha256", "jcore_lcb_pp",
        "rmst_relative_reduction_lcb",
    }
    if set(co_primary) != required:
        raise ValueError("co-primary result schema mismatch")
    jcore = float(co_primary["jcore_lcb_pp"])
    recovery = float(co_primary["rmst_relative_reduction_lcb"])
    if not math.isfinite(jcore) or not math.isfinite(recovery):
        raise ValueError("co-primary bounds must be finite")
    if jcore >= 0.3:
        return "j_core"
    if recovery >= 0.10:
        return "relative_rmst_reduction"
    return None

def joint_endpoint_attribution_gate(attribution, selected_endpoint, plan_sha256):
    if set(attribution) != {"crossed_bootstrap_plan_sha256", "gains"}:
        raise ValueError("attribution result schema mismatch")
    if attribution["crossed_bootstrap_plan_sha256"] != plan_sha256:
        raise ValueError("attribution used a different crossed bootstrap plan")
    gains_by_endpoint = attribution["gains"]
    endpoints = ("j_core", "relative_rmst_reduction")
    if set(gains_by_endpoint) != set(endpoints):
        raise ValueError("both attribution endpoints are mandatory")
    samples = {}
    for endpoint in endpoints:
        gains = gains_by_endpoint[endpoint]
        if set(gains) != set(ATTRIBUTION_ENDPOINT_ROWS):
            raise ValueError(f"attribution method keys mismatch: {endpoint}")
        full = np.asarray(gains["full_four_spectrum"], dtype=np.float64)
        for control in GEOMETRY_CONTROLS:
            samples[f"{endpoint}|geometry|{control}"] = (
                full - np.asarray(gains[control], dtype=np.float64)
            )
        for family, row in STRENGTH_MATCHED_LOO_ROWS.items():
            samples[f"{endpoint}|loo|{family}"] = (
                full - np.asarray(gains[row], dtype=np.float64)
            )
        for row in SHUFFLE_ROWS:
            samples[f"{endpoint}|shuffle|{row}"] = (
                full - 2.0 * np.asarray(gains[row], dtype=np.float64)
            )
    if len(samples) != 20:
        raise AssertionError("joint endpoint-attribution family must contain 20 contrasts")
    bounds = simultaneous_lower_bounds(samples, familywise_alpha=0.025)
    selected = {
        group: {
            name: value for name, value in bounds.items()
            if name.startswith(f"{selected_endpoint}|{group}|")
        }
        for group in ("geometry", "loo", "shuffle")
    }
    expected_counts = {"geometry": 3, "loo": 4, "shuffle": 3}
    if any(len(selected[group]) != count for group, count in expected_counts.items()):
        raise ValueError("selected endpoint attribution cells are incomplete")
    passes = {
        group: all(value > 0.0 for value in values.values())
        for group, values in selected.items()
    }
    return passes, selected, bounds

UNIT_INTERVAL_FIELDS = frozenset({
    "success_auc", "f_score_sequence", "precision_sequence", "recall_sequence",
    "f_score_frame", "precision_frame", "recall_frame",
})

def validate_metric_payload(payload):
    if payload.get("metric_scale") != "unit_interval":
        raise ValueError("benchmark metrics must use unit_interval scale")
    for field in UNIT_INTERVAL_FIELDS.intersection(payload):
        value = float(payload[field])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} is outside unit_interval")
    for field in ("threshold_sequence", "threshold_frame"):
        if field not in payload:
            continue
        threshold = payload[field]
        if threshold.get("kind") not in {"finite", "pos_inf", "neg_inf"}:
            raise ValueError(f"invalid {field} sentinel")
        if threshold["kind"] == "finite":
            value = threshold.get("value")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid finite {field}")

def select_admission(rows, coverage_candidates=(0.20, 0.30, 0.40, 0.50)):
    ordered = sorted(
        rows,
        key=lambda row: (
            row["benchmark"], int(row["base_seed"]),
            row["sequence"], int(row["frame_index"]),
        ),
    )
    q_values = np.asarray([row["q_memory"] for row in ordered], dtype=np.float64)
    paired_valid = np.asarray([row["paired_valid"] for row in ordered], dtype=np.bool_)
    strata = sorted({(int(row["base_seed"]), row["benchmark"]) for row in ordered})
    if len(strata) != 6 or not np.isfinite(q_values).all():
        raise ValueError("admission selection requires six finite calibration strata")
    survivors = []
    for coverage in coverage_candidates:
        threshold, scheduled = threshold_and_schedule(q_values, paired_valid, coverage)
        overall = float(scheduled.mean())
        coverage_by_stratum = {}
        for stratum in strata:
            mask = np.asarray([
                (int(row["base_seed"]), row["benchmark"]) == stratum
                for row in ordered
            ])
            stratum_key = f"seed{stratum[0]}|{stratum[1]}"
            coverage_by_stratum[stratum_key] = float(scheduled[mask].mean())
        if overall < 0.20 or min(coverage_by_stratum.values()) < 0.20:
            continue
        gt_valid_scheduled_by_stratum = {}
        gt_valid_total_by_stratum = {}
        for stratum in strata:
            stratum_mask = np.asarray([
                (int(row["base_seed"]), row["benchmark"]) == stratum
                for row in ordered
            ], dtype=bool)
            gt_valid = np.asarray(
                [bool(row["gt_valid"]) for row in ordered], dtype=bool
            ) & stratum_mask
            stratum_key = f"seed{stratum[0]}|{stratum[1]}"
            gt_valid_total_by_stratum[stratum_key] = int(gt_valid.sum())
            gt_valid_scheduled_by_stratum[stratum_key] = int(
                (gt_valid & scheduled).sum()
            )
        if min(gt_valid_scheduled_by_stratum.values()) == 0:
            continue
        gt_scheduled = np.asarray([
            bool(row["gt_valid"]) and bool(admit)
            for row, admit in zip(ordered, scheduled)
        ])
        if not gt_scheduled.any():
            raise ValueError("candidate has no GT-valid scheduled frame")
        false_count = sum(
            bool(mask) and float(row["evaluator_iou"]) < 0.50
            for row, mask in zip(ordered, gt_scheduled)
        )
        false_rate = false_count / int(gt_scheduled.sum())
        survivors.append({
            "coverage_candidate": float(coverage),
            "q_memory_threshold": threshold,
            "scheduled": scheduled,
            "realized_coverage": overall,
            "coverage_by_stratum": coverage_by_stratum,
            "gt_valid_total_by_stratum": gt_valid_total_by_stratum,
            "gt_valid_scheduled_by_stratum": gt_valid_scheduled_by_stratum,
            "false_admission_rate": false_rate,
        })
    if not survivors:
        raise ValueError("no admission candidate reaches registered coverage")
    return min(
        survivors,
        key=lambda row: (
            row["false_admission_rate"], -row["realized_coverage"],
            row["q_memory_threshold"], row["coverage_candidate"],
        ),
    )

def select_coefficient_step(candidates):
    required = tuple(range(0, 1001, 100))
    by_step = {int(row["successful_step"]): row for row in candidates}
    if tuple(sorted(by_step)) != required or len(by_step) != len(candidates):
        raise ValueError("coefficient candidates must be the eleven registered successful steps")
    if any(not math.isfinite(float(row["j_core_raw"])) for row in candidates):
        raise ValueError("coefficient J_core candidates must be finite")
    return max(by_step.values(), key=lambda row: (
        float(row["j_core_raw"]), -int(row["successful_step"])
    ))

def select_corruption_severity(rows):
    by_severity = {float(row["severity"]): row for row in rows}
    if set(by_severity) != {0.50, 0.75, 1.00} or len(by_severity) != len(rows):
        raise ValueError("corruption rows must contain the three registered severities")
    scored = []
    for severity, row in by_severity.items():
        clean = float(row["j_core_clean_raw"])
        corrupt = float(row["j_core_corrupt_raw"])
        if (
            not math.isfinite(clean) or not math.isfinite(corrupt)
            or not 0.0 <= clean <= 1.0 or not 0.0 <= corrupt <= 1.0
        ):
            raise ValueError("corruption selector requires raw unit-interval metrics")
        drop_raw = clean - corrupt
        drop_pp = float(raw_contrast_to_percentage_points(drop_raw))
        scored.append((abs(drop_pp - 10.0), severity, drop_raw, drop_pp))
    _, severity, drop_raw, drop_pp = min(scored, key=lambda item: (item[0], item[1]))
    return {"selected_severity": severity, "drop_raw": drop_raw, "drop_pp": drop_pp}

def evaluate_s0_gate(
    co_primary, clean_noninferiority, attribution, recomputed_audit_pass,
):
    if type(recomputed_audit_pass) is not bool:
        raise TypeError("recomputed audit result must be an exact bool")
    selected_endpoint = select_passing_co_primary(co_primary)
    plan_sha256 = co_primary["crossed_bootstrap_plan_sha256"]
    if (
        set(clean_noninferiority) != {"crossed_bootstrap_plan_sha256", "lcbs_pp"}
        or clean_noninferiority["crossed_bootstrap_plan_sha256"] != plan_sha256
    ):
        raise ValueError("clean noninferiority used a different bootstrap plan")
    noninferiority = evaluate_keyed_clean_noninferiority(
        clean_noninferiority["lcbs_pp"]
    )
    if (
        not isinstance(attribution, Mapping)
        or attribution.get("crossed_bootstrap_plan_sha256") != plan_sha256
    ):
        raise ValueError("attribution used a different bootstrap plan")
    if selected_endpoint is None:
        attribution_passes = {"geometry": False, "loo": False, "shuffle": False}
        selected_lcbs, all_attribution_lcbs = {}, {}
    else:
        attribution_passes, selected_lcbs, all_attribution_lcbs = (
            joint_endpoint_attribution_gate(
                attribution, selected_endpoint, plan_sha256
            )
        )
    checks = {
        "recomputed_engineering_audit": recomputed_audit_pass,
        "co_primary_effect": selected_endpoint is not None,
        "clean_noninferiority": all(noninferiority.values()),
        "geometry_specificity": attribution_passes["geometry"],
        "strength_matched_loo": attribution_passes["loo"],
        "all_three_shuffle_semantics": attribution_passes["shuffle"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "selected_passing_endpoint": selected_endpoint,
        "crossed_bootstrap_plan_sha256": plan_sha256,
        "clean_noninferiority": noninferiority,
        "clean_noninferiority_lcbs_pp": {
            key: float(clean_noninferiority["lcbs_pp"][key])
            for key in sorted(REQUIRED_CLEAN_NI_ARTIFACT_KEYS)
        },
        "selected_endpoint_attribution_lcbs": selected_lcbs,
        "all_twenty_attribution_lcbs": all_attribution_lcbs,
    }
~~~

Registry validation requires `statistics.adapter_metric_scale=unit_interval` and `raw_delta_to_percentage_points=100.0`. Convert each already-formed raw contrast exactly once before taking pp confidence bounds; never multiply adapter output, RMST, or an already-pp field. Evidence stores paired `estimate_raw` and `estimate_pp` with explicit unit tags and rejects a missing/duplicate conversion.

`J_core` below is the equal-benchmark registered-corruption aggregate at the frozen selected severity; the recovery endpoint uses the same registered-corruption episodes and shared risk set. Clean rows are opened only for the separately keyed noninferiority family. Pass disjoint condition-specific row sets into the helpers so `validate_crossed_cells()` rejects accidental clean/corruption pooling. `full_four_spectrum` versus `confidence_only_scalar_history` passes the effect gate if either:

- familywise one-sided 97.5% Bonferroni LCB for `Delta J_core`, using quantile `0.025/2 = 0.0125`, is at least `0.3` percentage points; or
- familywise one-sided 97.5% Bonferroni LCB for relative RMST reduction, also using quantile `0.0125`, is at least `0.10`.

It must simultaneously satisfy for each benchmark:

~~~python
lasher_vs_confidence = lasher_lcb_full_minus_confidence_pp > -0.3
depthtrack_vs_confidence = depthtrack_lcb_full_minus_confidence_pp > -0.3
lasher_vs_legacy = lasher_lcb_full_minus_legacy_pp > -0.3
depthtrack_vs_legacy = depthtrack_lcb_full_minus_legacy_pp > -0.3
~~~

Compute all four clean contrasts from the same crossed paired replicates and serialize them under the exact JSON keys `benchmark|comparator`. Use Bonferroni lower quantile `0.025/4 = 0.00625`; these are simultaneous familywise 97.5% lower bounds. The expert-review A0 gate therefore requires clean noninferiority against both the scalar-confidence-history comparator and matched routing-disabled legacy. `select_passing_co_primary()` is the only endpoint selector: it chooses `J_core` whenever its corrected bound passes, otherwise chooses RMST only if its corrected bound passes, otherwise returns no endpoint. The caller cannot supply or override that choice, and `s0_gate.json` records it.

Always construct attribution gains for both candidate endpoints before selection. Put both endpoints times all ten contrasts into one 20-member family and use the common Bonferroni lower quantile `0.025/20 = 0.00125`. This joint family controls the data-dependent endpoint choice; the unselected endpoint's bounds enter the multiplicity correction but need not pass. On the internally selected passing endpoint, require all of the following:

- `full_four_spectrum` minus each of `random_orthogonal`, `pooled_same`, and `target_balanced_identity` has a strict positive joint-family LCB;
- `full_four_spectrum` minus each of the four mandatory strength-matched LOO gains has a strict positive joint-family LCB;
- each pair/temporal/mask attenuation contrast `G_full - 2*G_shuffle` has a strict positive joint-family LCB.

The co-primary, clean-NI, and attribution bundles must carry the exact same canonical crossed-plan SHA-256, and attribution must expose both exact endpoint keys and all exact method keys. Wrong-plan, missing-endpoint, only-RMST-passes/J-gains, only-J-passes/RMST-gains, and both-pass-but-RMST-selected fixtures fail closed. All three semantic shuffles must pass; the former “two of three plus mean” rule is removed. A failure stops the four-spectrum main line. It may motivate a new design/registry on unopened data, but gate-confirmation results cannot be used to delete a branch and continue.

Implement all four calibration selectors in the same analyzer commit. `select_admission()` follows Task 7's exact float64 population/`1-c`/`method="higher"` rule and emits the six schedules only after choosing the registered false-admission winner. `select_common_rank()` follows Task 2's smallest-common-rank exact-trace/gap rule. `select_coefficient_step()` validates the eleven successful-step checkpoints and maximizes the equal-seed/equal-benchmark raw `J_core`, with earlier-successful-step ties. `select_corruption_severity()` forms raw clean-minus-corrupt `J_core`, invokes the sole pp conversion once, and selects closest to `10.0 pp`, lower-severity ties. Each mode writes its input hashes and refuses gate-confirmation manifests.

- **Benchmark-adapter implementation:** execute these independently reviewable actions:

- [ ] **Step 5a: Add canonical finite/±∞ threshold serialization**
- [ ] **Step 5b: Add validated DepthTrack sequence inputs**
- [ ] **Step 5c: Add sequence-macro and frame-pooled scoring at one threshold**
- [ ] **Step 5d: Add unique-confidence plus ±∞ selection with higher-threshold ties**
- [ ] **Step 5e: Add strict unit-interval JSON and result writers**
- [ ] **Step 5f: Add normalized LasHeR tree hashing and isolated archive extraction**
- [ ] **Step 5g: Add explicit MATLAB version/invocation helpers**
- [ ] **Step 5h: Add the 21-threshold sequence-macro LasHeR AUC parser boundary**
- [ ] **Step 5i: Wire both adapters and run the committed fixtures**

Add this core to `lib/test/evaluation/benchmark_metrics.py`:

~~~python
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np

UNIT_INTERVAL_FIELDS = frozenset({
    "success_auc", "precision_sequence", "recall_sequence",
    "f_score_sequence", "precision_frame", "recall_frame", "f_score_frame",
})

def threshold_to_json(value):
    value = float(value)
    if math.isinf(value):
        return {"kind": "pos_inf" if value > 0 else "neg_inf"}
    if not math.isfinite(value):
        raise ValueError("threshold cannot be NaN")
    return {"kind": "finite", "value": value}

def threshold_from_json(payload):
    if payload == {"kind": "pos_inf"}:
        return math.inf
    if payload == {"kind": "neg_inf"}:
        return -math.inf
    if (
        isinstance(payload, dict)
        and set(payload) == {"kind", "value"}
        and payload["kind"] == "finite"
        and isinstance(payload["value"], (int, float))
        and not isinstance(payload["value"], bool)
        and math.isfinite(float(payload["value"]))
    ):
        return float(payload["value"])
    raise ValueError("invalid canonical threshold")

@dataclass(frozen=True)
class DepthTrackSeries:
    overlap: np.ndarray
    gt_visible: np.ndarray
    confidence: np.ndarray

    def __post_init__(self):
        overlap = np.asarray(self.overlap, dtype=np.float64)
        visible = np.asarray(self.gt_visible, dtype=bool)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        if overlap.ndim != 1 or not (
            overlap.shape == visible.shape == confidence.shape
        ):
            raise ValueError("DepthTrack arrays must be equal-length vectors")
        if overlap.size == 0:
            raise ValueError("DepthTrack sequence cannot be empty")
        if (
            not np.isfinite(overlap).all()
            or not np.isfinite(confidence).all()
            or ((overlap < 0.0) | (overlap > 1.0)).any()
            or ((confidence < 0.0) | (confidence > 1.0)).any()
        ):
            raise ValueError("DepthTrack overlap/confidence must lie in [0,1]")
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(self, "gt_visible", visible)
        object.__setattr__(self, "confidence", confidence)

def _f_score(precision, recall):
    return 2.0 * precision * recall / max(precision + recall, 1e-12)

def _metrics_at_threshold(series, threshold):
    sequence_precision = []
    sequence_recall = []
    pooled_overlap = 0.0
    pooled_pred = 0
    pooled_gt = 0
    for item in series:
        predicted_visible = item.confidence >= threshold
        both_visible = predicted_visible & item.gt_visible
        overlap_sum = float(item.overlap[both_visible].sum())
        n_pred = int(predicted_visible.sum())
        n_gt = int(item.gt_visible.sum())
        sequence_precision.append(overlap_sum / max(n_pred, 1e-12))
        sequence_recall.append(overlap_sum / max(n_gt, 1e-12))
        pooled_overlap += overlap_sum
        pooled_pred += n_pred
        pooled_gt += n_gt
    precision_sequence = float(np.mean(sequence_precision))
    recall_sequence = float(np.mean(sequence_recall))
    precision_frame = pooled_overlap / max(pooled_pred, 1e-12)
    recall_frame = pooled_overlap / max(pooled_gt, 1e-12)
    return {
        "precision_sequence": precision_sequence,
        "recall_sequence": recall_sequence,
        "f_score_sequence": _f_score(precision_sequence, recall_sequence),
        "precision_frame": precision_frame,
        "recall_frame": recall_frame,
        "f_score_frame": _f_score(precision_frame, recall_frame),
    }

def _threshold_order(value):
    if value == -math.inf:
        return 0, 0.0
    if value == math.inf:
        return 2, 0.0
    return 1, float(value)

def score_depthtrack(series):
    series = tuple(series)
    if not series:
        raise ValueError("DepthTrack scoring requires at least one sequence")
    finite = np.unique(np.concatenate([item.confidence for item in series]))
    candidates = (-math.inf, *finite.tolist(), math.inf)
    scored = {
        value: _metrics_at_threshold(series, value) for value in candidates
    }
    threshold_sequence = max(candidates, key=lambda value: (
        scored[value]["f_score_sequence"], _threshold_order(value),
    ))
    threshold_frame = max(candidates, key=lambda value: (
        scored[value]["f_score_frame"], _threshold_order(value),
    ))
    result = {
        "precision_sequence": scored[threshold_sequence]["precision_sequence"],
        "recall_sequence": scored[threshold_sequence]["recall_sequence"],
        "f_score_sequence": scored[threshold_sequence]["f_score_sequence"],
        "threshold_sequence": threshold_to_json(threshold_sequence),
        "precision_frame": scored[threshold_frame]["precision_frame"],
        "recall_frame": scored[threshold_frame]["recall_frame"],
        "f_score_frame": scored[threshold_frame]["f_score_frame"],
        "threshold_frame": threshold_to_json(threshold_frame),
    }
    validate_unit_interval_metrics(result)
    return result

def validate_unit_interval_metrics(payload):
    for field in UNIT_INTERVAL_FIELDS & payload.keys():
        value = payload[field]
        if (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{field} must be a finite raw unit-interval metric")

def strict_metric_json_dumps(payload):
    validate_unit_interval_metrics(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

def write_depthtrack_result(output_dir, sequence, boxes_xywh, search_scores):
    boxes = np.asarray(boxes_xywh, dtype=np.float64)
    scores = np.asarray(search_scores, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] < 1:
        raise ValueError("boxes must be a nonempty [frames,4] array")
    if scores.shape != (boxes.shape[0] - 1,) or not np.isfinite(scores).all():
        raise ValueError("one finite best_score is required per search frame")
    if not np.isfinite(boxes).all():
        raise ValueError("DepthTrack boxes must be finite")
    confidence = np.concatenate(([1.0], np.clip(scores, 0.0, 1.0)))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    box_path = root / f"{sequence}_001.txt"
    confidence_path = root / f"{sequence}_001_confidence.value"
    integer_boxes = np.rint(boxes).astype(np.int64)
    box_path.write_text(
        "".join(",".join(str(int(value)) for value in row) + "\n" for row in integer_boxes),
        encoding="utf-8", newline="\n",
    )
    confidence_path.write_text(
        "\t".join(format(float(value), ".17g") for value in confidence) + "\n",
        encoding="utf-8", newline="\n",
    )
    return box_path, confidence_path, confidence

def lasher_success_auc(sequence_curves):
    curves = np.asarray(tuple(sequence_curves), dtype=np.float64)
    if (
        curves.ndim != 2 or curves.shape[0] == 0 or curves.shape[1] != 21
        or not np.isfinite(curves).all()
        or ((curves < 0.0) | (curves > 1.0)).any()
    ):
        raise ValueError("LasHeR requires finite [sequence,21] success curves")
    return float(curves.mean(axis=0).mean())

def normalized_tree_sha256(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()

def extract_lasher_author_archive(archive, workspace):
    archive = Path(archive).resolve()
    workspace = Path(workspace).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.unpack_archive(str(archive), str(workspace))
    entries = sorted(workspace.rglob("run_tracker_performance_evaluation.m"))
    if len(entries) != 1:
        raise ValueError("LasHeR archive must contain exactly one verified entry")
    return {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "tree_sha256": normalized_tree_sha256(workspace),
        "entry": entries[0],
    }

def matlab_version(matlab_executable):
    executable = Path(matlab_executable).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("MATLAB executable must be an explicit executable file")
    completed = subprocess.run(
        [str(executable), "-batch", "disp(version)"],
        check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()

def run_lasher_author_artifact(matlab_executable, entry):
    entry = Path(entry).resolve()
    escaped_root = str(entry.parent).replace("'", "''")
    command = f"cd('{escaped_root}'); run_tracker_performance_evaluation"
    return subprocess.run(
        [str(Path(matlab_executable).resolve()), "-batch", command],
        check=True, capture_output=True, text=True,
    )
~~~

Freeze these executable fixtures in `tests/test_spectral_s0_evaluation.py`:

~~~python
def test_depthtrack_all_wrong_uses_pos_inf_and_strict_json(self):
    series = DepthTrackSeries(
        overlap=np.zeros(3), gt_visible=np.ones(3, dtype=bool),
        confidence=np.array([1.0, 0.7, 0.2]),
    )
    result = score_depthtrack([series])
    self.assertEqual(result["threshold_sequence"], {"kind": "pos_inf"})
    encoded = strict_metric_json_dumps(result)
    self.assertNotIn("Infinity", encoded)
    json.loads(encoded, parse_constant=lambda value: self.fail(value))

def test_all_visible_tie_selects_minimum_finite_threshold(self):
    series = DepthTrackSeries(
        overlap=np.ones(3), gt_visible=np.ones(3, dtype=bool),
        confidence=np.array([1.0, 0.7, 0.2]),
    )
    result = score_depthtrack([series])
    self.assertEqual(
        result["threshold_sequence"], {"kind": "finite", "value": 0.2}
    )

def test_threshold_schema_round_trips_negative_infinity(self):
    self.assertEqual(
        threshold_from_json(threshold_to_json(-math.inf)), -math.inf
    )
~~~

`lasher_success_auc()` accepts only per-sequence 21-point curves parsed from the author's MATLAB artifact; it never replaces that artifact. The validator compares the artifact result with the same-input in-repository reference within `1e-8`. Pass a nonexistent temporary child directory to `extract_lasher_author_archive()`.

`BenchmarkMetricAdapter` is the common runner interface; its provenance policy differs by benchmark. The implementation command used by all calibration and gate rows is:

~~~bash
: "${LASHER_AUTHOR_ARCHIVE:?set absolute author-archive path}"
: "${MATLAB_EXECUTABLE:?set absolute MATLAB executable path}"
test -f "$LASHER_AUTHOR_ARCHIVE"
test -x "$MATLAB_EXECUTABLE"
.venv/bin/python tracking/evaluate_benchmark_metrics.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
  --dataset lasher \
  --condition clean \
  --lasher-author-archive "$LASHER_AUTHOR_ARCHIVE" \
  --matlab-executable "$MATLAB_EXECUTABLE" \
  --predictions output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/clean \
  --output output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/clean/benchmark_metrics.json
~~~

For LasHeR, use the author's downloaded MATLAB artifact, not a community reimplementation or a trusted pre-extracted directory. There is no stable direct-download URL in the repository: before the first metric-based calibration choice, a human must download the archive from `https://chenglongli.cn/Datasets-and-benchmark-code/` and provide its absolute path plus an executable MATLAB path. Missing archive/MATLAB is an explicit external blocker; stop rather than substituting a mirror or in-repo formula. The validator hashes the original archive, extracts it itself into a fresh temporary directory, requires exactly one verified `run_tracker_performance_evaluation.m`, hashes the normalized extracted tree, records MATLAB executable/version, and runs the fixture there. Record the source-page URL, archive SHA-256, extracted-tree SHA-256, MATLAB version, verified entry, and normalized command template equivalent to `matlab -batch "cd('<isolated-toolkit-root>'); run_tracker_performance_evaluation"`. Every later adapter invocation re-hashes the supplied archive, requires the frozen SHA, extracts to a new isolated workspace, and verifies the frozen tree hash before execution. The adapter writes one comma-separated `x,y,w,h` row per frame, installs the locked gate-manifest sequence list into that workspace, and parses the sequence-macro success-curve AUC over thresholds `0:0.05:1` as `success_auc`. It labels a gate-subset result as such; only an untouched official `testingsetList.txt` run may be labeled Protocol 2.

DepthTrack's official DeT repository exposes result writing but no public scoring entry point, so never claim an official DepthTrack toolkit. Implement a paper-faithful adapter from the ICCV 2021 main/supplement formulas and label it `depthtrack_paper_faithful_v1`. It emits `<sequence>_001.txt` with comma-separated integer `x,y,w,h` and `<sequence>_001_confidence.value` with one tab-separated confidence per frame. Freeze the visibility confidence as `1.0` for the legal initialization frame and `clamp(tracker_out["best_score"],0,1)` for every search frame; reject missing/nonfinite scores instead of inventing confidence. A prediction is visible at threshold `tau` iff `confidence >= tau`. For each threshold, let `overlap_sum` be summed IoU on frames where prediction and GT are visible, `N_pred` predicted-visible frames, and `N_gt` GT-visible frames; compute sequence precision `overlap_sum/max(N_pred,eps)`, recall `overlap_sum/max(N_gt,eps)`, macro-average each over sequences, then `F=2PR/max(P+R,eps)`. Also compute the frame-pooled form. Scan all unique reported confidence values plus conceptual `+inf/-inf`, choose maximal F, and break exact ties toward the higher threshold under total order `-inf < finite < +inf`.

Serialize `threshold_sequence` and `threshold_frame` canonically as exactly one of `{"kind":"finite","value":x}`, `{"kind":"pos_inf"}`, or `{"kind":"neg_inf"}`; finite `x` must be a finite JSON number. Strict JSON forbids literal `NaN`, `Infinity`, and `-Infinity`. Parsing recovers the conceptual sentinel only for comparison, never as an emitted float. Fixtures cover equality at a finite confidence, all-wrong selecting `pos_inf`, all-visible selecting the minimum finite threshold, exact tie ordering, strict JSON parsing, canonical hash stability, and an explicit `neg_inf` schema round trip without requiring the selector to choose it. The primary field is `f_score_sequence`; also store `precision_sequence`, `recall_sequence`, and all four frame-pooled analogues. The gate adapter reads GT only through the locked custom non-test manifest and verifies its source-tree hash against `target_spectral_split_audit.json`; label the result `heldout_non_test_gate`. Record official dataset DOI `10.5281/zenodo.5792146` as the protocol source, not as gate input. An optional official-50 fixture may audit published formatting/results but can never replace the locked gate manifest. Bbox-only results fail.

`tools/validate_benchmark_evaluators.py` requires `--lasher-author-archive` and `--matlab-executable`; it never accepts only an extracted root or an executable inferred from `PATH`. It runs committed empty-GT, empty-prediction, all-correct, all-wrong, equal-confidence, low-confidence-filter, and unequal-sequence-length fixtures. It compares LasHeR artifact output with the in-repo reference formula within `1e-8`, verifies the author's archive/tree/entry provenance, and checks the DepthTrack adapter's reference fixtures. Freeze fails on any mismatch. The frozen registry stores command arrays and explicitly records `lasher_author_artifact` versus `depthtrack_paper_faithful_v1`.

- [ ] **Step 6: Implement acyclic gate-registry freezing**

The freeze tool must fill concrete:

- selected common rank and M1 energy table;
- the canonical required-M1-cell list and its SHA-256;
- confidence threshold and achieved calibration coverage;
- selected coefficient checkpoint and alpha values;
- the fixed fit-attempt manifest, initial and selected-checkpoint feasibility hashes, and the named 48-cell pair-alignment audit hash;
- `parent_calibration_registry_sha256`, locked-calibration artifact hash, all manifest/base-checkpoint hashes, and the single global coefficient-checkpoint hash;
- LasHeR author-artifact source URL, archive/tree hashes, MATLAB version/entry/command, and metric field `success_auc`;
- DepthTrack paper/DOI sources, actual gate-manifest/source-tree hashes, adapter code hash, scope `heldout_non_test_gate`, and primary field `f_score_sequence`;
- one exact batch-one profile CUDA fingerprint and two ordered, distinct DDP CUDA fingerprints (`name`, UUID, PCI bus ID, compute capability, total bytes, driver), plus timing protocol;
- adapter raw scale `unit_interval`, the sole raw-contrast-to-pp multiplier `100.0`, the six-stratum frozen profile sequence rule/warmup/iteration counts, FPS/memory limits, and frame-level actual-commit coverage definition/threshold;
- fixed recovery/corruption fields;
- schedule generator algorithm/version, threshold, attention floor, seed, and six input checkpoint hashes, but no future schedule/result hash;
- freeze timestamp.

It writes `spectral_s0_v1.frozen.yaml`, validates the parent chain, and refuses overwrite if content differs. Candidate and byte-identical global coefficient checkpoints store only already-existing calibration, locked-calibration, attempt-manifest, initial-feasibility, base-checkpoint, and fit-manifest parent hashes; they never store a coefficient-selection or selected-feasibility descendant hash. The coefficient-selection record points to the chosen candidate, the selected-feasibility child points to selection plus candidate, and the frozen registry points to that complete one-way chain. Gate schedules are generated only after this registry is committed, so there is no hash cycle.

Implement the freeze as a pure parent-to-child materialization; `artifacts` contains already parsed objects together with their verified file hashes:

~~~python
from collections.abc import Mapping

def canonical_yaml_bytes(value):
    return yaml.safe_dump(
        value, sort_keys=True, allow_unicode=True, default_flow_style=False
    ).encode("utf-8")

def require_parent(child, field, expected):
    actual = child.get(field)
    if actual != expected:
        raise ValueError(f"{field} mismatch: expected {expected}, got {actual}")

def validate_frozen_hardware(hardware):
    fingerprint_fields = {
        "name", "uuid", "pci_bus_id", "compute_capability",
        "total_memory_bytes", "driver_version",
    }
    if set(hardware) != {
        "profile_cuda_device", "ddp_cuda_devices", "batch_size",
        "timing_protocol",
    } or type(hardware["batch_size"]) is not int or hardware["batch_size"] != 1:
        raise ValueError("frozen hardware schema/batch mismatch")
    if hardware["timing_protocol"] != "synchronized_complete_episode_v1":
        raise ValueError("unknown frozen timing protocol")
    profile = hardware["profile_cuda_device"]
    ddp_devices = hardware["ddp_cuda_devices"]
    def valid_fingerprint(device):
        return (
            set(device) == fingerprint_fields
            and all(
                isinstance(device[field], str) and bool(device[field])
                for field in fingerprint_fields - {"total_memory_bytes"}
            )
            and type(device["total_memory_bytes"]) is int
            and device["total_memory_bytes"] > 0
        )
    if (
        not valid_fingerprint(profile)
        or len(ddp_devices) != 2
        or any(not valid_fingerprint(device) for device in ddp_devices)
        or len({device["uuid"] for device in ddp_devices}) != 2
        or len({device["pci_bus_id"] for device in ddp_devices}) != 2
        or profile != ddp_devices[0]
    ):
        raise ValueError("invalid profile/DDP CUDA fingerprints")
    return copy.deepcopy(hardware)

def materialize_frozen_registry(calibration, artifacts, hardware, code_commit, freeze_time):
    locked = artifacts["locked_calibration"]
    coefficient = artifacts["coefficient_selection"]
    attempt_manifest = artifacts["fit_attempt_manifest"]
    initial_feasibility = artifacts["fit_feasibility_initial"]
    selected_feasibility = artifacts["fit_feasibility_selected"]
    pair_alignment = artifacts["pair_alignment_audit"]
    corruption = artifacts["corruption_selection"]
    evaluators = artifacts["evaluator_validation"]
    bases = artifacts["base_checkpoint_index"]
    calibration_sha = artifacts["calibration_registry_sha256"]
    hardware = validate_frozen_hardware(hardware)
    require_parent(locked, "calibration_registry_sha256", calibration_sha)
    require_parent(coefficient, "calibration_registry_sha256", calibration_sha)
    require_parent(corruption, "evaluator_validation_sha256", artifacts["evaluator_validation_sha256"])
    require_parent(coefficient, "evaluator_validation_sha256", artifacts["evaluator_validation_sha256"])
    if len(bases["checkpoints"]) != 6:
        raise ValueError("frozen registry requires six base checkpoints")
    if coefficient["attempt_manifest_sha256"] != artifacts["fit_attempt_manifest_sha256"]:
        raise ValueError("coefficient selection has wrong attempt manifest")
    if coefficient["selected_checkpoint_sha256"] != artifacts["coefficient_checkpoint_sha256"]:
        raise ValueError("selected coefficient checkpoint is not the byte-identical copy")
    if coefficient["initial_feasibility_sha256"] != artifacts["fit_feasibility_initial_sha256"]:
        raise ValueError("coefficient selection has wrong initial feasibility audit")
    for report in (initial_feasibility, selected_feasibility):
        if report["attempt_manifest_sha256"] != artifacts["fit_attempt_manifest_sha256"]:
            raise ValueError("feasibility audit has wrong attempt manifest")
        for field in (
            "calibration_registry_sha256", "locked_calibration_sha256",
            "base_checkpoint_index_sha256", "fit_manifest_sha256",
            "config_sha256",
        ):
            require_parent(report, field, coefficient[field])
    if (
        selected_feasibility["coefficient_selection_sha256"]
            != artifacts["coefficient_selection_sha256"]
        or selected_feasibility["selected_checkpoint_sha256"]
            != coefficient["selected_checkpoint_sha256"]
    ):
        raise ValueError("selected feasibility has wrong selection/checkpoint parent")
    expected_attempts = (
        6
        * int(calibration["coefficient_fit"]["attempted_clips_per_stratum_per_superstep"])
        * int(calibration["coefficient_fit"]["maximum_attempted_supersteps"])
    )
    if int(attempt_manifest["attempt_count"]) != expected_attempts:
        raise ValueError("fit attempt manifest has wrong fixed size")
    if len(pair_alignment["cells"]) != 48:
        raise ValueError("pair-alignment audit must contain 48 named cells")
    for field in (
        "calibration_registry_sha256", "locked_admission_sha256",
        "rank_sketches_sha256", "base_checkpoint_index_sha256",
        "calibration_manifest_sha256_by_modality",
    ):
        require_parent(pair_alignment, field, locked[field])
    if set(pair_alignment["calibration_manifest_sha256_by_modality"]) != {
        "rgbd", "rgbt",
    }:
        raise ValueError("pair-alignment calibration manifests must be RGB-D/RGB-T exact")
    if (
        locked["pair_alignment_audit_sha256"]
            != artifacts["pair_alignment_audit_sha256"]
        or int(pair_alignment["bootstrap_seed"])
            != int(calibration["statistics"]["bootstrap_seed"])
        or int(pair_alignment["shuffle_seed"])
            != int(calibration["controls"]["shuffle_seed"])
    ):
        raise ValueError("pair-alignment provenance/seed mismatch")
    required_successes = int(
        calibration["coefficient_fit"]["required_successful_optimizer_steps"]
    )
    maximum_attempts = int(
        calibration["coefficient_fit"]["maximum_attempted_supersteps"]
    )
    selected_step = int(coefficient["selected_successful_step"])
    selected_attempts = int(coefficient["selected_checkpoint_attempted_supersteps"])
    completed_successes = int(coefficient["fit_completed_successful_steps"])
    completed_attempts = int(coefficient["fit_completed_attempted_supersteps"])
    if (
        selected_step not in calibration["coefficient_fit"]["checkpoint_successful_steps"]
        or completed_successes != required_successes
        or not selected_step <= selected_attempts <= completed_attempts
        or not completed_successes <= completed_attempts <= maximum_attempts
    ):
        raise ValueError("coefficient fit did not satisfy frozen step ceilings")

    frozen = copy.deepcopy(calibration)
    frozen["status"] = "frozen"
    frozen["freeze"] = {
        "timestamp": freeze_time,
        "code_commit": code_commit,
        "hardware": hardware,
        "parent_calibration_registry_sha256": calibration_sha,
        "locked_calibration_sha256": artifacts["locked_calibration_sha256"],
        "base_checkpoint_index_sha256": artifacts["base_checkpoint_index_sha256"],
        "evaluator_validation_sha256": artifacts["evaluator_validation_sha256"],
    }
    frozen["memory"]["selected_rank"] = int(locked["selected_rank"])
    frozen["memory"]["m1_table"] = locked["m1_table"]
    frozen["memory"]["required_m1_cells"] = locked["required_m1_cells"]
    frozen["admission"]["q_memory_threshold"] = float(locked["q_memory_threshold"])
    frozen["admission"]["attention_floor"] = float(locked["attention_floor"])
    frozen["admission"]["realized_coverage"] = locked["realized_coverage"]
    frozen["admission"]["pair_alignment_audit_sha256"] = artifacts[
        "pair_alignment_audit_sha256"
    ]
    frozen["coefficient"] = {
        "selected_successful_step": int(coefficient["selected_successful_step"]),
        "selected_checkpoint_attempted_supersteps": int(
            coefficient["selected_checkpoint_attempted_supersteps"]
        ),
        "fit_completed_successful_steps": int(
            coefficient["fit_completed_successful_steps"]
        ),
        "fit_completed_attempted_supersteps": int(
            coefficient["fit_completed_attempted_supersteps"]
        ),
        "selection_sha256": artifacts["coefficient_selection_sha256"],
        "checkpoint_sha256": coefficient["selected_checkpoint_sha256"],
        "alpha": [float(value) for value in coefficient["alpha"]],
        "shared_across_six_bases": True,
        "attempt_manifest_sha256": coefficient["attempt_manifest_sha256"],
        "initial_feasibility_sha256": coefficient["initial_feasibility_sha256"],
        "selected_feasibility_sha256": artifacts["fit_feasibility_selected_sha256"],
    }
    frozen["base_checkpoints"] = sorted(
        bases["checkpoints"], key=lambda row: (row["base_seed"], row["modality"])
    )
    frozen.setdefault("corruption", {})["selected_severity"] = float(
        corruption["selected_severity"]
    )
    frozen["benchmarks"]["lasher"]["provenance"] = evaluators["lasher_author_artifact"]
    frozen["benchmarks"]["depthtrack"]["provenance"] = evaluators["depthtrack_paper_faithful_v1"]
    frozen["schedule_generator"] = locked["schedule_generator"]
    forbidden = {"gate_schedule_manifest_sha256", "gate_result_sha256", "s0_gate_sha256"}
    def walk_key_paths(value, path=()):
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield from walk_key_paths(child, path + (str(key),))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                yield from walk_key_paths(child, path + (str(index),))
        else:
            yield path, value
    for path, _ in walk_key_paths(frozen):
        if path and path[-1] in forbidden:
            raise ValueError(f"frozen registry contains future child hash at {path}")
    validate_registry(frozen, purpose="gate_confirmation")
    return frozen

def write_frozen_registry(path, frozen):
    payload = canonical_yaml_bytes(frozen)
    output = Path(path)
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError("refusing to overwrite a different frozen registry")
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
~~~

The CLI constructs `artifacts` only after hashing and strict-loading every supplied path. It records the source/addendum/manifests, six base hashes, coefficient checkpoint hash, exact recovery and corruption fields, LasHeR archive/tree/MATLAB/entry/command, DepthTrack DOI/source-tree/adapter-code fields, raw metric scale/multiplier, timing protocol, selected profile rule, and coverage definition before `write_frozen_registry()`. The returned object contains schedule-generator inputs but no schedule/result child hash.

- [ ] **Step 7: Analyze the engineering smoke**

Run:

~~~bash
.venv/bin/python tracking/analyze_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --input output/spectral_s0/smoke \
  --output output/spectral_s0/smoke/summary.json \
  --engineering-smoke
~~~

Expected: complete schema, `engineering_smoke=true`, `gate_decision=null`, and explicit insufficient-seed/sequence warnings.

- [ ] **Step 8: Verify statistics and benchmark adapters**

~~~bash
.venv/bin/python -m unittest tests.test_spectral_s0_evaluation \
  tests.test_spectral_config_registry -v
repo="$PWD"
for script in \
  tracking/analyze_spectral_s0.py \
  tracking/evaluate_benchmark_metrics.py \
  tools/freeze_spectral_s0_registry.py \
  tools/validate_benchmark_evaluators.py; do
  .venv/bin/python "$script" --help >/dev/null
  (cd /tmp && "$repo/.venv/bin/python" "$repo/$script" --help >/dev/null)
done
~~~

Expected: both test modules PASS and all eight root/`/tmp` CLI help invocations exit zero; the unit-scale, nonlinear seed-slot bootstrap, recovery, strict noninferiority, evaluator-provenance, and acyclic-freeze fixtures all pass.

- [ ] **Step 9: Commit preregistered analysis code**

~~~bash
git add lib/test/evaluation/spectral_statistics.py \
  lib/test/evaluation/benchmark_metrics.py tracking/analyze_spectral_s0.py \
  tracking/evaluate_benchmark_metrics.py tools/freeze_spectral_s0_registry.py \
  tools/validate_benchmark_evaluators.py tests/test_spectral_s0_evaluation.py \
  tests/test_spectral_config_registry.py
git commit -m "feat: add preregistered spectral s0 statistics"
~~~

### Task 12: M0/M1, DDP, Efficiency, and S0 Execution Gates

**Files:**

- Create: `tools/profile_spectral_s0.py`
- Create: `tests/test_profile_spectral_s0.py`
- Modify: `tracking/analyze_spectral_s0.py`
- Modify: `tests/test_spectral_s0_evaluation.py`
- Modify: `knowledge_base/Target-Spectral-S0-实验记录.md`
- Create: `knowledge_base/Target-Spectral-S0-gate.json`

**Interfaces:**

- Profiler reports `released_no_context_legacy` (`spectral_context=None`, no observer/controller), observer-instrumented `routing_disabled_legacy`, zero-strength, and active rows at batch one with synchronized CUDA timing.
- Its read-only `--capture-hardware` mode records the exact visible ordered CUDA fingerprints used by formal batch-one profiling and two-process DDP; it refuses fewer/more than two devices, duplicate UUIDs, or a profile device other than ordered device zero.
- M0 reports bitwise output identity and explicit mutable buffers.
- M1 reports exact total-trace energy and boundary eigengap for every family/key/rank and DDP/single-process relative operator error.
- `frozen_gate_audit.json` hashes six frozen-registry profile artifacts, six clean full-row M1 tables, one frozen-selected-rank two-process DDP operator artifact, the coefficient-selection/attempt/dual-feasibility chain, twelve condition summaries, and the named 48-cell pair-alignment audit; it computes completion, full-row/profile actual-commit coverage, and every engineering gate from raw sources and is mandatory for final analysis.
- The final runner emits a machine-readable `s0_gate.json`; it never silently starts Stage R.

- [ ] **Step 1: Write failing profiler schema and accounting tests**

Require:

~~~python
import math
import struct

def canonical_public_prediction_bytes(frame_index, result):
    allowed = {"target_bbox", "best_score", "all_boxes", "target_spectral"}
    if set(result) - allowed:
        raise ValueError("unexpected public prediction field")
    target_bbox = tuple(float(value) for value in result["target_bbox"])
    best_score = float(result["best_score"])
    all_boxes = tuple(float(value) for value in result.get("all_boxes", ()))
    if (
        type(frame_index) is not int or frame_index < 1
        or len(target_bbox) != 4 or len(all_boxes) % 4
        or not all(math.isfinite(value) for value in (*target_bbox, best_score, *all_boxes))
    ):
        raise ValueError("invalid public prediction record")
    header = b"TSM0\x01" + struct.pack(">II", frame_index, len(all_boxes))
    values = (*target_bbox, best_score, *all_boxes)
    return header + struct.pack(f">{len(values)}d", *values)

REQUIRED_PROFILE_FIELDS = {
    "method", "base_seed", "benchmark", "profile_sequence_id",
    "profile_sequence_rule", "single_method_process", "process_pid",
    "post_initialization_frames", "warmup_frames", "measured_frames",
    "raw_frame_latency_ms", "raw_commit_latency_ms", "raw_actual_committed",
    "initialization_latency_ms",
    "tracking_latency_ms_total", "episode_latency_ms_total",
    "tracking_fps", "episode_fps", "frame_latency_ms_p50",
    "frame_latency_ms_p95", "commit_latency_ms_p50", "commit_latency_ms_p95",
    "resting_allocated_bytes", "resting_reserved_bytes",
    "absolute_peak_allocated_bytes", "absolute_peak_reserved_bytes",
    "persistent_state_bytes", "padding_state_bytes",
    "episode_actual_commits", "episode_write_coverage",
    "measured_actual_commits", "measured_write_coverage",
    "proposed_writes", "scheduled_admits", "factor_rejections",
    "raw_prediction_frame_bytes_hex",
    "initialization_calibration_forward_count",
    "tracking_network_forward_count", "total_network_forward_count",
    "base_parameter_sha256_before", "base_parameter_sha256_after",
    "cuda_device", "batch_size", "registry_sha256", "schedule_manifest_sha256",
    "checkpoint_sha256", "coefficient_sha256", "baseline_method",
}
~~~

`raw_prediction_frame_bytes_hex` contains exactly 60 lowercase hex encodings returned by `canonical_public_prediction_bytes()` in chronological order; it compares only the existing public prediction compatibility fields and deliberately excludes active-only `target_spectral` diagnostics. It is evidence, not a claimed equality flag. Each child hashes the sorted named model parameters immediately before initialization and after the episode and emits both hashes plus raw forward counters. The frozen audit validates the magic/version/frame/length layout and compares the prediction bytes itself, checks the before/after hashes itself, and derives every forward-count result from the phase counters; producer-supplied `bitwise_*`, `*_equals_*`, `*_unchanged`, or M0 `pass` booleans are forbidden.

Also assert persistent state is at most 8 MiB; the locked rank still reaches 90% exact total-trace energy with relative eigengap at least 0.01 for every key/family; and `released_no_context_legacy`, observer-instrumented routing-disabled, and zero-strength outputs are bitwise equal after raw-byte recomputation. In isolated child processes, active complete-episode FPS (including initialization) must be at least 0.80 times the true no-context baseline, while both absolute peak allocated and absolute peak reserved GPU bytes must be at most 1.25 times that baseline. The observer-instrumented row is used for matched schedules/diagnostics but never as the denominator. A gate-analysis test omits or mutates each raw prediction byte, phase forward count, or pre/post parameter hash and each of the six profile/M1 artifacts or the frozen DDP artifact in turn and requires fail-closed rejection. Coverage fixtures define one frame-level `actual_committed` bit and prove `sum(actual_committed)/post_initialization_frames`, not scheduled-admit coverage or family-write count, is gated at `0.20` overall, in every one of the six `(base_seed,benchmark)` strata, and inside the measured region of every frozen profile.

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m unittest tests.test_profile_spectral_s0 tests.test_spectral_s0_evaluation -v`

Expected: FAIL because the profiler does not exist.

- [ ] **Step 3a: Select the one frozen efficiency sequence per stratum**

~~~python
def select_profile_sequence(dataset, sequences, warmup_frames=10, measured_frames=50):
    required = warmup_frames + measured_frames
    eligible = []
    for sequence in sequences:
        post_init_frames = int(sequence["num_frames"]) - 1
        if sequence["condition"] != "clean" or post_init_frames < required:
            continue
        name = str(sequence["sequence_name"])
        digest = hashlib.sha256(f"{dataset}:{name}".encode("utf-8")).hexdigest()
        eligible.append((digest, name, sequence))
    if not eligible:
        raise ValueError("no clean sequence has 60 post-initialization frames")
    return min(eligible, key=lambda item: (item[0], item[1]))[2]
~~~

Call this once for each frozen `(base_seed,benchmark)` stratum before running any method. Pass the same returned sequence ID to all four methods; never select a replacement from observed writes or latency.

- [ ] **Step 3b: Implement isolated-process complete-episode timing and absolute GPU peaks**

~~~python
import gc
import os
import time

def profile_complete_episode(
    initialize_fn, predict_fn, commit_fn, episode,
    warmup_frames=10, measured_frames=50,
):
    if len(episode.search_frames) != warmup_frames + measured_frames:
        raise ValueError("profile episode requires exactly 60 search frames")
    # Model load and one separate warm-up episode occur before this function.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resting_allocated = int(torch.cuda.memory_allocated())
    resting_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()

    episode_started = time.perf_counter_ns()
    torch.cuda.synchronize()
    initialization_started = time.perf_counter_ns()
    initialize_fn(episode.initial_image, episode.init_info)
    torch.cuda.synchronize()
    initialization_finished = time.perf_counter_ns()

    frame_latency_ms = []
    commit_latency_ms = []
    committed = []
    for image, safe_info in episode.search_frames:
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        pending = predict_fn(image, safe_info)
        torch.cuda.synchronize()
        update_started = time.perf_counter_ns()
        result = commit_fn(pending)
        torch.cuda.synchronize()
        finished = time.perf_counter_ns()
        frame_latency_ms.append((finished - started) / 1_000_000.0)
        commit_latency_ms.append((finished - update_started) / 1_000_000.0)
        actual_committed = result["actual_committed"]
        if type(actual_committed) is not bool:
            raise TypeError("actual_committed must be an exact bool")
        committed.append(actual_committed)
    episode_finished = time.perf_counter_ns()
    initialization_ms = (initialization_finished - initialization_started) / 1e6
    tracking_ms = float(sum(frame_latency_ms))
    episode_ms = (episode_finished - episode_started) / 1e6
    measured_latency = frame_latency_ms[warmup_frames:]
    measured_commit = commit_latency_ms[warmup_frames:]
    measured_committed = committed[warmup_frames:]
    if min(tracking_ms, episode_ms) <= 0.0:
        raise RuntimeError("profile duration must be positive")
    return {
        "single_method_process": True,
        "process_pid": int(os.getpid()),
        "post_initialization_frames": len(frame_latency_ms),
        "warmup_frames": int(warmup_frames),
        "measured_frames": len(measured_latency),
        "raw_frame_latency_ms": [float(value) for value in frame_latency_ms],
        "raw_commit_latency_ms": [float(value) for value in commit_latency_ms],
        "raw_actual_committed": committed,
        "initialization_latency_ms": initialization_ms,
        "tracking_latency_ms_total": tracking_ms,
        "episode_latency_ms_total": episode_ms,
        "tracking_fps": len(frame_latency_ms) / (tracking_ms / 1000.0),
        "episode_fps": len(frame_latency_ms) / (episode_ms / 1000.0),
        "frame_latency_ms_p50": float(np.quantile(measured_latency, 0.50)),
        "frame_latency_ms_p95": float(np.quantile(measured_latency, 0.95)),
        "commit_latency_ms_p50": float(np.quantile(measured_commit, 0.50)),
        "commit_latency_ms_p95": float(np.quantile(measured_commit, 0.95)),
        "resting_allocated_bytes": resting_allocated,
        "resting_reserved_bytes": resting_reserved,
        "absolute_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "absolute_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "episode_actual_commits": int(sum(committed)),
        "episode_write_coverage": float(sum(committed) / len(committed)),
        "measured_actual_commits": int(sum(measured_committed)),
        "measured_write_coverage": float(
            sum(measured_committed) / len(measured_committed)
        ),
    }

def finalize_profile(method, measured, snapshot=None, inert_padding=()):
    if method == "released_no_context_legacy":
        measured["persistent_state_bytes"] = 0
        measured["padding_state_bytes"] = 0
    else:
        measured["persistent_state_bytes"] = persistent_state_bytes(
            snapshot, inert_padding=inert_padding
        )
        measured["padding_state_bytes"] = unique_tensor_storage_bytes(inert_padding)
    return measured
~~~

The parent profiler launches four fresh child processes; each child accepts exactly one `--method`, loads the same checkpoint/episode, performs a separate unmeasured warm-up episode, resets globally, then calls `profile_complete_episode()`. The parent rejects repeated PIDs or a child reporting `single_method_process != true`. `predict_fn()` performs the full tracker prediction and returns a pending detached transaction; `commit_fn()` performs only the prediction-dependent spectral commit and returns its exact frame boolean. The episode timer includes the additional active initialization calibration forward and all 60 search frames; p50/p95 and measured write coverage use the final fixed 50 frames. The released baseline supplies a no-op `commit_fn`, invokes prediction with `spectral_context=None`, constructs no controller/observer transaction, and is the sole denominator. Binding gates use `episode_fps` and both absolute allocated/reserved peaks, never the subtracted temporary delta. The observer-instrumented legacy row remains a separately named cost audit. Reject `full_four_spectrum` when `measured_write_coverage < 0.20`.

- [ ] **Step 3c: Build and require the frozen six-stratum audit**

~~~python
import struct

REQUIRED_GATE_STRATA = frozenset(
    (seed, benchmark)
    for seed in (0, 1, 2)
    for benchmark in ("lasher", "depthtrack")
)

def index_exact_strata(rows, label):
    indexed = {}
    for row in rows:
        key = (int(row["base_seed"]), row["benchmark"])
        if key in indexed:
            raise ValueError(f"duplicate {label} stratum: {key}")
        indexed[key] = row
    if set(indexed) != REQUIRED_GATE_STRATA:
        raise ValueError(f"{label} strata are incomplete")
    return indexed

def strict_bool(value, label):
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value

def strict_sha256(value, label):
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value

def decode_raw_prediction_trace(row, expected_frames):
    encoded = row["raw_prediction_frame_bytes_hex"]
    if not isinstance(encoded, list) or len(encoded) != expected_frames:
        raise ValueError("raw prediction trace has wrong frame count")
    decoded = []
    for frame_index, value in enumerate(encoded, 1):
        if not isinstance(value, str) or not value or len(value) % 2:
            raise ValueError(f"invalid prediction hex at frame {frame_index}")
        try:
            raw = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError(
                f"invalid prediction hex at frame {frame_index}"
            ) from error
        if not raw.startswith(b"TSM0\x01") or len(raw) < 13 or raw.hex() != value:
            raise ValueError(f"noncanonical prediction hex at frame {frame_index}")
        stored_frame, all_box_count = struct.unpack(">II", raw[5:13])
        expected_length = 13 + 8 * (5 + all_box_count)
        if (
            stored_frame != frame_index or all_box_count % 4
            or len(raw) != expected_length
        ):
            raise ValueError(f"prediction byte layout mismatch at frame {frame_index}")
        values = struct.unpack(f">{5 + all_box_count}d", raw[13:])
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"nonfinite prediction bytes at frame {frame_index}")
        decoded.append(raw)
    return tuple(decoded)

def reject_producer_conclusion_fields(value, path="profile"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if (
                name in {
                    "m0", "m0_pass", "m0_checks", "m0_cells", "pass",
                    "forward_counts_match_protocol",
                }
                or name.startswith("bitwise_")
                or "_equals_" in name
                or "_matches_" in name
                or name.endswith("_unchanged")
                or name.endswith("_pass")
                or name.endswith("_passed")
                or name in {"checks", "computed_gates", "gate_decision"}
            ):
                raise ValueError(
                    f"producer-supplied conclusion field is forbidden: {path}.{key}"
                )
            reject_producer_conclusion_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_producer_conclusion_fields(child, f"{path}[{index}]")

PROFILE_ARTIFACT_FIELDS = frozenset({
    "schema_version", "base_seed", "benchmark", "baseline_method", "methods",
})

def recompute_m0_from_raw(artifact, expected_frames, expected_methods):
    reject_producer_conclusion_fields(artifact)
    if (
        set(artifact) != PROFILE_ARTIFACT_FIELDS
        or type(artifact["schema_version"]) is not int
        or artifact["schema_version"] != 1
    ):
        raise ValueError("profile artifact schema mismatch")
    methods = artifact.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(expected_methods):
        raise ValueError("M0 method set mismatch")
    traces, before_hashes, unchanged, count_protocol = {}, set(), {}, {}
    for method in sorted(expected_methods):
        row = methods[method]
        if not isinstance(row, dict) or set(row) != REQUIRED_PROFILE_FIELDS:
            raise ValueError(f"profile method schema mismatch: {method}")
        traces[method] = decode_raw_prediction_trace(row, expected_frames)
        before = strict_sha256(
            row["base_parameter_sha256_before"], f"{method} before parameters"
        )
        after = strict_sha256(
            row["base_parameter_sha256_after"], f"{method} after parameters"
        )
        before_hashes.add(before)
        unchanged[method] = before == after
        count_fields = (
            "initialization_calibration_forward_count",
            "tracking_network_forward_count", "total_network_forward_count",
        )
        if any(type(row[field]) is not int or row[field] < 0 for field in count_fields):
            raise ValueError(f"invalid raw forward count: {method}")
        initialization_count = row["initialization_calibration_forward_count"]
        tracking_count = row["tracking_network_forward_count"]
        total_count = row["total_network_forward_count"]
        expected_initialization = (
            0 if method == "released_no_context_legacy" else 1
        )
        count_protocol[method] = (
            initialization_count == expected_initialization
            and tracking_count == expected_frames
            and total_count == initialization_count + tracking_count
        )
    if len(before_hashes) != 1:
        raise ValueError("profile methods did not start from one parameter state")
    return {
        "routing_disabled_equals_released": (
            traces["routing_disabled_legacy"]
            == traces["released_no_context_legacy"]
        ),
        "zero_strength_equals_released": (
            traces["zero_strength_instrumented"]
            == traces["released_no_context_legacy"]
        ),
        "forward_counts_match_protocol": all(count_protocol.values()),
        "base_parameter_hash_unchanged": all(unchanged.values()),
    }

def recompute_tangent_summary(signed_leaf_rows, alpha, sum_tolerance):
    signed_leaf = np.asarray(signed_leaf_rows, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if (
        signed_leaf.ndim != 2 or signed_leaf.shape[1] != 4
        or signed_leaf.shape[0] < 3 or alpha.shape != (4,)
        or not np.isfinite(signed_leaf).all() or not np.isfinite(alpha).all()
        or (alpha <= 0.0).any()
    ):
        raise ValueError("invalid raw tangent-feasibility inputs")
    jacobian = np.diag(alpha) - np.outer(alpha, alpha) / float(alpha.sum())
    u_gradient = signed_leaf @ jacobian
    if np.abs(u_gradient.sum(axis=1)).max() > sum_tolerance:
        raise ValueError("softmax tangent gradient does not sum to zero")
    helmert = np.asarray([
        [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
        [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
        [0.0, -2 / np.sqrt(6), 1 / np.sqrt(12)],
        [0.0, 0.0, -3 / np.sqrt(12)],
    ], dtype=np.float64)
    tangent = u_gradient @ helmert
    singular = np.linalg.svd(
        tangent / np.sqrt(tangent.shape[0]), compute_uv=False
    )
    return {
        "singular": singular,
        "relative_min": float(
            singular[-1] / max(singular[0], np.finfo(np.float64).tiny)
        ),
        "leaf_abs_mean": np.abs(signed_leaf).mean(axis=0),
        "valid_attempts": int(signed_leaf.shape[0]),
    }

def recompute_profile_derived(row, registry, expected_device):
    expected_frames = int(registry["efficiency"]["complete_episode_search_frames"])
    expected_warmup = int(registry["efficiency"]["warmup_frames"])
    expected_measured = int(registry["efficiency"]["measured_frames"])
    if (
        int(row["batch_size"]) != 1
        or row["cuda_device"] != expected_device
        or int(row["post_initialization_frames"]) != expected_frames
        or int(row["warmup_frames"]) != expected_warmup
        or int(row["measured_frames"]) != expected_measured
    ):
        raise ValueError("profile hardware/batch/frame protocol mismatch")
    frame = np.asarray(row["raw_frame_latency_ms"], dtype=np.float64)
    commit = np.asarray(row["raw_commit_latency_ms"], dtype=np.float64)
    committed = tuple(row["raw_actual_committed"])
    if (
        frame.shape != (expected_frames,)
        or commit.shape != (expected_frames,)
        or len(committed) != expected_frames
        or not np.isfinite(frame).all()
        or not np.isfinite(commit).all()
        or (frame <= 0.0).any()
        or (commit < 0.0).any()
        or any(type(value) is not bool for value in committed)
    ):
        raise ValueError("invalid raw profile vectors")
    measured_frame = frame[expected_warmup:]
    measured_commit = commit[expected_warmup:]
    measured_committed = committed[expected_warmup:]
    tracking_ms = float(frame.sum())
    episode_ms = float(row["episode_latency_ms_total"])
    initialization_ms = float(row["initialization_latency_ms"])
    if (
        not math.isfinite(episode_ms) or episode_ms <= 0.0
        or not math.isfinite(initialization_ms) or initialization_ms <= 0.0
        or episode_ms + 1e-9 < initialization_ms + tracking_ms
    ):
        raise ValueError("invalid complete-episode latency")
    derived = {
        "tracking_latency_ms_total": tracking_ms,
        "tracking_fps": expected_frames / (tracking_ms / 1000.0),
        "episode_fps": expected_frames / (episode_ms / 1000.0),
        "frame_latency_ms_p50": float(np.quantile(measured_frame, 0.50)),
        "frame_latency_ms_p95": float(np.quantile(measured_frame, 0.95)),
        "commit_latency_ms_p50": float(np.quantile(measured_commit, 0.50)),
        "commit_latency_ms_p95": float(np.quantile(measured_commit, 0.95)),
        "episode_actual_commits": int(sum(committed)),
        "episode_write_coverage": float(sum(committed) / expected_frames),
        "measured_actual_commits": int(sum(measured_committed)),
        "measured_write_coverage": float(
            sum(measured_committed) / expected_measured
        ),
    }
    for field, expected in derived.items():
        actual = row[field]
        if isinstance(expected, int):
            if type(actual) is not int or actual != expected:
                raise ValueError(f"profile derived count mismatch: {field}")
        elif not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"profile derived float mismatch: {field}")
    byte_fields = (
        "resting_allocated_bytes", "resting_reserved_bytes",
        "absolute_peak_allocated_bytes", "absolute_peak_reserved_bytes",
        "persistent_state_bytes", "padding_state_bytes",
    )
    if any(type(row[field]) is not int or row[field] < 0 for field in byte_fields):
        raise ValueError("profile byte fields must be nonnegative integers")
    if (
        row["absolute_peak_allocated_bytes"] < row["resting_allocated_bytes"]
        or row["absolute_peak_reserved_bytes"] < row["resting_reserved_bytes"]
    ):
        raise ValueError("absolute CUDA peaks are below resting memory")
    return {**row, **derived}

def recompute_efficiency_gate(
    full, released, minimum_fps_ratio, maximum_peak_memory_ratio,
):
    required = (
        "episode_fps", "absolute_peak_allocated_bytes",
        "absolute_peak_reserved_bytes",
    )
    full_values = {field: float(full[field]) for field in required}
    released_values = {field: float(released[field]) for field in required}
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (*full_values.values(), *released_values.values())
    ):
        raise ValueError("efficiency sources must be finite and positive")
    ratios = {
        "episode_fps_ratio": full_values["episode_fps"] / released_values["episode_fps"],
        "peak_allocated_ratio": full_values["absolute_peak_allocated_bytes"] / released_values["absolute_peak_allocated_bytes"],
        "peak_reserved_ratio": full_values["absolute_peak_reserved_bytes"] / released_values["absolute_peak_reserved_bytes"],
    }
    if not all(math.isfinite(value) and value > 0.0 for value in ratios.values()):
        raise ValueError("invalid complete-episode efficiency ratio")
    return {
        **ratios,
        "episode_fps_pass": ratios["episode_fps_ratio"] >= minimum_fps_ratio,
        "peak_allocated_pass": (
            ratios["peak_allocated_ratio"] <= maximum_peak_memory_ratio
        ),
        "peak_reserved_pass": (
            ratios["peak_reserved_ratio"] <= maximum_peak_memory_ratio
        ),
    }

def recompute_pair_alignment_lcbs(pair_alignment, registry, expected_cells):
    if (
        int(pair_alignment["bootstrap_replicates"])
            != int(registry["statistics"]["bootstrap_replicates"])
        or float(pair_alignment["one_sided_confidence"])
            != float(registry["admission"]["pair_alignment_lcb_confidence"])
        or int(pair_alignment["bootstrap_seed"])
            != int(registry["statistics"]["bootstrap_seed"])
        or int(pair_alignment["shuffle_seed"])
            != int(registry["controls"]["shuffle_seed"])
    ):
        raise ValueError("pair-alignment parameters differ from frozen registry")
    replicates = int(pair_alignment["bootstrap_replicates"])
    output = {}
    for name in sorted(expected_cells):
        cell = pair_alignment["cells"][name]
        sequence_deltas = cell["sequence_deltas"]
        ids = tuple(sequence_deltas)
        if ids != tuple(sorted(set(ids))) or not ids:
            raise ValueError(f"pair-alignment support is not sorted/unique: {name}")
        values = np.asarray([sequence_deltas[item] for item in ids], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite pair-alignment sequence delta: {name}")
        digest = hashlib.sha256(
            f"{pair_alignment['bootstrap_seed']}:{name}:pair-alignment".encode("utf-8")
        ).digest()
        derived_seed = int.from_bytes(digest[:8], "big")
        if int(cell["derived_bootstrap_seed"]) != derived_seed:
            raise ValueError(f"pair-alignment derived seed mismatch: {name}")
        rng = np.random.default_rng(derived_seed)
        boot = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            indices = rng.integers(0, len(values), size=len(values), endpoint=False)
            boot[replicate] = float(values[indices].mean())
        lower_alpha = 1.0 - float(pair_alignment["one_sided_confidence"])
        if not 0.0 < lower_alpha < 1.0:
            raise ValueError("invalid pair-alignment confidence")
        lcb = float(np.quantile(boot, lower_alpha, method="linear"))
        if not math.isclose(float(cell["lcb"]), lcb, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"stored pair-alignment LCB mismatch: {name}")
        output[name] = lcb
    return output

def recompute_frozen_gate_audit(
    registry, profiles, m1_tables, ddp, condition_sources,
    fit_attempt_manifest, coefficient_selection, fit_feasibility, pair_alignment,
):
    profile_by_key = index_exact_strata(profiles, "profile")
    m1_by_key = index_exact_strata(m1_tables, "M1")
    gates, details = {}, {}

    expected_methods = {
        "released_no_context_legacy", "routing_disabled_legacy",
        "zero_strength_instrumented", "full_four_spectrum",
    }
    expected_profile_frames = int(
        registry["efficiency"]["complete_episode_search_frames"]
    )
    m0_cells, m0_checks = {}, {}
    for key, artifact in profile_by_key.items():
        m0 = recompute_m0_from_raw(
            artifact, expected_profile_frames, expected_methods
        )
        label = f"{key[0]}:{key[1]}"
        m0_checks[label] = m0
        m0_cells[label] = all(m0.values())
    gates["m0"] = all(m0_cells.values())
    details["m0_cells"] = m0_cells
    details["m0_checks"] = m0_checks

    required_cells = tuple(registry["memory"]["required_m1_cells"])
    selected_rank = int(registry["memory"]["selected_rank"])
    energy_min = float(registry["memory"]["trace_energy_threshold"])
    gap_min = float(registry["memory"]["eigengap_relative"])
    m1_cells = {}
    for key, table in m1_by_key.items():
        cells = table.get("cells")
        if not isinstance(cells, dict) or set(cells) != set(required_cells):
            raise ValueError(f"M1 cell-set mismatch for {key}")
        passed = True
        for name in required_cells:
            cell = cells[name]
            numeric = tuple(float(cell[field]) for field in (
                "trace_energy", "relative_gap", "lambda_r", "lambda_r_plus_1",
                "retained_trace", "total_trace",
            ))
            energy, gap, lambda_r, lambda_next, retained_trace, total_trace = numeric
            if (
                not all(math.isfinite(value) for value in numeric)
                or total_trace <= 0.0
                or not 0.0 <= retained_trace <= total_trace
                or not 0.0 <= energy <= 1.0
                or gap < 0.0
                or lambda_r < 0.0
                or lambda_next < 0.0
                or lambda_next > lambda_r
            ):
                raise ValueError(f"nonfinite M1 cell: {key}/{name}")
            recomputed_energy = retained_trace / total_trace
            recomputed_gap = (lambda_r - lambda_next) / max(
                lambda_r, float(registry["memory"]["epsilon"])
            )
            if (
                not math.isclose(energy, recomputed_energy, rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(gap, recomputed_gap, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(f"derived M1 field mismatch: {key}/{name}")
            passed &= (
                int(cell["rank"]) == selected_rank
                and energy >= energy_min and gap >= gap_min
            )
        m1_cells[f"{key[0]}:{key[1]}"] = bool(passed)
    gates["m1"] = all(m1_cells.values())
    details["m1_cells"] = m1_cells

    expected_conditions = {
        (seed, benchmark, condition)
        for seed, benchmark in REQUIRED_GATE_STRATA
        for condition in ("clean", "registered_corruption")
    }
    condition_index = {}
    for source in condition_sources:
        row = source["summary"]
        frames = tuple(source["frames"])
        key = (int(row["base_seed"]), row["benchmark"], row["condition"])
        if key in condition_index or row["method"] != "full_four_spectrum":
            raise ValueError(f"invalid or duplicate condition summary: {key}")
        if row["frames_jsonl_sha256"] != source["verified_frames_jsonl_sha256"]:
            raise ValueError(f"condition summary has wrong frames parent: {key}")
        commits = []
        frame_keys = []
        for frame in frames:
            if type(frame["actual_committed"]) is not bool:
                raise TypeError(f"condition commit must be exact bool: {key}")
            commits.append(frame["actual_committed"])
            frame_keys.append((str(frame["sequence_id"]), int(frame["frame_index"])))
        if frame_keys != sorted(set(frame_keys)) or not frames:
            raise ValueError(f"condition frames are empty/duplicate/unordered: {key}")
        recomputed = {
            "post_initialization_frames": len(frames),
            "actual_commit_frames": int(sum(commits)),
        }
        if any(int(row[field]) != value for field, value in recomputed.items()):
            raise ValueError(f"condition summary count mismatch: {key}")
        condition_index[key] = recomputed
    if set(condition_index) != expected_conditions:
        raise ValueError("condition summaries are incomplete")
    total_frames = sum(int(row["post_initialization_frames"]) for row in condition_index.values())
    total_commits = sum(int(row["actual_commit_frames"]) for row in condition_index.values())
    if total_frames <= 0:
        raise ValueError("coverage denominator is empty")
    coverage_cells = {}
    for seed, benchmark in REQUIRED_GATE_STRATA:
        selected = [condition_index[(seed, benchmark, condition)] for condition in ("clean", "registered_corruption")]
        denominator = sum(int(row["post_initialization_frames"]) for row in selected)
        numerator = sum(int(row["actual_commit_frames"]) for row in selected)
        if denominator <= 0:
            raise ValueError("empty stratum coverage denominator")
        coverage_cells[f"{seed}:{benchmark}"] = numerator / denominator
    details["coverage_overall"] = total_commits / total_frames
    details["coverage_cells"] = coverage_cells
    minimum_write_coverage = float(registry["admission"]["minimum_write_coverage"])
    gates["coverage_overall"] = (
        details["coverage_overall"] >= minimum_write_coverage
    )
    gates["coverage_all_strata"] = (
        min(coverage_cells.values()) >= minimum_write_coverage
    )

    expected_device = registry["freeze"]["hardware"]["profile_cuda_device"]
    efficiency_cells, profile_coverage, persistent_cells = {}, {}, {}
    for key, artifact in profile_by_key.items():
        methods = artifact.get("methods", {})
        if set(methods) != expected_methods:
            raise ValueError(f"profile method set mismatch for {key}")
        pids = {int(row["process_pid"]) for row in methods.values()}
        if len(pids) != 4 or not all(
            strict_bool(row["single_method_process"], "single_method_process")
            for row in methods.values()
        ):
            raise ValueError(f"methods did not run in isolated processes: {key}")
        if artifact["baseline_method"] != "released_no_context_legacy":
            raise ValueError(f"wrong efficiency baseline for {key}")
        sequence_ids = {row["profile_sequence_id"] for row in methods.values()}
        if len(sequence_ids) != 1:
            raise ValueError(f"profile methods used different sequences: {key}")
        recomputed_methods = {}
        for method, row in methods.items():
            if (
                row["method"] != method
                or int(row["base_seed"]) != key[0]
                or row["benchmark"] != key[1]
                or row["profile_sequence_rule"]
                    != registry["efficiency"]["profile_sequence_rule"]
            ):
                raise ValueError(f"profile identity mismatch: {key}/{method}")
            recomputed_methods[method] = recompute_profile_derived(
                row, registry, expected_device
            )
        full = recomputed_methods["full_four_spectrum"]
        released = recomputed_methods["released_no_context_legacy"]
        label = f"{key[0]}:{key[1]}"
        efficiency_cells[label] = recompute_efficiency_gate(
            full, released,
            float(registry["efficiency"]["minimum_active_fps_ratio"]),
            float(registry["efficiency"]["maximum_active_peak_memory_ratio"]),
        )
        profile_coverage[label] = float(full["measured_write_coverage"])
        persistent_cells[label] = int(full["persistent_state_bytes"])
    details["efficiency_cells"] = efficiency_cells
    details["profile_coverage"] = profile_coverage
    details["persistent_state_bytes"] = persistent_cells
    gates["profile_coverage"] = (
        min(profile_coverage.values()) >= minimum_write_coverage
    )
    gates["episode_fps"] = all(item["episode_fps_pass"] for item in efficiency_cells.values())
    gates["peak_allocated"] = all(item["peak_allocated_pass"] for item in efficiency_cells.values())
    gates["peak_reserved"] = all(item["peak_reserved_pass"] for item in efficiency_cells.values())
    gates["persistent_state_8mib"] = max(persistent_cells.values()) <= 8 * 1024 * 1024

    ddp_error = float(ddp["operator_relative_error"])
    if not math.isfinite(ddp_error) or ddp_error < 0.0:
        raise ValueError("DDP operator error must be finite and nonnegative")
    gates["ddp"] = (
        int(ddp["selected_rank"]) == selected_rank
        and int(ddp["world_size"]) == 2
        and ddp["cuda_devices"]
            == registry["freeze"]["hardware"]["ddp_cuda_devices"]
        and strict_bool(ddp["byte_identical_state_hashes"], "ddp state hashes")
        and ddp_error < 1e-4
    )
    coefficient = registry["coefficient"]
    required_successes = int(registry["coefficient_fit"]["required_successful_optimizer_steps"])
    maximum_attempts = int(registry["coefficient_fit"]["maximum_attempted_supersteps"])
    attempts_per_superstep = int(
        registry["coefficient_fit"]["attempted_clips_per_stratum_per_superstep"]
    )
    expected_manifest_attempts = 6 * attempts_per_superstep * maximum_attempts
    fit_completion = (
        int(fit_attempt_manifest["attempt_count"]) == expected_manifest_attempts
        and int(coefficient_selection["fit_completed_successful_steps"]) == required_successes
        and int(coefficient_selection["fit_completed_attempted_supersteps"]) <= maximum_attempts
        and int(coefficient_selection["fit_completed_attempted_supersteps"]) >= required_successes
        and int(coefficient_selection["fit_completed_successful_steps"])
            == int(coefficient["fit_completed_successful_steps"])
        and int(coefficient_selection["fit_completed_attempted_supersteps"])
            == int(coefficient["fit_completed_attempted_supersteps"])
        and int(coefficient_selection["selected_successful_step"])
            == int(coefficient["selected_successful_step"])
        and int(coefficient_selection["selected_checkpoint_attempted_supersteps"])
            == int(coefficient["selected_checkpoint_attempted_supersteps"])
        and coefficient_selection["selected_checkpoint_sha256"]
            == coefficient["checkpoint_sha256"]
        and coefficient_selection["attempt_manifest_sha256"]
            == coefficient["attempt_manifest_sha256"]
        and coefficient_selection["initial_feasibility_sha256"]
            == coefficient["initial_feasibility_sha256"]
    )
    gates["fit_completion"] = bool(fit_completion)
    details["fit_completion"] = {
        "manifest_attempt_count": int(fit_attempt_manifest["attempt_count"]),
        "completed_successful_steps": int(
            coefficient_selection["fit_completed_successful_steps"]
        ),
        "completed_attempted_supersteps": int(
            coefficient_selection["fit_completed_attempted_supersteps"]
        ),
    }

    feasibility_threshold = float(registry["coefficient_fit"]["minimum_family_active_coverage"])
    route_threshold = float(registry["coefficient_fit"]["minimum_effective_route_coverage"])
    sum_tolerance = float(registry["coefficient_fit"]["tangent_gradient_sum_tolerance"])
    minimum_singular = float(
        registry["coefficient_fit"]["minimum_tangent_rms_singular_value"]
    )
    minimum_relative = float(
        registry["coefficient_fit"]["minimum_tangent_relative_singular_value"]
    )
    feasibility_cells = {}
    expected_base_keys = {
        f"seed{seed}:{modality}"
        for seed in range(3) for modality in ("rgbd", "rgbt")
    }
    expected_route_cells = {
        f"{base}|{block}:{site}:{family}"
        for base in expected_base_keys
        for block, site in ((5, "attn"), (5, "ffn"), (9, "attn"), (9, "ffn"))
        for family in ("identity", "dynamic", "private", "background")
    }
    for label in ("initial", "selected"):
        report = fit_feasibility[label]
        u_initial = np.asarray(registry["coefficient_fit"]["u_initial"], dtype=np.float64)
        exp_u = np.exp(u_initial - u_initial.max())
        expected_alpha = (
            float(registry["routing"]["alpha_budget"]) * exp_u / exp_u.sum()
            if label == "initial"
            else np.asarray(registry["coefficient"]["alpha"], dtype=np.float64)
        )
        if (
            report["attempt_manifest_sha256"] != coefficient["attempt_manifest_sha256"]
            or int(report["attempts_per_stratum"]) != int(
                registry["coefficient_fit"]["feasibility_attempts_per_stratum"]
            )
            or set(report["base_keys"]) != expected_base_keys
            or set(report["active_coverage"]) != expected_route_cells
            or set(report["effective_route_coverage"]) != expected_route_cells
            or set(report["tangent_rms_singular_values"]) != expected_base_keys
            or set(report["tangent_relative_min"]) != expected_base_keys
            or set(report["tangent_valid_attempts"]) != expected_base_keys
            or set(report["signed_leaf_alpha_gradients"]) != expected_base_keys
            or set(report["leaf_alpha_abs_mean_diagnostic"]) != expected_base_keys
            or not np.allclose(
                np.asarray(report["frozen_alpha"], dtype=np.float64),
                expected_alpha, rtol=0.0, atol=1e-12,
            )
        ):
            raise ValueError("fit feasibility attempt manifest mismatch")
        if label == "selected" and (
            report["coefficient_selection_sha256"] != coefficient["selection_sha256"]
            or report["selected_checkpoint_sha256"] != coefficient["checkpoint_sha256"]
        ):
            raise ValueError("selected feasibility has wrong frozen parent")
        coverages = tuple(float(value) for value in report["active_coverage"].values())
        route_coverages = tuple(
            float(value) for value in report["effective_route_coverage"].values()
        )
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (*coverages, *route_coverages)
        ):
            raise ValueError("feasibility coverage is outside [0,1]")
        tangent_pass = True
        for base_key in sorted(expected_base_keys):
            summary = recompute_tangent_summary(
                report["signed_leaf_alpha_gradients"][base_key],
                expected_alpha, sum_tolerance,
            )
            stored_singular = np.asarray(
                report["tangent_rms_singular_values"][base_key],
                dtype=np.float64,
            )
            stored_leaf = np.asarray(
                report["leaf_alpha_abs_mean_diagnostic"][base_key],
                dtype=np.float64,
            )
            if (
                not np.allclose(stored_singular, summary["singular"], rtol=1e-12, atol=1e-12)
                or not np.allclose(stored_leaf, summary["leaf_abs_mean"], rtol=1e-12, atol=1e-12)
                or not math.isclose(
                    float(report["tangent_relative_min"][base_key]),
                    summary["relative_min"], rel_tol=1e-12, abs_tol=1e-12,
                )
                or int(report["tangent_valid_attempts"][base_key])
                    != summary["valid_attempts"]
            ):
                raise ValueError(f"derived tangent report mismatch: {label}/{base_key}")
            tangent_pass &= (
                float(summary["singular"][-1]) >= minimum_singular
                and summary["relative_min"] >= minimum_relative
            )
        passed = (
            min(coverages) >= feasibility_threshold
            and min(route_coverages) >= route_threshold
            and tangent_pass
        )
        feasibility_cells[label] = bool(passed)
    details["fit_feasibility"] = feasibility_cells
    gates["fit_feasibility"] = all(feasibility_cells.values())

    expected_alignment_cells = {
        f"seed{seed}|{benchmark}|{block}:{site}:{scope}"
        for seed in range(3)
        for benchmark in ("lasher", "depthtrack")
        for block in (5, 9)
        for site in ("attn", "ffn")
        for scope in ("template", "search")
    }
    alignment_cells = pair_alignment["cells"]
    if set(alignment_cells) != expected_alignment_cells:
        raise ValueError("pair-alignment audit requires 48 exact named cells")
    alignment_lcbs = recompute_pair_alignment_lcbs(
        pair_alignment, registry, expected_alignment_cells
    )
    details["pair_alignment_lcbs"] = alignment_lcbs
    gates["pair_alignment"] = min(alignment_lcbs.values()) > float(
        registry["admission"]["pair_alignment_min_lcb"]
    )
    return {"computed_gates": gates, "details": details}

def validate_and_recompute_frozen_audit(
    stored_audit, verified_source_hashes, **raw_inputs,
):
    if stored_audit.get("source_sha256") != verified_source_hashes:
        raise ValueError("frozen-audit source hash mismatch")
    coefficient = raw_inputs["registry"]["coefficient"]
    required_parent_hashes = {
        "coefficient_selection": coefficient["selection_sha256"],
        "fit_attempt_manifest": coefficient["attempt_manifest_sha256"],
        "fit_feasibility_initial": coefficient["initial_feasibility_sha256"],
        "fit_feasibility_selected": coefficient["selected_feasibility_sha256"],
        "pair_alignment_audit": raw_inputs["registry"]["admission"][
            "pair_alignment_audit_sha256"
        ],
    }
    for label, expected in required_parent_hashes.items():
        if verified_source_hashes.get(label) != expected:
            raise ValueError(f"frozen coefficient parent hash mismatch: {label}")
    rebuilt = recompute_frozen_gate_audit(**raw_inputs)
    if (
        stored_audit.get("computed_gates") != rebuilt["computed_gates"]
        or stored_audit.get("details") != rebuilt["details"]
    ):
        raise ValueError("stored audit differs from source recomputation")
    rebuilt["all_gates_passed_computed"] = all(rebuilt["computed_gates"].values())
    return rebuilt
~~~

The CLI computes `sha256_file(path)` from every supplied artifact's bytes before parsing, compares it with the frozen input index, and never trusts a self-reported JSON hash. It accepts exactly six four-child-process profiles with their canonical 60-frame prediction bytes, phase forward counts, pre/post parameter hashes, and raw timing/commit vectors; six clean full-row M1 tables with raw trace terms; one frozen-selected-rank two-process artifact; twelve full-row condition summaries plus their hashed `frames.jsonl` sources; the coefficient-selection record; the fixed attempt manifest; both fixed-attempt feasibility reports with signed leaf-gradient matrices; and the 48-cell pair-alignment audit with raw per-sequence deltas. The profiler and analyzer share the one exact `REQUIRED_PROFILE_FIELDS` constant; the analyzer requires exact top-level and method-row schemas and also recursively rejects producer-supplied M0 conclusion keys at any profile-artifact depth before recomputing every M0 identity/count/hash gate, profile FPS/percentile/coverage, condition count, tangent singular value, M1 energy/gap, and pair-alignment LCB. `--build-frozen-gate-audit` writes `{source_sha256, computed_gates, details}` from `recompute_frozen_gate_audit()` and contains no caller-supplied `pass` field. Final analysis rehashes every constituent, reloads the raw sources, calls `validate_and_recompute_frozen_audit()`, and passes only its freshly computed `all_gates_passed_computed` boolean into `evaluate_s0_gate()`. Tests inject an arbitrary unknown field plus `m0`, `m0_checks`, `m0_cells`, `forward_counts_match_protocol`, `pass`, `m0_pass`, `bitwise_*`, `*_equals_*`, `*_matches_*`, `*_unchanged`, `*_pass`, `*_passed`, `checks`, `computed_gates`, and `gate_decision` keys at the top level, method-row level, and one nested mapping and require rejection; they also mutate raw prediction bytes, phase counts, pre/post parameter hashes, and raw/derived M1/coverage/FPS/allocated/reserved/state/DDP/hardware/batch/attempt-count/completed-step/tangent-feasibility/alignment fields one at a time and require fail-closed rejection, including a stored all-true map backed by failing raw sources or a summary whose `frames.jsonl` parent/count differs.

- [ ] **Step 4: Run the complete unit and smoke gate**

~~~bash
.venv/bin/python -m unittest discover -s tests -v
repo="$PWD"
.venv/bin/python tools/profile_spectral_s0.py --help >/dev/null
(cd /tmp && "$repo/.venv/bin/python" \
  "$repo/tools/profile_spectral_s0.py" --help >/dev/null)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/profile_spectral_s0.py \
  --config rgbt_spectral_s0_short \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --base-seed 0 \
  --coefficient-checkpoint output/spectral_s0/smoke/global_coefficients.pt \
  --schedule output/spectral_s0/smoke/schedule.jsonl \
  --orchestrate-methods released_no_context_legacy,routing_disabled_legacy,zero_strength_instrumented,full_four_spectrum \
  --warmup 10 --iterations 50 \
  --output output/spectral_s0/smoke/profile.json
~~~

Expected: all tests PASS; released no-context, routing-disabled observer, and zero-strength predictions are bitwise identical; model hash unchanged; finite diagnostics; byte/latency schema complete; ratios name `released_no_context_legacy` as denominator.

- [ ] **Step 5: Run the two-process M0/M1 operator check**

~~~bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  tools/profile_spectral_s0.py \
  --config rgbt_spectral_s0_short \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --base-seed 0 \
  --coefficient-checkpoint output/spectral_s0/smoke/global_coefficients.pt \
  --schedule output/spectral_s0/smoke/schedule.jsonl \
  --ddp-operator-check \
  --output output/spectral_s0/smoke/ddp_operator.json
~~~

Expected: deterministic cross-rank factor merge, matching state hashes, and single/DDP operator relative error below `1e-4`.

- [ ] **Step 6: Commit profiler code before any frozen provenance is created**

~~~bash
git add tools/profile_spectral_s0.py tests/test_profile_spectral_s0.py \
  tracking/analyze_spectral_s0.py tests/test_spectral_s0_evaluation.py
git commit -m "test: add frozen spectral stage zero profiler audit"
~~~

- [ ] **Step 7: Select the calibration admission rule, paired floor, and common rank**

First record routing-disabled observables across all six base checkpoints without admitting any write. Select/freeze the admission threshold and paired floor, materialize six calibration schedules, and only then collect rank-32 sketches:

~~~bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/record_spectral_schedule.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --all-base-seeds --split spectral_calibration \
  --purpose calibration-observables \
  --output output/spectral_s0/calibration/legacy_observables

.venv/bin/python tracking/analyze_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --select-admission \
  --observables output/spectral_s0/calibration/legacy_observables \
  --locked-admission output/spectral_s0/calibration/locked_admission.json \
  --schedule-dir output/spectral_s0/calibration/schedules

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/run_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --all-base-seeds --split spectral_calibration \
  --methods routing_disabled_legacy --collect-max-rank 32 --collect-pair-alignment \
  --locked-admission output/spectral_s0/calibration/locked_admission.json \
  --schedule-dir output/spectral_s0/calibration/schedules \
  --output output/spectral_s0/calibration/rank_sketches

.venv/bin/python tracking/analyze_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --select-rank \
  --locked-admission output/spectral_s0/calibration/locked_admission.json \
  --rank-sketches output/spectral_s0/calibration/rank_sketches \
  --pair-alignment-output output/spectral_s0/calibration/pair_alignment_audit.json \
  --output output/spectral_s0/calibration/locked_calibration.json
~~~

Expected: coverage candidate is selected by the preregistered false-admission rule; realized post-`paired_valid` coverage is at least 0.20 overall and in every base/benchmark stratum with recorded denominators; paired attention floor and alignment audit pass; and the smallest common rank in `8,16,32` reaches 90% exact total-trace energy plus the registered boundary gap for every family/key. Gate mode later fails if any locked-rank M1 check falls below these values; it never changes rank.

- [ ] **Step 8: Validate evaluators before first `J_core`, then fit/select coefficients**

~~~bash
: "${LASHER_AUTHOR_ARCHIVE:?set absolute author-archive path}"
: "${MATLAB_EXECUTABLE:?set absolute MATLAB executable path}"
test -f "$LASHER_AUTHOR_ARCHIVE"
test -x "$MATLAB_EXECUTABLE"
"$MATLAB_EXECUTABLE" -batch "disp(version)"
.venv/bin/python tools/validate_benchmark_evaluators.py \
  --lasher-author-archive "$LASHER_AUTHOR_ARCHIVE" \
  --matlab-executable "$MATLAB_EXECUTABLE" \
  --split-audit lib/train/data_specs/target_spectral_split_audit.json \
  --output output/spectral_s0/calibration/evaluator_validation.json

.venv/bin/python tracking/fit_spectral_coefficients.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 --split spectral_fit \
  --fit-seed 20260713 --attempts-per-stratum 2 \
  --maximum-attempted-supersteps 2000 --write-attempt-manifest-only \
  --output output/spectral_s0/calibration/fit_attempt_manifest.json

.venv/bin/python tracking/fit_spectral_coefficients.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --attempt-manifest output/spectral_s0/calibration/fit_attempt_manifest.json \
  --engineering-audit output/spectral_s0/smoke/fit_engineering_smoke.json \
  --resource-preflight-only --minimum-free-disk-multiplier 2.0 \
  --output output/spectral_s0/calibration/fit_resource_preflight.json

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/fit_spectral_coefficients.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --locked-calibration output/spectral_s0/calibration/locked_calibration.json \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 \
  --split spectral_fit --fit-seed 20260713 \
  --attempts-per-stratum 2 --maximum-attempted-supersteps 2000 \
  --required-successful-steps 1000 \
  --checkpoint-successful-steps 0,100,200,300,400,500,600,700,800,900,1000 \
  --attempt-manifest output/spectral_s0/calibration/fit_attempt_manifest.json \
  --resource-preflight output/spectral_s0/calibration/fit_resource_preflight.json \
  --initial-feasibility output/spectral_s0/calibration/fit_feasibility_initial.json \
  --output-dir output/spectral_s0/calibration/coefficients

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/run_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --locked-calibration output/spectral_s0/calibration/locked_calibration.json \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 --all-base-seeds \
  --split spectral_calibration --methods full_four_spectrum \
  --coefficient-candidates output/spectral_s0/calibration/coefficients \
  --schedule-dir output/spectral_s0/calibration/schedules \
  --lasher-author-archive "$LASHER_AUTHOR_ARCHIVE" \
  --matlab-executable "$MATLAB_EXECUTABLE" \
  --evaluator-validation output/spectral_s0/calibration/evaluator_validation.json \
  --output output/spectral_s0/calibration/coefficient_selection_rows

.venv/bin/python tracking/analyze_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --select-coefficient-step \
  --input output/spectral_s0/calibration/coefficient_selection_rows \
  --checkpoint-dir output/spectral_s0/calibration/coefficients \
  --evaluator-validation output/spectral_s0/calibration/evaluator_validation.json \
  --selected-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
  --output output/spectral_s0/calibration/coefficient_selection.json

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/fit_spectral_coefficients.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --locked-calibration output/spectral_s0/calibration/locked_calibration.json \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 --split spectral_fit \
  --attempt-manifest output/spectral_s0/calibration/fit_attempt_manifest.json \
  --audit-selected-feasibility \
  --coefficient-selection output/spectral_s0/calibration/coefficient_selection.json \
  --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
  --output output/spectral_s0/calibration/fit_feasibility_selected.json
~~~

Evaluator validation must finish before any candidate `J_core` is computed. Manifest-only mode loads sequence IDs/lengths but performs no model forward and writes all 24,000 fixed attempt records before fitting. Resource preflight verifies that manifest byte hash/count, extrapolates total prefix/inner/outer forwards, wall time, and audit/checkpoint bytes from the registered 12-clip engineering audit, records current free disk, and fails if free disk is below twice the projected artifact bytes; its estimates are capacity planning, not performance evidence and never inspect labels, activity, losses, or outcomes. The full fit requires matching manifest/preflight hashes and cannot regenerate either artifact. Both candidate rows and selection JSON record evaluator-validation SHA-256 and reject a mismatch. Selection validates that candidates equal the eleven preregistered successful steps, that the fit completed 1,000 successful supersteps within at most 2,000 manifested attempts, maximizes equal-weight `J_core` across all six bases, and resolves exact ties to the earliest successful step. The selection JSON separately records `selected_successful_step`, its candidate SHA-256 and attempted-superstep counter, plus the completed-fit successful/attempted counters; `global_coefficients.pt` is a byte-identical copy whose hash must match. The repeated feasibility command reuses the exact fixed first 100 attempts per stratum and writes a child hash for the frozen registry. Failure stops; it does not change `u`, choose another checkpoint, alter rank/admission/attempt budgets, or resample the candidate set.

- [ ] **Step 9: Select corruption severity using the validated evaluators**

~~~bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/run_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --locked-calibration output/spectral_s0/calibration/locked_calibration.json \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --configs rgbt_spectral_s0,rgbd_spectral_s0 --all-base-seeds \
  --split spectral_calibration --methods routing_disabled_legacy \
  --corruption-severities 0.50,0.75,1.00 \
  --lasher-author-archive "$LASHER_AUTHOR_ARCHIVE" \
  --matlab-executable "$MATLAB_EXECUTABLE" \
  --evaluator-validation output/spectral_s0/calibration/evaluator_validation.json \
  --output output/spectral_s0/calibration/corruption_selection_rows

.venv/bin/python tracking/analyze_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --select-corruption-severity \
  --input output/spectral_s0/calibration/corruption_selection_rows \
  --evaluator-validation output/spectral_s0/calibration/evaluator_validation.json \
  --output output/spectral_s0/calibration/corruption_selection.json
~~~

For each severity, first compute raw unit-interval `drop_raw = J_core_clean_raw - J_core_corrupt_raw`, then call the same `raw_contrast_to_percentage_points(drop_raw)` exactly once and choose the `drop_pp` closest to `10.0`; exact ties choose the lower severity. Store both values/unit tags and reject a negative or already-pp input. Do not compare raw `0.10` directly with `10.0`. The selection refuses any evaluator-validation hash other than the one already used for coefficient selection. Validation must have passed archive/entry/tree checks and every fixture and must explicitly identify the DepthTrack adapter as paper-faithful, not official-toolkit output.

- [ ] **Step 10: Freeze without opening gate confirmation**

~~~bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python tools/profile_spectral_s0.py \
  --capture-hardware --profile-device-index 0 --require-ddp-devices 2 \
  --batch-size 1 --timing-protocol synchronized_complete_episode_v1 \
  --output output/spectral_s0/calibration/frozen_hardware.json

.venv/bin/python tools/freeze_spectral_s0_registry.py \
  --calibration-registry experiments/seatrack/registries/spectral_s0_v1.calibration.yaml \
  --locked-calibration output/spectral_s0/calibration/locked_calibration.json \
  --coefficient-selection output/spectral_s0/calibration/coefficient_selection.json \
  --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
  --fit-attempt-manifest output/spectral_s0/calibration/fit_attempt_manifest.json \
  --fit-feasibility-initial output/spectral_s0/calibration/fit_feasibility_initial.json \
  --fit-feasibility-selected output/spectral_s0/calibration/fit_feasibility_selected.json \
  --pair-alignment-audit output/spectral_s0/calibration/pair_alignment_audit.json \
  --corruption-selection output/spectral_s0/calibration/corruption_selection.json \
  --evaluator-validation output/spectral_s0/calibration/evaluator_validation.json \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --hardware-json output/spectral_s0/calibration/frozen_hardware.json \
  --output experiments/seatrack/registries/spectral_s0_v1.frozen.yaml
.venv/bin/python -m unittest tests.test_spectral_config_registry -v
~~~

Expected: the hardware artifact contains exactly one batch-one profile fingerprint and two ordered, distinct DDP fingerprints with exact name/UUID/PCI bus ID/compute capability/total-memory/driver fields; the profile fingerprint equals ordered DDP device zero. The frozen registry validates, contains no unresolved fields, and all referenced files match their hashes.

- [ ] **Step 11: Commit the frozen registry before gate confirmation**

~~~bash
git add experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
  knowledge_base/Target-Spectral-S0-实验记录.md
git commit -m "chore: freeze spectral s0 calibration registry"
~~~

- [ ] **Step 12: Generate and seal the twelve legacy gate schedules**

~~~bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/record_spectral_schedule.py \
    --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
    --config rgbt_spectral_s0 \
    --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
    --base-seed "$seed" --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
    --dataset lasher --split spectral_gate_confirmation \
    --output "output/spectral_s0/gate/schedules/seed_${seed}_lasher.jsonl"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/record_spectral_schedule.py \
    --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
    --config rgbt_spectral_s0 \
    --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
    --base-seed "$seed" --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
    --dataset lasher --split spectral_gate_confirmation --registered-corruption \
    --output "output/spectral_s0/gate/schedules/seed_${seed}_lasher_corrupt.jsonl"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/record_spectral_schedule.py \
    --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
    --config rgbd_spectral_s0 \
    --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
    --base-seed "$seed" --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
    --dataset depthtrack --split spectral_gate_confirmation \
    --output "output/spectral_s0/gate/schedules/seed_${seed}_depthtrack.jsonl"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/record_spectral_schedule.py \
    --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
    --config rgbd_spectral_s0 \
    --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
    --base-seed "$seed" --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
    --dataset depthtrack --split spectral_gate_confirmation --registered-corruption \
    --output "output/spectral_s0/gate/schedules/seed_${seed}_depthtrack_corrupt.jsonl"
done

.venv/bin/python tracking/record_spectral_schedule.py \
  --seal-manifest output/spectral_s0/gate/schedules \
  --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
  --output experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json
~~~

The manifest contains exactly twelve canonical schedule hashes (clean and registered-corruption for each of six base/dataset pairs) plus the frozen-registry hash and is immutable. Each compared row selects the schedule matching its condition. Do not rewrite the frozen registry.

- [ ] **Step 13: Commit the schedule manifest before compared rows**

~~~bash
git add experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json
git commit -m "chore: seal spectral s0 gate schedules"
~~~

- [ ] **Step 14: Run locked S0 controls on all three base-checkpoint seeds**

For both LasHeR and DepthTrack gate-confirmation manifests, run all required core, branch, leave-one-out, cumulative, and shuffle rows using only the frozen registry and committed schedule manifest:

~~~bash
: "${LASHER_AUTHOR_ARCHIVE:?set absolute author-archive path}"
: "${MATLAB_EXECUTABLE:?set absolute MATLAB executable path}"
METHODS=routing_disabled_legacy,zero_strength_instrumented,confidence_only_scalar_history,random_orthogonal,pooled_same,target_balanced_identity,full_four_spectrum,rgbx_pair_shuffle,temporal_order_shuffle,target_background_mask_shuffle,identity_only,private_only,dynamic_only,background_only,full_minus_identity,full_minus_private,full_minus_dynamic,full_minus_background,full_minus_identity_strength_matched,full_minus_private_strength_matched,full_minus_dynamic_strength_matched,full_minus_background_strength_matched,identity_plus_dynamic,identity_plus_dynamic_plus_private
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/run_spectral_s0.py \
    --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
    --schedule-manifest experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json \
    --config rgbt_spectral_s0 --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
    --base-seed "$seed" --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
    --dataset lasher --split spectral_gate_confirmation --methods "$METHODS" \
    --lasher-author-archive "$LASHER_AUTHOR_ARCHIVE" \
    --matlab-executable "$MATLAB_EXECUTABLE" \
    --include-clean-and-registered-corruption \
    --output "output/spectral_s0/gate/seed_${seed}/lasher"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tracking/run_spectral_s0.py \
    --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
    --schedule-manifest experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json \
    --config rgbd_spectral_s0 --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
    --base-seed "$seed" --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
    --dataset depthtrack --split spectral_gate_confirmation --methods "$METHODS" \
    --include-clean-and-registered-corruption \
    --output "output/spectral_s0/gate/seed_${seed}/depthtrack"
done

for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/profile_spectral_s0.py \
    --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
    --schedule-manifest experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json \
    --config rgbt_spectral_s0 --dataset lasher --split spectral_gate_confirmation \
    --condition clean --profile-sequence-rule sha256_min_clean_eligible \
    --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
    --base-seed "$seed" --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
    --orchestrate-methods released_no_context_legacy,routing_disabled_legacy,zero_strength_instrumented,full_four_spectrum \
    --warmup 10 --iterations 50 \
    --output "output/spectral_s0/gate/audits/profile_seed_${seed}_lasher.json"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/profile_spectral_s0.py \
    --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
    --schedule-manifest experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json \
    --config rgbd_spectral_s0 --dataset depthtrack --split spectral_gate_confirmation \
    --condition clean --profile-sequence-rule sha256_min_clean_eligible \
    --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
    --base-seed "$seed" --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
    --orchestrate-methods released_no_context_legacy,routing_disabled_legacy,zero_strength_instrumented,full_four_spectrum \
    --warmup 10 --iterations 50 \
    --output "output/spectral_s0/gate/audits/profile_seed_${seed}_depthtrack.json"
done

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  tools/profile_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
  --schedule-manifest experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json \
  --config rgbt_spectral_s0 --dataset lasher --split spectral_gate_confirmation \
  --condition clean --profile-sequence-rule sha256_min_clean_eligible \
  --base-checkpoint-index output/spectral_s0/base/base_checkpoints.json \
  --base-seed 0 --coefficient-checkpoint output/spectral_s0/calibration/global_coefficients.pt \
  --ddp-operator-check --use-frozen-selected-rank \
  --output output/spectral_s0/gate/audits/frozen_selected_rank_ddp.json

.venv/bin/python tracking/analyze_spectral_s0.py \
  --build-frozen-gate-audit \
  --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
  --schedule-manifest experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json \
  --input output/spectral_s0/gate \
  --profiles output/spectral_s0/gate/audits \
  --ddp-audit output/spectral_s0/gate/audits/frozen_selected_rank_ddp.json \
  --coefficient-selection output/spectral_s0/calibration/coefficient_selection.json \
  --fit-attempt-manifest output/spectral_s0/calibration/fit_attempt_manifest.json \
  --fit-feasibility-initial output/spectral_s0/calibration/fit_feasibility_initial.json \
  --fit-feasibility-selected output/spectral_s0/calibration/fit_feasibility_selected.json \
  --pair-alignment-audit output/spectral_s0/calibration/pair_alignment_audit.json \
  --output output/spectral_s0/gate/audits/frozen_gate_audit.json

.venv/bin/python tracking/analyze_spectral_s0.py \
  --registry experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
  --schedule-manifest experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json \
  --evaluator-validation output/spectral_s0/calibration/evaluator_validation.json \
  --frozen-gate-audit output/spectral_s0/gate/audits/frozen_gate_audit.json \
  --profiles output/spectral_s0/gate/audits \
  --ddp-audit output/spectral_s0/gate/audits/frozen_selected_rank_ddp.json \
  --coefficient-selection output/spectral_s0/calibration/coefficient_selection.json \
  --fit-attempt-manifest output/spectral_s0/calibration/fit_attempt_manifest.json \
  --fit-feasibility-initial output/spectral_s0/calibration/fit_feasibility_initial.json \
  --fit-feasibility-selected output/spectral_s0/calibration/fit_feasibility_selected.json \
  --pair-alignment-audit output/spectral_s0/calibration/pair_alignment_audit.json \
  --input output/spectral_s0/gate \
  --bootstrap-plan-output output/spectral_s0/gate/audits/crossed_bootstrap_plan.json \
  --output output/spectral_s0/gate/s0_gate.json \
  --evidence-copy knowledge_base/Target-Spectral-S0-gate.json
~~~

Each runner invokes the registered benchmark-metric adapter separately for `clean` and `registered_corruption` and stores each parsed aggregate in its own condition directory. LasHeR revalidates the supplied archive SHA/tree and MATLAB executable/version against the frozen registry before either condition; DepthTrack identifies its adapter as paper-faithful. Every result/evidence file records the committed schedule-manifest hash and its condition-specific schedule hash. The six profiles use the registry's SHA256-min eligible clean sequence rule; missing 60-frame eligibility fails rather than changing the rule. `--build-frozen-gate-audit` requires exactly one profile and one clean full-row locked-rank table for each of the six `(base_seed,benchmark)` strata, verifies all parent hashes, and hashes every source artifact.

For coverage, define `post_initialization_frames` as every frame after legal initialization in both conditions, whether valid/admitted or not. Define `actual_commit_frames = sum(int(frame.actual_committed))` for `full_four_spectrum`; never sum key/family counts. The audit requires `actual_commit_frames/post_initialization_frames >= 0.20` overall and independently in all six `(base_seed,benchmark)` strata after pooling their two predeclared conditions; these binding full-stream rates always come from complete gate rows, never the profile subset. It also reports nonbinding condition-specific full-stream rates. Separately, each profile's fixed measured region must meet 0.20 actual-commit coverage so the reported active latency is observed under headline write activity; this applicability check cannot replace the full-stream gate.

The registered gate hardware must expose two CUDA devices for this formal check; absence blocks the gate rather than accepting the earlier smoke. The frozen DDP artifact must hash the frozen registry/schedule manifest/base checkpoint/coefficient, state the selected rank from the registry, cover every key/source family, show byte-identical post-broadcast state hashes, and show single-process/DDP operator relative error below `1e-4`; calibration/provisional-rank DDP output is rejected. The audit also requires every clean locked-rank table cell to meet energy/gap, every profile to pass M0 and the frozen 0.80/1.25 efficiency bounds against `released_no_context_legacy`, and persistent state to stay within 8 MiB. The final analyzer refuses absent/failed audit fields, a source-hash mismatch, a missing condition, a metric outside its condition directory, or any attempt to combine prediction files across conditions. `s0_gate.json` and its evidence copy store the audit SHA-256 plus every constituent hash. Store every command and artifact hash in the evidence record.

Expected artifacts:

~~~text
output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/clean/frames.jsonl
output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/clean/sequences.json
output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/clean/benchmark_metrics.json
output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/clean/locked_rank_m1.json
output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/registered_corruption/frames.jsonl
output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/registered_corruption/sequences.json
output/spectral_s0/gate/seed_0/lasher/full_four_spectrum/registered_corruption/benchmark_metrics.json
output/spectral_s0/gate/audits/profile_seed_0_lasher.json
output/spectral_s0/gate/audits/frozen_selected_rank_ddp.json
output/spectral_s0/gate/audits/frozen_gate_audit.json
output/spectral_s0/gate/audits/crossed_bootstrap_plan.json
output/spectral_s0/gate/s0_gate.json
~~~

The three core artifacts per condition exist for seeds 1/2 and every listed dataset/method; every clean `full_four_spectrum` row additionally has `locked_rank_m1.json`.

- [ ] **Step 15: Apply the stop/go rule**

Proceed to a separate Workstream B design review only if the final analyzer consumed a hash-matching `frozen_gate_audit.json` and:

- M0 is freshly rebuilt from canonical prediction bytes, phase forward counts, and pre/post model-parameter hashes, and both M0 and M1 pass;
- the frozen attempt manifest is unchanged, every attempted superstep processed all twelve exact manifested clips before deciding success/failure, both the pre-fit and selected-checkpoint real-data feasibility audits pass every registered stratum/key/family coverage threshold and both absolute/relative three-dimensional simplex-tangent singular-value floors, and 1,000 successful supersteps were obtained within the ceiling of 2,000 attempted supersteps;
- the frozen-selected-rank two-process DDP artifact passes byte-hash and `<1e-4` operator-equivalence checks and is included in the audit hash;
- the locked-rank exact-trace/eigengap checks pass for every persistent key/family and the joint-family matched-vs-shuffled pair-alignment audit passes every required base/benchmark/key cell;
- one S0 co-primary effect gate passes;
- LasHeR and DepthTrack each have strict clean noninferiority LCB above `-0.3` pp against both `confidence_only_scalar_history` and `routing_disabled_legacy` under the one frozen crossed-bootstrap plan;
- both endpoints contribute all ten contrasts to the one 20-member attribution correction family, the endpoint is selected only by the registered J-core-first co-primary rule, and on that selected passing endpoint full routing has a strict positive simultaneous LCB versus `random_orthogonal`, `pooled_same`, and `target_balanced_identity`;
- on that same endpoint, full routing has a strict positive simultaneous LCB versus each of the four strength-matched leave-one-family-out rows; raw leave-one-out rows remain diagnostic only;
- on that same endpoint, pair, temporal, and mask shuffles each have a strict positive simultaneous LCB for `full - 2 * shuffle`, so every shuffle attenuates to less than half of the signed full gain;
- persistent spectral state is at most 8 MiB;
- `full_four_spectrum` frame-level actual-commit coverage is at least `0.20` overall and in each of the six `(base_seed,benchmark)` strata;
- each fixed 50-frame efficiency profile also observes at least `0.20` full-row actual-commit coverage in its measured region;
- active frozen Stage 0 complete-episode FPS, including initialization and all 60 registered search frames, is at least 80% of `released_no_context_legacy`; both absolute peak allocated and absolute peak reserved GPU memory are independently at most 1.25 times that true no-context baseline on the exact frozen batch-one profile CUDA fingerprint, while the two-process check uses the exact two ordered frozen CUDA fingerprints; observer-instrumented `routing_disabled_legacy` is reported separately;
- all budgets/hashes/schedules match, and a fresh recomputation from the raw named sources produces `all_gates_passed_computed=true` without trusting a stored pass flag.

If any condition fails, record the failed estimand/control and stop. Do not introduce an optimizer or tune on gate-confirmation results.

- [ ] **Step 16: Commit the evidence digest and verify clean scope**

~~~bash
git add knowledge_base/Target-Spectral-S0-实验记录.md \
  knowledge_base/Target-Spectral-S0-gate.json
git commit -m "test: verify spectral workstream a gates"
git diff --check HEAD~1 HEAD
git ls-files --error-unmatch \
  experiments/seatrack/registries/spectral_s0_v1.frozen.yaml \
  experiments/seatrack/registries/spectral_s0_v1.gate_schedule_manifest.json \
  tools/profile_spectral_s0.py knowledge_base/Target-Spectral-S0-gate.json
git status --short --branch
~~~

Expected: no whitespace errors; only intended Workstream A artifacts are committed; no Stage R/E implementation exists.

---

## Final Verification Checklist

- [ ] Decision 7 has explicit user ratification before Task 0 writes the normative addendum or any implementation source/config is edited.
- [ ] `.venv/bin/python -m unittest discover -s tests -v` passes, including the original 40 tests.
- [ ] Disabled, empty-state, and zero-strength routes are bitwise legacy-identical in evaluation mode.
- [ ] The final M0 gate derives routing-disabled/zero-strength identity, exact disabled/active forward counts, and parameter immutability from canonical raw prediction bytes, phase counts, and pre/post hashes; no producer-supplied M0 conclusion boolean is accepted.
- [ ] A disabled tracker constructs no Stage 0 controller, performs no x0/initialization network forward, emits no spectral diagnostics, and takes the exact one-forward legacy track path.
- [ ] Observer-only anchor capture is output-identical and produces only detached immutable anchor factors.
- [ ] Routed logits may change; expert inputs are computed from raw `H`.
- [ ] Template routing is legacy and search routing is restricted to blocks 5/9 attention+FFN.
- [ ] Every real frame uses one pre-frame snapshot; writes become visible only next frame.
- [ ] Snapshot confidence/asymmetry update only on actual commits; scheduled admits, factor rejections, and commits are counted separately.
- [ ] No post-init label or future field reaches `track()`.
- [ ] Ordered optimization clips use frame 0 as anchor and `t-2..t+1`; the image-only prefix `1..t-3` is replayed causally after reset, every search frame through `t+1` is prediction-centred, and no prefix/history label is read.
- [ ] The fixed attempted-clip manifest is independent of activity, confidence, labels, loss, and coefficient values; outer-label validity is inspected only after the outer crop/forward, and failed attempts are recorded without replacement or reweighting.
- [ ] Pre-fit and selected-checkpoint feasibility audits each cover the exact same 100 manifested attempts in every registered base/modality stratum, pass every adaptive key/family activity and unit-route threshold, and pass both registered singular-value floors for a full-rank signed gradient design on the three-dimensional fixed-budget alpha-simplex tangent; leaf-alpha absolute sensitivity is diagnostic only.
- [ ] Every attempted superstep validates its exact manifest slice before the first forward and processes/records all twelve clips even if an early or middle stratum has zero valid outer labels; calibration reaches exactly 1,000 successful supersteps within at most 2,000 attempted supersteps, with checkpoints and selection indexed by successful step while retaining attempted-step provenance.
- [ ] Only one four-scalar external coefficient vector is fitted; every model parameter hash stays fixed.
- [ ] S0 schedules originate from routing-disabled frozen legacy output and match across rows.
- [ ] Single-modality unreliability makes all four paired families skip under the shared sealed `paired_valid` record.
- [ ] Pair shuffle, temporal shuffle, mask shuffle, branch-only, cumulative, raw leave-one-out, strength-matched leave-one-out, `random_orthogonal`, `pooled_same`, and `target_balanced_identity` controls are present.
- [ ] Matched rows report identical state/capacity/logit/schedule budgets; `*_strength_matched` is explicitly alpha-budget-matched only and reports unclipped/clipped realized route-norm distributions rather than claiming norm equality.
- [ ] The registry and benchmark-evaluator/checkpoint chain are frozen before any gate schedule is generated; the twelve-schedule manifest is committed before compared gate rows.
- [ ] The LasHeR author archive and explicit MATLAB executable pass isolated fixture validation before the first calibration `J_core`; missing external prerequisites stop execution.
- [ ] Clean and registered-corruption results live in disjoint method/condition directories with distinct schedule and metric hashes.
- [ ] The final gate consumes six frozen profiles/M1 tables plus a frozen-selected-rank DDP artifact, twelve hashed raw `frames.jsonl` sources, both raw signed-gradient feasibility reports, and the raw per-sequence pair-alignment artifact; full-stream actual-commit coverage reaches 0.20 overall/every base-benchmark stratum and each measured profile reaches 0.20 only after raw-source recomputation.
- [ ] Each profile runs one batch-one method in its own child process on the exact frozen profile CUDA fingerprint, reports raw 60-frame timing/commit vectors plus initialization and complete-episode latency/FPS, and binds both absolute allocated and absolute reserved CUDA peaks; no PID is reused across the four compared methods, and the DDP artifact names the exact two ordered frozen CUDA fingerprints.
- [ ] Nonlinear benchmark aggregates are recomputed inside sampled seed-slots before equal weighting; adapters remain raw `[0,1]` and only gate analysis converts raw contrasts to percentage points once.
- [ ] Recovery uses the fifth qualifying frame, treatment-independent risk set, fixed horizon, and right censoring.
- [ ] Rank energy uses exact cumulative total trace and a stored boundary eigenvalue, not the retained-spectrum denominator.
- [ ] Both clean benchmark noninferiority inequalities are strict.
- [ ] J-core and RMST each contribute all three geometry, four strength-matched family-attribution, and three shuffle contrasts to one 20-member simultaneous-LCB family under the same frozen crossed-bootstrap plan; the internally selected endpoint's ten strict bounds all pass.
- [ ] The final pass bit is freshly recomputed from hash-verified raw inputs; no serialized `pass` field is accepted as evidence.
- [ ] Claims are limited to frozen-parameter, sequence-local, strictly causal uncentered-moment routing; no no-loss guarantee, continual parameter learning, or cross-sequence retention claim is made.
- [ ] Workstream B/C code is absent.

## Plan Self-Review Commands

Run before handing this plan to an implementation worker:

~~~bash
bad='TO''DO''|''TB''D''|''FIX''ME''|''place''holder''|<''frozen''|<''commit''|<''path'
rg -n "$bad" \
  docs/superpowers/plans/2026-07-13-target-spectral-workstream-a.md
rg -n "test-time optimizer|EMA teacher|rollback mechanism|Stage [RE] implementation" \
  docs/superpowers/plans/2026-07-13-target-spectral-workstream-a.md
.venv/bin/python - <<'PY'
from pathlib import Path
p = Path("docs/superpowers/plans/2026-07-13-target-spectral-workstream-a.md")
text = p.read_text()
assert chr(92) + chr(96) not in text
assert text.count("~~~") % 2 == 0
print({"lines": len(text.splitlines()), "fences": text.count("~~~")})
PY
git add docs/superpowers/plans/2026-07-13-target-spectral-workstream-a.md
git diff --cached --check
git diff --cached --name-only
~~~

Expected:

- the unresolved-marker scan returns no matches;
- scope-exclusion terms appear only in constraints, rejection tests, and stop/go language;
- the inline-backtick and fence assertions pass;
- `git diff --cached --check` returns no output;
- the cached name list contains only this plan before its plan-only commit.

## Execution Handoff

After the user explicitly ratifies Decision 7, choose one execution mode:

1. **Subagent-Driven (recommended):** execute task-by-task in this session with a fresh implementation subagent and review checkpoint after every task.
2. **Inline Execution:** execute the same checkboxes sequentially in the primary session, pausing at each task-local verification/commit boundary.

Neither mode may begin Task 0 or edit implementation files before Decision 7 is ratified.
