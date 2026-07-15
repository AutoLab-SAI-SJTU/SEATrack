# LiftTrack Rank-Collapsed BiLift Design

**Status:** Approved for a falsifiable implementation and pilot experiment

**Research objective:** Replace SEATrack's sample-invariant AMG interpolation and dense HMoE with a simpler cross-modal operator that preserves information, matches the observed effective rank, and is no slower than reproduced SEATrack.

## Evidence Behind the Design

The design is based on the current code, trained checkpoints, and paired diagnostics rather than paper terminology.

1. AMG applies a two-by-two modality mixing matrix to attention logits:

   ```text
   S_rgb' = (1 - w_x) S_rgb + w_x S_x
   S_x'   = w_rgb S_rgb + (1 - w_rgb) S_x
   ```

   Therefore:

   ```text
   S_rgb' - S_x' = (1 - w_rgb - w_x) (S_rgb - S_x)
   ```

   In the matched RGB-T epoch-50 checkpoint, the absolute difference multiplier is 0.0050 at layer 1 and 0.0546 at layer 3. Attention agreement can therefore be produced by an almost singular operator that suppresses modality-specific information.

2. Forcing `w_rgb = w_x = 0.5` at all six AMG layers makes the two post-guidance attention maps identical by construction. On 640 fixed-seed validation samples this changed mean IoU by only -0.00040 relative to learned AMG, whereas disabling exchange changed it by -0.01566. Information exchange is useful; perfect attention equality is not the causal target.

3. HMoE executes every expert and uses dense dispatch and combine softmax operations. Six selected layers call HMoE four times each, for 24 calls per forward.

4. Every HMoE output passes through a shared rank-4 `linear2`. After centering over output tokens, its channel rank is at most four. A 32-sample empirical SVD confirmed numerical rank four for all 12 HMoE modules, with 99 percent of energy in two to four components.

5. Frozen-checkpoint interventions indicate that early HMoE is dispensable. Keeping only middle and late HMoE while disabling layers 1 and 3 changed mean IoU by +0.00385 on 640 samples. The strongest repeatable single-layer effects were around layers 5 and 9.

6. Dynamic action routing is not currently learnable from available features. A four-action late-layer oracle gained +0.01644 IoU, but an offline router trained on 1,200 high-dimensional target-conditioned samples achieved 26.75 percent action accuracy and -0.00045 IoU gain on 400 held-out samples. Router-based SCGE is therefore not the first implementation candidate.

## Proposed Method

The provisional paper method is **LiftTrack: Rank-Collapsed Information-Preserving Lifting for Efficient Multimodal Tracking**.

### Keep Mergeable Domain Adaptation

Keep standard LoRA on the K and V projections at layers `[1, 3, 5, 7, 9, 11]`. Disable only AMG cross-guidance. At evaluation, the existing `MergedLinear` folds LoRA into QKV, so this adaptation has no separate inference kernel.

### Replace AMG and HMoE with BiLift

Insert BiLift after the independent attention residuals and before candidate elimination/MLP at blocks 5 and 9. A BiLift unit uses two sequential additive couplings:

```text
rgb_1 = rgb + x_to_rgb(x)
x_1   = x   + rgb_to_x(rgb_1)
```

The second BiLift reverses the order:

```text
x_1   = x   + rgb_to_x(rgb)
rgb_1 = rgb + x_to_rgb(x_1)
```

Each cross update is a rank-8 adapter:

```text
update(source) = up(GELU(down(parameter_free_layer_norm(source))))
```

`down` uses Xavier initialization and `up` is zero-initialized. A new model is therefore exactly equivalent to the LoRA-only model before training.

For the first ordering, the exact inverse is:

```text
x   = x_1   - rgb_to_x(rgb_1)
rgb = rgb_1 - x_to_rgb(x)
```

The coupling Jacobian is block triangular with determinant one. This claim applies to each BiLift interaction, not to the complete Transformer.

### Final Readout

Keep the existing final sum and frozen Center Head for the first candidate. BiLift has already injected complementary information into both branches. A separate dynamic fusion head is excluded until a controlled ablation proves it is necessary.

## Configuration Contract

Existing YAML files retain current behavior by default:

```yaml
MODEL:
  AMG_ENABLED: true
  HMOE_ENABLED: true
  BILIFT:
    ENABLED: false
    LAYERS: [5, 9]
    RANK: 8
    DROPOUT: 0.0
    DIAGNOSTICS: false
```

LiftTrack configurations must use:

```yaml
MODEL:
  AMG_ENABLED: false
  HMOE_ENABLED: false
  BILIFT:
    ENABLED: true
    LAYERS: [5, 9]
    RANK: 8
    DROPOUT: 0.0
    DIAGNOSTICS: true
```

`GRA.ENABLED` and `GRA.DIAGNOSTICS` must both be false for LiftTrack. Invalid combinations fail during model construction.

## Static Efficiency Budget

For two streams with 320 tokens, dimension 768, and rank 8:

```text
legacy HMoE MACs       = 0.2259 G
two BiLift MACs        = 0.0157 G
estimated SEATrack     = 56.466 G
estimated LiftTrack    = 56.256 G
```

Expected trainable parameters:

```text
K/V LoRA               = 147,456
two rank-8 BiLift       = 49,152
total                   = 196,608 (plus no affine norm parameters)
SEATrack                = 636,324
```

These are analytical expectations, not accepted results. Model construction and paired CUDA measurements must verify them.

## Predeclared Gates

### Engineering gates

- All unit and integration tests pass.
- Zero-initialized LiftTrack is output-identical to LoRA-only in evaluation mode.
- A randomized BiLift round-trip reconstructs both streams within `1e-5` absolute tolerance.
- LiftTrack instantiates no `HMoE`, AMG scaling, GRA, or RGAE parameters.
- Trainable parameters are only LoRA and BiLift parameters.
- One-sample real-data forward/backward produces finite predictions, losses, and gradients.

### Efficiency gates against reproduced SEATrack

- Worst-path analytical MACs: at most `1.00x`.
- Mean model latency at batch 1: at most `1.00x`.
- P90 latency: at most `1.02x`.
- Peak allocated inference memory: at most `1.00x`.
- Training step time: at most `1.00x` after the same warmup and logging settings.

### Accuracy gates

- A matched five-epoch pilot uses the same seed, sample order, optimizer, data, and initialization for LoRA-only, SEATrack, and LiftTrack.
- LiftTrack mean validation IoU must not be more than 0.002 below matched SEATrack at any predeclared checkpoint.
- LiftTrack must exceed LoRA-only at the final pilot checkpoint.
- A long run is forbidden if the five-epoch pilot fails either condition.

## Primary Ablations

1. Frozen dual-stream OSTrack plus sum.
2. LoRA-only, without AMG or HMoE.
3. Full reproduced SEATrack.
4. Parameter-matched parallel bidirectional adapters.
5. BiLift at `[5, 9]`.
6. BiLift rank 4, 8, and 16.
7. BiLift at `[7, 11]` and `[5, 7, 9, 11]` only after the default pilot.
8. Forward-first and reverse-first coupling order.

## Non-Goals for the First Candidate

- No counterfactual router, Gumbel routing, Taylor target, or branch-specific prediction head.
- No OT, Sinkhorn, ProbAlign, token pruning, early exit, distillation, or temporal memory.
- No claim that attention equality is reliability.
- No 60-epoch run before all engineering, efficiency, and pilot gates pass.

