# Experiment Plan for NeurIPS 2026 Camera-Ready

## Timeline and GPU Scheduling

| Date | GPUs | Experiment | Duration | Priority |
|------|------|------------|----------|----------|
| Apr 16-17 | 8 | v16 21-benchmark eval (running) | ~24h | P0 |
| Apr 17-18 | 8 | **Exp 1**: Base model eval (21 benchmarks) | ~24h | P0 |
| Apr 19 | 8 | **Exp 2a**: Multi-seed v16, seed 42 | ~10h | P0 |
| Apr 19-20 | 8 | **Exp 2b**: Multi-seed v16, seed 123 | ~10h | P0 |
| Apr 20-21 | 8 | **Exp 3**: GRPO baseline (Qwen3.5-9B multi-turn) | ~10h | P0 |
| Apr 21-22 | 8 | **Exp 4**: OPSD frozen-teacher baseline | ~10h | P1 |
| Apr 22-23 | 4 | **Exp 5**: v12-v15 MedQA only (4 parallel) | ~6h | P1 |
| Apr 22-23 | 2 | **Exp 6**: Tool ablation (think-only + no-think) | ~12h | P1 |
| Apr 22 | 1 | **Exp 7**: Obstetrics debug | ~1h | P2 |
| Apr 24+ | 8 | **Exp 8**: Dr.GRPO on Qwen3.5-9B (if time) | ~10h | P2 |

## Experiment Details

### Exp 1: Base Model Evaluation (P0)
- **Purpose**: Fill "Base Model" column in Table 4
- **Model**: `checkpoints/models/Qwen3.5-9B`
- **Script**: `scripts/eval_base_model_queue.sh` (already created)
- **Output**: `results/benchmarks_multiturn/base_qwen35_9b/`

### Exp 2: Multi-Seed TT-OPD (P0)
- **Purpose**: 3-seed confidence intervals for 61.1% headline
- **Script**: Copy `run_qwen35_9b_self_distill_v16_ema_cosine.sh`, add `algorithm.seed=42/123`
- **Change**: `trainer.experiment_name`, `trainer.rollout_data_dir`
- **Output**: Mean +/- std for Table 2

### Exp 3: GRPO Baseline on Qwen3.5-9B (P0)
- **Purpose**: Same-model GRPO comparison (currently uses Lingshu-7B)
- **Script**: Adapt existing GRPO script with multi-turn flags from v16
- **Key**: Remove distillation, keep same data/batch/LR
- **Output**: GRPO baseline row in Table 2, flat-line in Figure 3

### Exp 4: OPSD Frozen-Teacher Baseline (P1)
- **Purpose**: Validate claim that standard OPSD fails in multi-turn
- **Script**: Copy v14 script, disable EMA (`use_ema=False`, `teacher_update_interval=999`)
- **Output**: New row in ablation showing monotonic decline

### Exp 5: Component Ablation via Checkpoint Eval (P1)
- **Purpose**: Ablation showing each component's contribution on MedQA
- **Checkpoints**: v12/step10, v13/step10, v14/step40, v15/step15
- **Benchmark**: MedQA only (1273 samples, ~6h per checkpoint)
- **Note**: Need to merge veRL FSDP checkpoints to HF format first
- **Output**: Ablation table: v12-v16 x MedQA accuracy
- **GPU**: 4 GPUs parallel = ~6 hours total

### Exp 6: Tool Ablation (P1)
- **Purpose**: Disentangle retrieval contribution to 86.3% MedQA
- **Variants**: (a) think+submit only, (b) full tools, (c) no-think
- **Need**: Add `--tool-config` arg to eval script, create tool config variants
- **Output**: Tool ablation table

### Exp 7: Obstetrics Debug (P2)
- **Purpose**: Explain 0.000 action score
- **Approach**: Debug tool matching, run manual tasks

### Exp 8: Dr.GRPO on Qwen3.5-9B (P2)
- **Purpose**: Cross-model validation of Dr.GRPO dynamics claim

### Exp 9: 5D Reward Weight Ablation (P1)
- **Purpose**: Justify the specific reward coefficients (0.30/0.25/0.20/0.15/0.10)
- **Variants**:
  - (a) Accuracy-only: $w_\text{acc}=1.0$, all others 0
  - (b) Equal weight: $w_i=0.2$ for all 5 dimensions
  - (c) Safety-heavy: $w_\text{safe}=0.4$, redistribute others
  - (d) No-format: $w_\text{fmt}=0$, redistribute to accuracy
- **Training**: Short runs (~15 steps each) on MedQA validation
- **Output**: Ablation table showing how reward composition affects accuracy, safety, and turn structure
- **Motivation**: Reviewer will ask why these specific coefficients; shows format dilution (24:1 ratio from Proposition 2) and that accuracy-only is unstable

## Pre-flight Blockers
1. veRL checkpoint merging (FSDP → HF) for Exp 5
2. `--tool-config` flag in eval script for Exp 6
3. Verify `algorithm.seed` parameter name in veRL for Exp 2
4. Test GRPO multi-turn config with 1-2 steps before full run (Exp 3)

## What Each Experiment Fixes

| Paper Weakness | Experiment |
|---------------|------------|
| Empty base model column in Table 4 | Exp 1 |
| Single seed, no CI | Exp 2 |
| Cross-model comparison | Exp 3 |
| OPSD claim without direct evidence | Exp 4 |
| No benchmark ablation for v12-v15 | Exp 5 |
| Unknown retrieval contribution | Exp 6 |
| Obstetrics 0.000 unexplained | Exp 7 |
| Dr.GRPO only on Lingshu-7B | Exp 8 |
| 5D reward weights unjustified | Exp 9 |
