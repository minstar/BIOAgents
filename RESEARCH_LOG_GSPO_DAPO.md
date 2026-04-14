# GYM Architecture Research — GSPO/DAPO Migration Plan

## B041 Root Cause (Confirmed by GSPO Paper, Qwen Team July 2025)

Standard GRPO uses token-level importance sampling weights based on a single sample from each
next-token distribution. This introduces high-variance training noise that progressively accumulates
with increased response length and is amplified by the clipping mechanism, ultimately causing
irreversible model collapse.

## Key References

| Paper | Method | Key Result | Relevance |
|-------|--------|------------|-----------|
| GSPO (Qwen3) | Sequence-level importance ratios + clipping | Continuous improvement, no collapse | Direct B041 fix |
| DAPO (ByteDance) | Clip-Higher + Dynamic Sampling + Token-Level PG | AIME 50pts w/ Qwen2.5-32B | Entropy collapse fix |
| Dr. GRPO (COLM 2025) | Remove length/std normalization | 43.3% AIME w/ 7B | Bias removal |
| MediX-R1 | GSPO/DAPO for medical VLM | MedQA 79.6% w/ 8B | Medical benchmark SOTA |
| Med-R1 | Rule-based reward, KL reg | 69.9% overall w/ 3B | Simple effective approach |
| RFT Forgetting Study | RL vs SFT forgetting analysis | RL: -2.3% forgetting vs SFT: -10.4% | RL is better for retention |
| WebRL (ICLR 2025) | Self-evolving curriculum + ORM | 4.8%→42.4% on WebArena | Curriculum design |
| MedAgentGym (ICLR 2026) | SFT→DPO two-stage | 16.9%→59.9% w/ 7B | Two-stage approach |

## DAPO 4 Key Techniques

1. **Clip-Higher** (asymmetric clipping): ε_low=0.2, ε_high=0.28
   - Prevents entropy collapse by allowing more exploration upside
2. **Dynamic Sampling**: Filter batches to only include prompts with mixed correct/incorrect
   - Skip uninformative prompts (all-correct or all-wrong groups)
3. **Token-Level Policy Gradient Loss**: Per-token gradients, not per-sequence
   - Reduces bias from variable response lengths
4. **Overlong Reward Shaping**: Linear penalty for responses > safe_length
   - Prevents verbosity gaming

## Implementation Plan

### Phase 1: GSPO/DAPO Trainer (This Week)
- Replace `grpo_trainer.py` core with GSPO sequence-level importance ratios
- Add DAPO's Clip-Higher and Dynamic Sampling
- Add Dr. GRPO's length/std normalization removal
- Set LR=1e-6, beta(KL)=0.01, LoRA r=64

### Phase 2: Training Pipeline (Next Week)
- Implement RIF-RFT (filter incompetent samples)
- Add SFT warmup stage before RL
- Add benchmark-guided early stopping (eval MedQA every N steps)
- Add entropy monitoring + adaptive entropy bonus

### Phase 3: Full Experiment (2 Weeks)
- Run Lingshu-7B with GSPO+DAPO on all 10 domains
- Run Qwen2.5-VL-7B with same setup
- Ablation: GSPO vs DAPO vs standard GRPO (fixed)
- Target: MedQA 70%+, MMLU 80%+
