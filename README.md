# Healthcare AI GYM

**A Gymnasium-compatible environment for training medical AI agents via Bidirectional Truncated On-Policy Distillation with Hint Injection (BT-OPD+Hint).**

Healthcare AI GYM provides a comprehensive training infrastructure spanning 10 clinical domains, 195K+ tasks, and 187+ domain-specific tools. Agents interact with realistic medical scenarios through multi-turn tool-use conversations -- gathering patient information, ordering tests, consulting guidelines, and submitting structured clinical assessments. The core training method, BT-OPD+Hint, uses teacher-student co-evolution with EMA updates, bidirectional KL across all conversation turns, correctness-conditioned hint injection, and cosine length-controlled rewards to achieve 61.1% validation accuracy (+8.5pp over baseline).

---

## Key Features

- **Healthcare AI GYM Environment** -- Gymnasium-compatible (`BioAgent-v0`) with 10 clinical specialties, 187+ tools (labs, vitals, imaging, prescribing, scoring, clinical guidelines), and 195K+ training tasks spanning diagnosis, triage, drug interaction, EHR management, psychiatry, obstetrics, radiology, and cross-domain pathways.

- **BT-OPD+Hint Training Method** -- Bidirectional Truncated On-Policy Distillation with correctness-conditioned hints. Teacher and student co-evolve via EMA (decay=0.995), with bidirectional KL applied across all conversation turns and dynamic hints injected into teacher context based on student correctness.

- **Multi-Domain Clinical Tools** -- 187+ tools organized by specialty. Agents make structured tool calls (JSON) to gather information, perform calculations (SOFA, NEWS, Bishop score), search clinical guidelines (AHA, ACOG, SSC), and submit diagnoses with ICD-10 codes.

- **Safety-Aware Rewards** -- 5-dimensional reward system (Accuracy 30%, Process 25%, Safety 20%, Format 15%, Coherence 10%) with cosine length-controlled reward to prevent response explosion (arXiv:2502.03373). Critical safety violations cap the composite score at 0.1.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │     Healthcare AI GYM         │
                    │   (Gymnasium BioAgent-v0)     │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼────────┐  ┌───────▼───────┐   ┌────────▼────────┐
     │   Student Model  │  │  EMA Teacher  │   │  Reward System  │
     │   (Qwen3.5-9B)  │  │ (decay=0.995) │   │  (5D + Cosine)  │
     └────────┬────────┘  └───────┬───────┘   └────────┬────────┘
              │                    │                     │
              ▼                    ▼                     ▼
     ┌──────────────────────────────────────────────────────────┐
     │              BT-OPD+Hint Training Loop                   │
     │                                                          │
     │  1. Student rollout (multi-turn tool-use, SGLang)        │
     │  2. Teacher generates logprobs + hint-augmented context  │
     │  3. Bidirectional KL loss across ALL turns               │
     │  4. Cosine length reward + 5D composite reward           │
     │  5. GRPO policy gradient update                          │
     │  6. EMA update: teacher <- 0.995*teacher + 0.005*student │
     └──────────────────────────────────────────────────────────┘
              │
              ▼
     ┌──────────────────────────────────────────────────────────┐
     │                 10 Clinical Domains                       │
     │  Diagnosis | Drug Interaction | EHR | Medical QA         │
     │  Triage | Psychiatry | OB/GYN | Visual Dx               │
     │  Radiology Report | Cross-Domain Pathways                │
     │                                                          │
     │  187+ tools | 195K+ tasks | 828K knowledge passages      │
     └──────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Create and activate virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,flash]"
```

### Requirements

| Requirement | Details |
|---|---|
| Python | 3.12+ |
| GPU | 8x A100-80GB (recommended) |
| CUDA | 12.8+ |
| Framework | veRL v0.8.0+ with SGLang backend |

### Run a Single Task

```python
import gymnasium as gym
from bioagents.gym.agent_env import register_bioagent_gym

register_bioagent_gym()
env = gym.make("BioAgent-v0", domain="clinical_diagnosis", task_id="dx_pneumonia_001")
obs, info = env.reset()
print(obs)  # Patient scenario + available tools
```

---

## Training with BT-OPD+Hint

The primary training method uses veRL multi-turn GRPO with self-distillation:

```bash
source .venv/bin/activate

# BT-OPD+Hint (v16): EMA self-distillation + hints + cosine length reward
bash scripts/verl/run_qwen35_9b_self_distill_v16_ema_cosine.sh
```

### Key Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `distillation.use_ema` | `True` | EMA teacher updates |
| `distillation.ema_decay` | `0.995` | Teacher momentum |
| `distillation.ema_update_interval` | `5` | Steps between EMA updates |
| `distillation.distillation_loss.loss_mode` | `bt_opd_kl` | Bidirectional truncated OPD |
| `distillation.distillation_loss.distillation_loss_coef` | `4.0` | Distillation loss weight |
| `HINT_OPD_ENABLED` | `1` | Correctness-conditioned hints |
| `COSINE_REWARD` | `1` | Cosine length penalty |
| `BT_OPD_BIDIRECTIONAL` | `1` | Bidirectional KL (positive + negative trajs) |
| `actor_rollout_ref.rollout.n` | `3` | Rollouts per prompt |
| `data.train_batch_size` | `8` | Training batch size |
| `actor_rollout_ref.actor.optim.lr` | `5e-7` | Learning rate |

### BT-OPD+Hint Method Details

1. **EMA Teacher Co-Evolution**: The teacher is an exponential moving average of the student (decay=0.995, updated every 5 steps), allowing the teacher to gradually improve alongside the student without catastrophic forgetting.

2. **Bidirectional KL**: KL divergence is computed in both directions across all conversation turns -- forward KL on correct trajectories (student learns from teacher) and reverse KL on incorrect trajectories (corrective gradient).

3. **Correctness-Conditioned Hints**: When the student answers incorrectly, a dynamic hint is injected into the teacher's context before generating teacher logprobs. This guides the teacher to produce a more informative distribution for the student to learn from.

4. **Cosine Length Reward**: Based on Yeo et al. (arXiv:2502.03373), a cosine function scales rewards based on response length to prevent response explosion. Correct answers get higher reward when concise (1.1 -> 0.7 as length grows), wrong answers get increasing penalty (-0.0 -> -0.3), and exceeding max length yields -0.5.

---

## Results

### v16 BT-OPD+Hint Training Trajectory (Qwen3.5-9B)

| Step | Val Accuracy | KL (student-teacher) | Avg Response Length | Notes |
|-----:|:------------:|:--------------------:|:-------------------:|-------|
| 0 | 52.6% | 0.00 | ~4,200 tokens | Baseline (SFT checkpoint) |
| 20 | 56.8% | 0.12 | ~5,100 tokens | Hint-driven plateau begins |
| 40 | 58.3% | 0.18 | ~5,400 tokens | Cosine reward stabilizes length |
| 60 | 61.1% | 0.21 | ~5,500 tokens | Peak accuracy (+8.5pp over baseline) |

**Key findings vs. prior versions:**
- v14 (EMA only): 53.8% at step 40, oscillating recovery but no sustained improvement
- v15 (EMA + Hints, no length penalty): 54.5% plateau (steps 10-20), collapsed to 49.0% by step 40 due to response explosion
- v16 (EMA + Hints + Cosine): Sustained improvement to 61.1%, response length stabilized by cosine reward

---

## Project Structure

```
BIOAgents/
├── bioagents/
│   ├── gym/                        # Gymnasium environment + autonomous training
│   │   ├── agent_env.py            # BioAgent-v0 environment
│   │   ├── autonomous_gym.py       # Multi-agent GYM orchestrator
│   │   └── autonomous_agent.py     # Self-directing agent
│   ├── domains/                    # 10 clinical domains (187+ tools)
│   ├── evaluation/
│   │   ├── grpo_rewards.py         # 5D reward system
│   │   ├── benchmark_eval.py       # 21 benchmark evaluation
│   │   └── safety_eval.py          # Adversarial + bias testing
│   ├── training/
│   │   └── grpo_trainer.py         # GRPO/MRPO/SARL trainer
│   └── tools/
│       └── knowledge_tools.py      # Unified search (828K passages)
├── scripts/verl/
│   ├── run_qwen35_9b_self_distill_v16_ema_cosine.sh  # BT-OPD+Hint training
│   ├── reward_fn.py                # Custom reward function
│   └── tool_config_full.yaml       # Tool configuration
├── data/
│   ├── domains/                    # Per-domain tasks and data
│   └── verl_parquet/               # Training data in parquet format
├── configs/                        # YAML configurations
└── checkpoints/                    # Model checkpoints
```

---

## Citation

```bibtex
@software{healthcare_ai_gym_2026,
  title={Healthcare AI GYM: Training Medical AI Agents via Bidirectional Truncated On-Policy Distillation with Hint Injection},
  year={2026},
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

- All patient data is **synthetic**. No real patient information.
- **Research tool only** -- NOT for clinical use.
