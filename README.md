# Healthcare AI GYM for Medical Agents

**A Gymnasium-compatible reinforcement learning environment for training medical AI agents through multi-turn clinical tool use.**

Healthcare AI GYM bridges the gap between static medical knowledge retrieval and agentic clinical execution. Built on the Gymnasium interface (`step(action)/render()`), it spans **10 clinical domains** with **3.6K+ tasks**, **135 domain-specific tools**, a **knowledge base of 828K medical passages**, and a **safety-aware 5D reward function** — enabling seamless integration with modern RL training pipelines.

We propose **Turn-Level Truncated On-Policy Distillation (TT-OPD)**, a self-distillation framework that stabilizes multi-turn agentic RL via a gradient-free EMA teacher, outcome-conditioned privileged hints, and cosine length-controlled reward shaping. TT-OPD achieves the best performance on **10 of 18 benchmarks** with an average **+3.9 pp improvement** over the non-RL baseline.

---

## Key Results

### Benchmark Performance (Table 1)

All models use **Qwen3.5-9B** backbone. **Base (text)** = log-probability evaluation without tools. **Base+AR** = multi-turn AgentRunner with 135 tools and 828K-passage KB but no RL training. **GRPO** and **TT-OPD** are RL-trained models evaluated via multi-turn AgentRunner.

| Category | Benchmark | Base (text) | Base+AR | GRPO | TT-OPD |
|----------|-----------|:-----------:|:-------:|:----:|:------:|
| **MC QA** | MedQA (USMLE) | 70.7 | 78.8 | 85.5 | **87.1** |
| | MMLU-Med. (6 sub.) | **83.8** | 60.6 | 60.1 | 65.5 |
| | MedMCQA | 63.8 | 55.8 | 58.0 | **66.2** |
| **Visual QA** | VQA-RAD | 52.5 | **63.2** | 60.7 | 63.1 |
| | PathVQA | 40.5 | 38.7 | 41.5 | **45.3** |
| | SLAKE | **79.0** | 30.6 | 29.5 | 32.1 |
| | PMC-VQA | **57.9** | 35.1 | 34.2 | 38.9 |
| | VQA-Med-2021 | 8.6 | 9.8 | 10.7 | **15.2** |
| | Quilt-VQA | 25.2 | 27.8 | 25.2 | **30.7** |
| **EHR** | MIMIC-III | 58.5 | 62.1 | 61.1 | **62.7** |
| | eICU | 53.2 | 55.9 | 55.5 | **57.1** |
| **LFQA** | LiveQA | 53.2 | 58.2 | 57.7 | **62.5** |
| | MedicationQA | 49.5 | 53.1 | 55.8 | **60.9** |
| | HealthSearchQA | 39.8 | 41.9 | 39.5 | **45.3** |
| | KQA-Golden | 55.7 | 62.1 | **65.3** | 64.1 |
| | KQA-Silver | 52.5 | 61.7 | **64.9** | 62.8 |

**Bold** = best per benchmark.

### Key Findings

1. **TT-OPD achieves best on 10/18 benchmarks** — broad competence across MC QA, Visual QA, EHR, and LFQA with +3.9 pp average improvement over Base+AR.
2. **Agentic evaluation overhead** — Multi-turn agentic evaluation introduces systematic overhead on knowledge-recall benchmarks (MMLU: 83.8% text → 60.6% Base+AR → 65.5% TT-OPD), confirming that agentic evaluation trades parametric precision for retrieval-augmented reasoning.
3. **GRPO excels on knowledge-intensive LFQA** — Vanilla GRPO leads on KQA-Golden (65.3%) and KQA-Silver (64.9%), suggesting higher peak training accuracy translates to better factual recall in open-ended settings.
4. **Agentic-textual transfer gap** — RL improves procedural competence but does not automatically transfer to text-based QA benchmarks due to format-reward dilution.

---

## Healthcare AI GYM: Environment Design

### 10 Clinical Domains

| Domain | Description |
|--------|-------------|
| Clinical Diagnosis | Differential diagnosis with lab/imaging ordering |
| Drug Interaction | Multi-drug safety checking and contraindication analysis |
| EHR Management | Structured EHR reasoning and clinical prediction tasks |
| Medical QA | Evidence-based medical question answering |
| Triage | Emergency severity assessment and resource allocation |
| Psychiatry | Mental health assessment with validated screening tools |
| Obstetrics/Gynecology | Prenatal care, labor management, and maternal health |
| Visual Diagnosis | Medical image interpretation (pathology, radiology, dermatology) |
| Radiology Report | Structured radiology report generation |
| Cross-Domain Pathways | Multi-specialty diagnostic pathways |

### 135 Domain-Specific Tools

Tools are consolidated into **25 user-facing categories** across 4 types:

| Type | Examples | Count |
|------|----------|:-----:|
| **Evidence Retrieval** | BM25-based KB querying (828K passages from PubMed, clinical guidelines) | — |
| **Clinical Assessment** | Validated scoring instruments (SOFA, NEWS, Bishop, PHQ-9, APGAR, etc.) | 22 |
| **Intervention Actions** | Lab ordering, imaging ordering, medication prescribing, triage assignment | — |
| **Reasoning Scaffolds** | Differential diagnosis generation, treatment planning, ICD-10 coding | — |

Tools use a decorator-based auto-generation pattern for OpenAI-compatible definitions, ensuring extensibility while maintaining ecological validity.

### 5D Reward Function

The reward function formalizes clinical priorities into a single optimization objective:

```
R_total = w_acc * R_acc + w_proc * R_proc + w_safe * R_safe + w_fmt * R_fmt + w_coh * R_coh
```

| Component | Weight | Description |
|-----------|:------:|-------------|
| Accuracy | 0.25 | Diagnostic precision and correctness |
| Process | 0.20 | Procedural quality (appropriate tool use, clinical workflow) |
| Safety | 0.20 | Safety-severity taxonomy with critical violation capping |
| Format | 0.10 | Structural correctness of tool calls and submissions |
| Coherence | 0.10 | Logical consistency across conversation turns |

An optional **assertion** dimension (w=0.15) is available when rubric annotations are provided. Critical safety violations cap the composite score at 0.1, addressing the "format reward dilution" problem where agents prioritize structural correctness over clinical utility.

### Knowledge Base

- **828K medical passages** indexed via BM25 search
- Sources: PubMed abstracts, clinical guidelines (AHA, ACOG, SSC), medical textbooks
- Integrated as tool calls within the agent's action space

---

## TT-OPD: Turn-Level Truncated On-Policy Distillation

### Motivation: Three Failure Modes in Multi-Turn Agentic RL

Training agents through multi-turn RL reveals three compounding pathologies absent in single-turn settings:

1. **Response Explosion** — Outputs grow monotonically to the context limit. Without intermediate feedback, the model adopts token-level coverage as a proxy for task completion.
2. **Multi-Turn Collapse** — The agentic structure degrades from coordinated tool-use dialogues into verbose single-turn monologues. These two pathologies are causally linked: single-turn verbosity and length explosion create a self-reinforcing collapse loop.
3. **Distillation Instability** — On-policy distillation (OPD), while effective for single-turn reasoning, fails in agentic settings because the combinatorial complexity of trajectory space causes teacher policies to become stale far more rapidly.

### Method

TT-OPD addresses these failures through three components:

**1. Gradient-Free EMA Teacher**
- Teacher tracks student via Exponential Moving Average: `theta_T <- alpha * theta_T + (1 - alpha) * theta_S`
- `alpha = 0.995`, updated every 5 steps
- Periodic hard-copy fallback (`theta_T <- theta_S` every 30 steps) prevents excessive teacher-student divergence
- No gradient updates for the teacher — ensures training stability

**2. Outcome-Conditioned Privileged Hints**
- Teacher receives correctness-dependent signals `h(tau)` for every trajectory
- **Reinforcing hints** (e.g., *"Reasoning appears sound"*) for correct trajectories — increase teacher confidence on successful paths
- **Corrective hints** (e.g., *"Revisit the differential diagnosis"*) for incorrect trajectories — shift teacher distribution away from error patterns
- Hints are inserted at the prompt-response boundary but **removed from the teacher's output logprobs** — the student never explicitly observes the hints
- Provides dense, outcome-aware KL regularization at every conversation turn

**3. Cosine Length-Controlled Reward**
- Based on [Yeo et al. (2025)](https://arxiv.org/abs/2502.03373)
- Correct answers: higher reward when concise, decaying as length grows
- Incorrect answers: increasing penalty with length
- Truncated responses: fixed penalty
- Prevents monotonic length growth as responses approach L_max

**Combined Loss:**
```
L_total = L_GRPO(theta_S; R_cos) + lambda_distill * D_KL(pi_theta_S || pi_theta_T)
```
where `lambda_distill = 4.0` provides strong regularization against agentic collapse.

### Training Dynamics

| Metric | TT-OPD | GRPO |
|--------|:------:|:----:|
| Peak validation accuracy | 61.1% (step 60) | 62.0% (step 55) |
| Mean accuracy (steps 40-60) | 59.5% (±1.4 pp) | — |
| Response length | 5.7–9.3K tokens (controlled) | 7.7–10.8K tokens (oscillating) |
| Avg turns per episode | 7.0–7.4 (stable) | Declining |
| Training stability | Non-monotonic convergence, stable | Significant oscillations |

**Key advantage of TT-OPD is not raw accuracy but training stability**: controlled response length and sustained multi-turn tool use throughout training.

### OPD Failure Progression (Ablation)

| Variant | Accuracy | Turns | Failure Mode |
|---------|:--------:|:-----:|-------------|
| (1) Periodic teacher reset | 56.9% → 49.3% | 7.65 → 5.52 | KL collapse at each copy event |
| (2) EMA teacher (no conditioning) | 53.8% (step 40) | 7.82 → 6.23 | Generic regularization, turns erode |
| (3) EMA + outcome hints (no length ctrl) | 54.5% plateau → 49.0% | — | Response explosion to L_max |
| (4) **Full TT-OPD** | **61.1%** (step 60) | **7.0–7.4** (stable) | Sustained convergence |

Each component addresses a distinct failure mode: EMA prevents KL collapse, outcome hints provide outcome-aware signal, and cosine reward prevents response explosion. Multi-turn collapse is identified as an **agentic-specific** failure mode absent from single-turn OPD.

---

## Architecture

```
                    +------------------------------+
                    |     Healthcare AI GYM         |
                    |   (Gymnasium BioAgent-v0)     |
                    +--------------+---------------+
                                   |
              +--------------------+--------------------+
              |                    |                     |
     +--------v--------+  +-------v-------+   +--------v--------+
     |   Student Model  |  |  EMA Teacher  |   |  Reward System  |
     |   (Qwen3.5-9B)  |  | (decay=0.995) |   |  (5D + Cosine)  |
     +--------+--------+  +-------+-------+   +--------+--------+
              |                    |                     |
              v                    v                     v
     +----------------------------------------------------------+
     |              TT-OPD Training Loop                         |
     |                                                           |
     |  1. Student rollout (multi-turn tool-use, SGLang)         |
     |  2. Outcome evaluation (5D reward)                        |
     |  3. Teacher re-scores with outcome-privileged hints       |
     |  4. Turn-level KL distillation (all conversation turns)   |
     |  5. GRPO policy gradient + cosine length reward           |
     |  6. EMA update: teacher <- 0.995*teacher + 0.005*student  |
     +----------------------------------------------------------+
              |
              v
     +----------------------------------------------------------+
     |                 10 Clinical Domains                       |
     |  Diagnosis | Drug Interaction | EHR | Medical QA         |
     |  Triage | Psychiatry | OB/GYN | Visual Dx               |
     |  Radiology Report | Cross-Domain Pathways                |
     |                                                           |
     |  135 tools | 3.6K+ tasks | 828K knowledge passages       |
     +----------------------------------------------------------+
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

## Training

### TT-OPD (Recommended)

```bash
source .venv/bin/activate
bash scripts/verl/run_qwen35_9b_self_distill_v16_ema_cosine.sh
```

### GRPO Baseline

```bash
source .venv/bin/activate
bash scripts/verl/run_qwen35_9b_grpo_baseline.sh
```

### Key Hyperparameters

| Parameter | TT-OPD | GRPO |
|-----------|:------:|:----:|
| Learning rate | 5e-7 | 5e-7 |
| Batch size | 8 | 8 |
| Rollouts per prompt | 3 | 3 |
| EMA decay | 0.995 | — |
| EMA update interval | 5 steps | — |
| Distillation coef (lambda) | 4.0 | — |
| Hard-copy fallback | Every 30 steps | — |
| Cosine length reward | Yes | No |
| Outcome-conditioned hints | Yes | — |

### Evaluation

Evaluation must use multi-turn AgentRunner (single-turn produces 0% because TT-OPD-trained models reason through tool calls):

```bash
# Multi-turn evaluation (required)
python scripts/eval_benchmark_multiturn.py \
    --model-path <checkpoint> \
    --benchmark medqa \
    --max-turns 10

# EHR action-based evaluation
python scripts/eval_benchmark_multiturn.py \
    --model-path <checkpoint> \
    --benchmark mimic_iii \
    --max-turns 10
```

### Experimental Setup

- **Base model**: Qwen3.5-9B, trained from scratch without SFT warmup
- **Hardware**: 8x A100 80GB
- **Data contamination**: Zero, verified via test-set fingerprinting
- **Validation set**: 307 held-out tasks (149 Medical QA, 37 Visual Diagnosis, 25 Clinical Diagnosis, 25 Drug Interaction, 25 EHR, 20 Triage, 20 Psychiatry, 6 Obstetrics)
- **Training pipeline**: Built on veRL (FSDP-based multi-turn GRPO with hybrid engine support)

---

## Project Structure

```
Healthcare_GYM/
├── bioagents/
│   ├── gym/                        # Gymnasium environment
│   │   ├── agent_env.py            # BioAgent-v0 environment
│   │   ├── autonomous_gym.py       # Multi-agent GYM orchestrator
│   │   └── autonomous_agent.py     # Self-directing agent
│   ├── domains/                    # 10 clinical domains (135 tools)
│   ├── evaluation/
│   │   ├── grpo_rewards.py         # 5D reward system
│   │   └── benchmark_eval.py       # 18-benchmark evaluation suite
│   └── tools/
│       └── knowledge_tools.py      # Unified search (828K passages)
├── scripts/
│   ├── verl/
│   │   ├── run_qwen35_9b_self_distill_v16_ema_cosine.sh  # TT-OPD training
│   │   ├── run_qwen35_9b_grpo_baseline.sh                # GRPO baseline
│   │   └── reward_fn.py                                   # Custom reward function
│   ├── eval_benchmark_multiturn.py  # Multi-turn evaluation
│   └── eval_benchmark.py           # Single-turn evaluation
├── configs/                        # Qwen3.5-9B YAML configurations
├── data/
│   └── domains/                    # Per-domain task data
├── tests/                          # Unit tests
├── LICENSE
├── pyproject.toml
└── requirements.lock
```

---

## Citation

```bibtex
@software{healthcare_ai_gym_2026,
  title={Healthcare AI GYM for Medical Agents},
  author={Jeong, Minbyul},
  year={2026},
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

- All patient data is **synthetic**. No real patient information is used.
- **Research tool only** — NOT for clinical use or medical decision-making.
