# BIOAgents - Detailed Progress Check

> `summary_and_planning.md`와 연동. 수치 · 로그 · 실행 상태 · 실패 이력을 추적.
> Claude가 context 없이도 즉시 현황을 파악하기 위한 reference.

---

## Last Updated: 2026-03-20 15:30 KST

---

## 환경 정보 (변경 시 반드시 업데이트)

| 항목 | 값 |
|------|-----|
| 프로젝트 경로 | `/data/project/private/minstar/workspace/BIOAgents` |
| Python 환경 | `.venv/bin/python` (uv 관리, Python 3.12) — **conda base 절대 금지** |
| W&B 프로젝트 | `pt2-minstar-gym-rl` |
| GPU 배정 (baseline 기간) | GPU 0,1: Lingshu / GPU 2,3: Qwen2VL / GPU 4+: Step3 / GPU 6: dryrun |
| 모델 경로 | `checkpoints/models/Lingshu-7B` |
| | `checkpoints/models/Qwen2.5-VL-7B-Instruct` |
| | `checkpoints/models/Step3-VL-10B` |
| Baseline 로그 | `logs/full_baseline_20260309/` |
| Dryrun 로그 | `logs/autonomous_gym_dryrun/launcher.log` |
| Autonomous GYM 로그 | `logs/autonomous/` (에이전트별 서브디렉토리) |

---

## [Phase 2-B2] Baseline Evaluation — 21 benchmarks × 3 models

### 실행 방법
```bash
cd /data/project/private/minstar/workspace/BIOAgents
.venv/bin/python scripts/run_all_benchmarks_parallel.py
# 마스터 로그: logs/full_baseline_20260309_master.log
# 모델별 로그: logs/full_baseline_20260309/{lingshu,qwen2vl,step3vl}_eval.log
```

### 카테고리별 평가 순서 (코드 내 순서 고정, run_all_benchmarks_parallel.py:main())
1. TEXT MC QA (8종): MedQA → MedMCQA → MMLU×6
2. MedLFQA (5종): KQA Golden → LiveQA → MedicationQA → HealthSearchQA → KQA Silver
3. Vision QA (5종): VQA-RAD → SLAKE → PathVQA → VQA-Med-2021 → Quilt-VQA
4. EHR (2종): MIMIC-III → eICU

---

### 🟢 Lingshu-7B (GPU 0,1) — 가장 진행 많이 됨

| 카테고리 | 벤치마크 | Score | N | 상태 |
|---------|---------|-------|---|------|
| Text MC QA | MedQA | **61.7%** acc | 1,273 | ✅ 완료 |
| | MedMCQA | **53.2%** acc | 4,183 | ✅ 완료 |
| | MMLU Clinical | **77.0%** acc | 265 | ✅ 완료 |
| | MMLU Professional | **79.8%** acc | 272 | ✅ 완료 |
| | MMLU Anatomy | **65.9%** acc | 135 | ✅ 완료 |
| | MMLU Genetics | **78.0%** acc | 100 | ✅ 완료 |
| | MMLU Biology | **87.5%** acc | 144 | ✅ 완료 |
| | MMLU College Med | **71.1%** acc | 173 | ✅ 완료 |
| | **Text QA Overall** | **58.8%** acc | 6,545 | ✅ 완료 |
| MedLFQA | KQA Golden | Token-F1=**27.0%** | 201 | ✅ 완료 |
| | LiveQA | Token-F1=**19.2%** | 100 | ✅ 완료 |
| | MedicationQA | Token-F1=**15.6%** | 666 | ✅ 완료 |
| | HealthSearchQA | Token-F1=**33.3%** | 3,077 | ✅ 완료 |
| | KQA Silver | Token-F1=**32.8%** | 904 | ✅ 완료 |
| Vision QA | VQA-RAD | EM=**52.9%**, F1=58.7% | 451 | ✅ 완료 |
| | SLAKE | EM=**24.0%**, F1=28.2% | 1,061 | ✅ 완료 |
| | PathVQA | EM=**54.7%**, F1=57.1% | 6,719 | ✅ 완료 |
| | VQA-Med-2021 | EM=**0.0%**, F1=4.7% | 500 | ✅ ⚠️ 이슈 |
| | Quilt-VQA | EM=**0.0%**, F1=27.0% | 326 | ✅ ⚠️ 이슈 |
| EHR | MIMIC-III | action_score=**36.9%** | **3,100**/5,000 | 🔄 62% 진행 |
| | eICU | — | — | ⏳ MIMIC 완료 후 |

---

### 🟡 Qwen2.5-VL-7B-Instruct (GPU 2,3)

| 카테고리 | 벤치마크 | Score | N | 상태 |
|---------|---------|-------|---|------|
| Text MC QA | MedQA | **54.0%** acc | 1,273 | ✅ 완료 |
| | MedMCQA | **51.0%** acc | 4,183 | ✅ 완료 |
| | MMLU Clinical | **78.1%** acc | 265 | ✅ 완료 |
| | MMLU Professional | **72.4%** acc | 272 | ✅ 완료 |
| | MMLU Anatomy | **68.2%** acc | 135 | ✅ 완료 |
| | MMLU Genetics | **79.0%** acc | 100 | ✅ 완료 |
| | MMLU Biology | **83.3%** acc | 144 | ✅ 완료 |
| | MMLU College Med | **66.5%** acc | 173 | ✅ 완료 |
| | **Text QA Overall** | **55.5%** acc | 6,545 | ✅ 완료 |
| MedLFQA | KQA Golden | Token-F1=**23.4%** | 201 | ✅ 완료 |
| | LiveQA | Token-F1=**16.5%** | 100 | ✅ 완료 |
| | MedicationQA | Token-F1=**12.8%** | 666 | ✅ 완료 |
| | HealthSearchQA | Token-F1=**28.8%** | 3,077 | ✅ 완료 |
| | KQA Silver | Token-F1=**27.4%** | 904 | ✅ 완료 |
| Vision QA | VQA-RAD | EM=**39.5%**, F1=43.9% | 451 | ✅ 완료 |
| | SLAKE | EM=**20.6%**, F1=23.7% | 1,061 | ✅ 완료 |
| | PathVQA | EM=**30.7%**, F1=31.7% | 6,719 | ✅ 완료 |
| | VQA-Med-2021 | EM=**0.0%**, F1=5.4% | 500 | ✅ ⚠️ 이슈 |
| | Quilt-VQA | EM=**0.0%**, F1=25.6% | 326 | ✅ 완료 |
| EHR | MIMIC-III | action_score=**58.3%** | **2,300**/5,000 | 🔄 46% 진행 |
| | eICU | — | — | ⏳ |

---

### 🔴 Step3-VL-10B (GPU 4+) — B039 수정 후 TextQA 재실행 중

| 카테고리 | 벤치마크 | Score | N | 상태 |
|---------|---------|-------|---|------|
| Vision QA | VQA-RAD | EM=**15.3%**, contains=43.5% | 451 | ✅ 완료 (v2) |
| | SLAKE | EM=**0.0%**, contains=52.9% | 1,061 | ✅ 완료 (v2) |
| | PathVQA | EM=**7.4%**, F1=9.5% | 6,719 | ✅ 완료 (v2) |
| | VQA-Med-2021 | EM=**0.0%**, F1=2.1% | 500 | ✅ 완료 (v2) |
| | Quilt-VQA | EM=**0.0%**, F1=21.2% | 326 | ✅ 완료 (v2) |
| | PMC-VQA | — | — | ⚠️ B040 fix 적용 (로딩 에러 수정) |
| Text MC QA | MedQA | 10.4% (B039 전: 51.5%) | 1,273 | 🔄 B039 fix → 재실행 중 |
| | MedMCQA | 10.7% (B039 전: 21.0%) | 4,183 | 🔄 재실행 중 |
| | MMLU Clinical | 10.6% (B039 전: 31.5%) | 1,089 | 🔄 재실행 중 |
| MedLFQA | KQA Golden | ROUGE-L=**9.5%**, F1=19.3% | 201 | ✅ 완료 |
| | LiveQA | ROUGE-L=**7.2%**, F1=14.3% | 100 | ✅ 완료 |
| | MedicationQA | ROUGE-L=**5.6%**, F1=10.7% | 666 | ✅ 완료 |
| | HealthSearchQA | ROUGE-L=**11.7%**, F1=22.4% | 3,077 | ✅ 완료 |
| | KQA Silver | ROUGE-L=**10.4%**, F1=21.2% | 904 | ✅ 완료 |
| EHR | MIMIC-III | — | — | ⏳ TextQA 완료 후 실행 |
| | eICU | — | — | ⏳ |

**B039 (03-20)**: B037 수정이 잘못 적용됨 → TextQA 10%로 붕괴
  - 원인: `add_generation_prompt=True`가 `<think>\n` 삽입하는데 strip 누락 + max_new_tokens=512
  - 수정: `<think>\n` strip + max_new_tokens=64 + `</think>` fallback strip
  - **TextQA 재실행 중** (GPU 4, B039 fix 적용)
- **B040 (03-20)**: PMC-VQA HF train_2.csv 스키마 불일치 수정 → data_files 파라미터 추가

---

## [Phase 2-B3] Autonomous GYM

### Dryrun (GPU 6) — 🔄 v5 진행 중 — B035 anti-herding fix
- Config: `configs/autonomous_gym_dryrun.yaml`
- PID: **2759172** → child **2759178**
- 로그: `logs/autonomous_gym_dryrun/launcher_v5_0317.log`
- v5 결과 (lease_0001~0047): **46 cycles**, 3개 도메인 분산, net avg +0.0057
  - clinical_diagnosis: 16 cycles, avg=+0.0121, 8/16 positive
  - drug_interaction: 15 cycles, avg=+0.0060, 8/15 positive
  - medical_qa: 15 cycles, avg=-0.0013, 4/15 positive

### GPU 5 Lingshu Weakness Fixer — 🔄 v5 진행 중
- Config: `configs/gym_gpu5_lingshu.yaml`
- PID: **2759173** → child **2759176**
- 로그: `logs/autonomous_gym/gpu5_lingshu_v5_0317.log`
- v5 결과 (lease_0001~0025): **24 cycles**, 5개 도메인 분산, net avg **+0.0039**
  - clinical_diagnosis: 7 cycles, avg=+0.0052, 3/7 positive
  - obstetrics: 8 cycles, avg=+0.0050, 4/8 positive
  - triage_emergency: 5 cycles, avg=-0.0001, 3/5 positive
  - psychiatry: 2 cycles, avg=+0.0020, drug_interaction: 2 cycles, avg=+0.0068

### GPU 7 Qwen2VL Weakness Fixer — 🔄 v5 진행 중 — ⚠️ net negative
- Config: `configs/gym_gpu7_qwen25vl.yaml`
- PID: **2759174** → child **2759177**
- 로그: `logs/autonomous_gym/gpu7_qwen25vl_v5_0317.log`
- v5 결과 (lease_0001~0018): **17 cycles**, 5개 도메인 분산, net avg **-0.0071**
  - clinical_diagnosis: 7/17 (41%) — avg=-0.0116 (peer_learning으로 과다 선택)
  - psychiatry: 3/3 positive (+0.0217) — 유일한 안정 positive 도메인
  - cross_domain: 0/2 positive (-0.0372) — 성능 하락 심각

---

## 코드 수정 이력

### 2026-03-14 — GRPO Degradation 근본 수정 (B027)

| 수정 | 파일 | 내용 |
|------|------|------|
| C-1 | `grpo_trainer.py` | `_compute_ref_log_probs()` 함수 추가 — LoRA disable로 reference log-probs 계산 |
| C-2 | `grpo_trainer.py:_grpo_policy_update()` | 실제 KL divergence penalty 추가 (`β * (log π_θ - log π_ref)`) — 기존 fake beta scaling 교체 |
| C-3 | `grpo_trainer.py:train_multiturn()` | ref log-probs 계산 후 `_grpo_policy_update(ref_log_probs=...)` 전달 |
| C-4 | `grpo_trainer.py:MultiTurnGRPOConfig` | `max_train_tasks: int = 20` 필드 추가 (overfitting 방지) |
| C-5 | `grpo_trainer.py:train_multiturn()` | task cap 로직 추가 (`max_train_tasks > 0`이면 random sample) |
| C-6 | `grpo_trainer.py:train_multiturn()` | cosine LR scheduler 추가 (warmup 10% + cosine decay) |
| C-7 | `grpo_trainer.py` | replay advantage 정규화 — 고정 `1.0` → rollout positive advantage 평균 |
| C-8 | `autonomous_agent.py:_evaluate()` | `eval_seed` 파라미터 추가 — pre/post eval 동일 task 보장 |
| C-9 | `autonomous_agent.py:run_workout()` | pre/post eval에 동일 `_eval_seed` 전달 |
| C-10 | `autonomous_agent.py:_get_reward_weights_from_benchmarks()` | accuracy 가중치 0.30→0.50, format 0.15→0.05 (reward hacking 방지) |
| C-11 | `autonomous_agent.py:AutonomousAgentConfig` | `training_epochs` 기본값 2→1, `learning_rate` 2e-5→5e-6, `max_train_tasks: 20` 추가 |
| C-12 | `configs/autonomous_gym_dryrun.yaml` | `learning_rate: 5e-6`, `max_train_tasks: 10` |
| C-13 | `configs/gym_gpu5_lingshu.yaml` | `training_epochs: 1`, `learning_rate: 5e-6`, `max_train_tasks: 20` |
| C-14 | `configs/gym_gpu7_qwen25vl.yaml` | 동일 |

### 2026-03-16 — B028~B033: GRPO silent fail + config 누락 + resource 충돌 수정

| 수정 | 파일 | 내용 | Bug |
|------|------|------|-----|
| D-1 | `grpo_trainer.py:109` | `BioAgentGRPOConfig`에 `max_train_tasks: int = 20` 필드 추가 | B028 |
| D-2 | `autonomous_gym.py:_run_subprocess()` | returncode==0일 때도 stderr 마지막 10줄 `logger.warning`으로 출력 | B029 |
| D-3 | `autonomous_gym.py:_build_cycle_script()` | 7개 필드 추가 (max_train_tasks, adaptive_quality, use_dr_grpo 등) | B030 |
| D-4 | `configs/autonomous_gym_dryrun.yaml` | eval_timeout_seconds 3600→7200, train_timeout 3600→7200 | B031 |
| D-5 | `scripts/run_full_benchmark_suite.py` (TextQA) | Step3-VL key_mapping + think-strip 추가 | B032 |
| D-6 | `scripts/run_full_benchmark_suite.py` (MedLFQA) | Step3-VL key_mapping + think-strip 추가 | B032 |
| D-7 | 메인 GYM (PID 2510494) | autonomous_gym.yaml 8-agent 프로세스 + orphaned workers 종료 | B033 |
| D-8 | GPU 5/6/7 | per-GPU config 프로세스 재시작 (v4, PID 2593880/2593881/2593882) | B030+B031 |
| D-9 | GPU 4 | Step3-VL TextQA 시작 (B032 fix 적용, PID 2596948 — hang으로 killed) | B032 |
| D-10 | `scripts/run_full_benchmark_suite.py` | `_load_step3vl_manual()` 헬퍼 함수 추가 — `no_init_weights` + 수동 safetensors 로딩 (Qwen3 init hang 근본 해결) | B032 final |
| D-11 | `scripts/run_full_benchmark_suite.py` (TextQA+MedLFQA) | `from_pretrained()` → `_load_step3vl_manual()` 호출로 교체 | B032 final |
| D-12 | GPU 4 | Qwen MedLFQA 재실행 (PID 2610744) — MedicationQA 진행 중 | — |

### 2026-03-18 — B036/B037/B038: Step3-VL batch size + thinking + model_profile 수정

| 수정 | 파일 | 내용 | Bug |
|------|------|------|-----|
| F-1 | `scripts/run_full_benchmark_suite.py:258` | MedLFQA `BATCH_SIZE = 1 if is_step3 else 8` | B036 |
| F-2 | `scripts/run_full_benchmark_suite.py:500` | TextQA `BATCH_SIZE = 1 if is_step3 else 8` | B036 |
| F-3 | `scripts/run_full_benchmark_suite.py:518-520` | TextQA: Step3 `<think>\n` generation prompt 제거 | B037 |
| F-4 | `scripts/run_full_benchmark_suite.py:533` | TextQA: `gen_max_tokens = 64 if is_step3 else 32` | B037 |
| F-5 | `scripts/run_full_benchmark_suite.py:277-279` | MedLFQA: Step3 `<think>\n` generation prompt 제거 | B037 |
| F-6 | `bioagents/gym/model_profile.py:261-268` | `load_model()`: `model_type=="step_robotics"` → `_load_step3vl_manual()` 분기 | B038 |
| F-7 | `bioagents/gym/model_profile.py:295-349` | `_load_step3vl_manual()` 메서드 추가 (key_mapping + no_init_weights + cpu_only 지원) | B038 |

### 2026-03-20 — B039/B040: Step3 TextQA B037 regression + PMC-VQA 로딩 수정

| 수정 | 파일 | 내용 | Bug |
|------|------|------|-----|
| G-1 | `scripts/run_full_benchmark_suite.py:525-527` | TextQA: Step3 `<think>\n` prompt strip 추가 | B039 |
| G-2 | `scripts/run_full_benchmark_suite.py:540` | TextQA: `gen_max_tokens = 64 if is_step3 else 32` (512→64) | B039 |
| G-3 | `scripts/run_full_benchmark_suite.py:550-552` | TextQA: `</think>` fallback strip 추가 | B039 |
| G-4 | `scripts/run_full_benchmark_suite.py:278-280` | MedLFQA: `<think>\n` prompt strip 추가 | B039 |
| G-5 | `scripts/run_full_benchmark_suite.py:303-305` | MedLFQA: `</think>` fallback strip 추가 | B039 |
| G-6 | `bioagents/data_pipeline/vqa_loader.py:370` | PMC-VQA `data_files` 파라미터 추가 (train_2.csv 스키마 불일치 회피) | B040 |

### 2026-03-17 — B034/B035: Step3 import 누락 + domain herding 근본 수정

| 수정 | 파일 | 내용 | Bug |
|------|------|------|-----|
| E-1 | `scripts/run_full_benchmark_suite.py:97-99` | `_load_step3vl_manual()`에 `import torch`, `from transformers import AutoConfig, AutoModelForCausalLM` 추가 | B034 |
| E-2 | `bioagents/gym/shared_logbook.py:276` | weakness threshold `0.1` → `max(avg * 0.15, 0.02)` 상대적 threshold | B035 |
| E-3 | `bioagents/gym/autonomous_agent.py:_score_domain()` | weakness 점수를 이산(0.8) → 연속 비례(`0.5 + (avg-score)/avg`)로 변경 | B035 |
| E-4 | `bioagents/gym/autonomous_agent.py:_score_domain()` | recency penalty 추가: 최근5회 중 3+회 동일 도메인 → 0.3배, 2회 → 0.6배 | B035 |
| E-5 | GPU 5/6/7 | GYM v5 프로세스 재시작 (PID 2759172/2759173/2759174) | B035 |
| E-6 | GPU 4 | Step3-VL TextQA 시작 (PID 2760873, B034 fix 적용) → MedLFQA 자동 순차 실행 | B034 |

### 2026-03-12

| 수정 | 파일 | 내용 | Bug |
|------|------|------|-----|
| B-1 | `configs/gym_gpu5_lingshu.yaml` | `eval_timeout_seconds: 3600`, `train_timeout_seconds: 3600` 추가 | B021 |
| B-2 | `configs/gym_gpu7_qwen25vl.yaml` | 동일 | B021 |
| B-3 | `scripts/run_all_benchmarks_parallel.py:533` | `{EHR_MAX_SAMPLES}` → `{{EHR_MAX_SAMPLES}}` 이스케이프 | B022 |
| B-4 | `bioagents/evaluation/ehr_benchmark_eval.py:191` | `set(sampled)` → `{id(t) for t in sampled}` 비교 | B023 |
| B-5 | `bioagents/evaluation/vqa_benchmark_eval.py:236-240` | Step3-VL `key_mapping` 명시적 전달 (가중치 랜덤초기화 방지) | B024 |
| B-6 | `bioagents/evaluation/vqa_benchmark_eval.py:538` | `<think>...</think>` 제거 후처리 추가 | B025 |
| B-7 | `bioagents/gym/autonomous_agent.py:~1159-1195` | Adaptive quality filtering (score normalization + ZPD 선택) | B026 |
| B-8 | `bioagents/gym/autonomous_agent.py:~113` | `AutonomousAgentConfig`에 `adaptive_quality*` 필드 3개 추가 | B026 |
| B-9 | `configs/gym_gpu5_lingshu.yaml` | `eval_tasks_per_domain: 5`, `adaptive_quality: true`, `quality_threshold: 0.1` | B026 |
| B-10 | `configs/gym_gpu7_qwen25vl.yaml` | 동일 | B026 |
| B-11 | `bioagents/evaluation/ehr_benchmark_eval.py` | `max_samples` 기본값 `0` → `5000` | B016 |
| B-12 | `configs/baseline_eval.yaml` | `max_samples: 0` → `5000` | B016 |

### 2026-03-11

| 수정 | 파일 | 라인 | 내용 |
|------|------|------|------|
| A-1a | `bioagents/gym/autonomous_agent.py` | ~1094 | `_evaluate()`: `eval_tasks_per_domain` → random sample → `RunConfig(task_ids=sampled_task_ids)` |
| A-1b | `bioagents/gym/autonomous_gym.py` | `AutonomousGymConfig` | `eval_timeout_seconds: int = 86400`, `train_timeout_seconds: int = 7200` 필드 추가 |
| A-1c | `bioagents/gym/autonomous_gym.py` | `AgentWorker.__init__` | `eval_timeout`, `train_timeout` 인자 추가 및 `_run_subprocess()` 호출에 적용 |
| A-1d | `bioagents/gym/autonomous_gym.py` | `AutonomousGym._dispatch_workers` | `AgentWorker(eval_timeout=config.eval_timeout_seconds, ...)` 전달 |
| A-1e | `configs/autonomous_gym_dryrun.yaml` | gym 섹션 | `eval_timeout_seconds: 3600`, `train_timeout_seconds: 3600` 추가 |
| A-2 | `configs/autonomous_gym.yaml` | step3 에이전트 3개 | `gpu_memory_utilization: 0.70`, `inference_batch_size: 1` 각각 추가 |
| A-3a | `bioagents/evaluation/ehr_benchmark_eval.py` | `load_benchmark()` | truncation → category stratified sampling 교체 |
| A-3b | `bioagents/evaluation/ehr_benchmark_eval.py` | `evaluate_benchmark()` | 100 태스크마다 checkpoint 저장 + resume 로직 추가 |
| A-3c | `scripts/run_all_benchmarks_parallel.py` | `run_ehr()` | `max_samples=0` → `EHR_MAX_SAMPLES=5000` 변경 |

---

## TODO (다음 세션 우선순위)

- [x] **[완료]** GPU5/7 gym timeout 설정 + 재시작 (B021)
- [x] **[완료]** WORKER_SCRIPT 이스케이프 수정 (B022)
- [x] **[완료]** Step3-VL VQA 재실행 (B019 — model_type 수정 코드 적용, GPU 4)
- [x] **[완료]** EHR baseline 재시작 (stratified 5K, 3모델)
- [x] **[완료]** B020 분석 — WON'T FIX (open-ended VQA, token_F1 사용)
- [x] **[완료]** Step3-VL VQA EM=0 근본 원인 해결 (B024: key_mapping, B025: think-strip)
- [x] **[완료]** Step3-VL VQA v3 재실행 (PID 1744651, GPU 4) — EM=0.200 at 50/451 확인!
- [x] **[완료]** GPU5/7 gym 사이클 진행 확인 (lease_0003 도달)
- [x] **[완료]** Step3-VL VQA v3 완료 (VQA-RAD EM=15.3%, PathVQA EM=7.4%, PMC-VQA 로딩 실패)
- [x] **[완료]** GRPO degradation 근본 수정 — KL penalty, task cap, LR 감소, reward rebalance 등 7개 수정 (B027)
- [x] **[완료]** GPU 5/6/7 프로세스 재시작 (수정 코드 적용, PID 2130038/2130181/2130329)
- [ ] **[진행 중]** EHR 5K baseline (Lingshu 1807/5000=36%, Qwen 666/5000=13%) — 로그: `logs/ehr_baseline_20260312/`
- [x] **[완료]** B028/B029 수정 — max_train_tasks 필드 추가 + stderr 로깅 (03-16 AM)
- [x] **[완료]** B030 수정 — _build_cycle_script() 7개 필드 누락 (03-16 PM)
- [x] **[완료]** B031 수정 — dryrun eval_timeout 3600→7200 (03-16 PM)
- [x] **[완료]** B032 수정 — Step3-VL TextQA/MedLFQA key_mapping + think-strip (03-16 PM)
- [x] **[완료]** B033 수정 — autonomous_gym.yaml 8-agent 프로세스 충돌 해결 (03-16 PM)
- [x] **[완료]** GPU 5/6/7 프로세스 재시작 (v4, PID 2593880/2593881/2593882)
- [x] **[완료]** GRPO v3 training 실행 확인 (GPU5: +0.003 ehr, GPU7: +0.004 drug)
- [x] **[완료]** B032 최종 해결 — Qwen3 init hang 원인 발견 + `_load_step3vl_manual()` workaround 구현
- [x] **[완료]** GRPO v4 첫 사이클 결과 확인 — Dryrun: medical_qa +9.7% (!), Lingshu: drug_interaction -2.0%
- [x] **[완료]** Qwen MedLFQA 완료 (GPU 4, 5개 dataset 전부 완료, 03-16 18:34)
- [x] **[완료]** B036 수정 — Step3-VL TextQA/MedLFQA BATCH_SIZE=1 (batch>1 시 4D tensor crash)
- [ ] **[진행 중]** Step3-VL TextQA+MedLFQA 재실행 (GPU 4, B036 fix 적용)
- [ ] Step3-VL EHR 실행 (GPU 4)
- [ ] GRPO v4 lease_0002+ 결과 확인
- [ ] PMC-VQA HuggingFace 로딩 에러 수정
- [x] **[완료]** B039 수정 — Step3-VL TextQA `<think>\n` strip + max_new_tokens=64 + `</think>` fallback (03-20)
- [x] **[완료]** B040 수정 — PMC-VQA data_files 파라미터 추가 (03-20)
- [ ] **[진행 중]** Step3-VL TextQA B039 fix 재실행 (GPU 4, PID 3305001)
- [x] **[완료]** GYM 에이전트 중단 (GPU 5/6/7) — 외부 벤치마크 검증 위해 (03-20)
- [x] **[완료]** GYM External Validation — GPU 5: GYM-Lingshu TextQA → MedQA 56.9%(-4.8%), MedMCQA 60.8%(+7.6%), MMLU 66.8%(-10.1%)
- [x] **[완료]** GYM External Validation — GPU 6: GYM-Qwen2VL (late) TextQA → MedQA 10.7% (💀 모델 파괴, B041)
- [x] **[완료]** GYM External Validation — GPU 7: GYM-Qwen2VL (early, 첫 merge) TextQA → MedQA 10.5% (💀 첫 merge부터 파괴)
- [x] **[완료]** GYM External Validation — 결과 집계 완료: Lingshu mixed, Qwen catastrophic → B041 기록, 대안 A/B/C/D 정리
- [ ] **[보류]** GYM External Validation — MedLFQA (대안 결정 후 진행 여부 판단, 03-21)
- [ ] **[보류]** GPU 5/6/7 GYM 재시작 (B041 수정 or 논문 scope 결정 후)
- [ ] 전체 baseline 완료 후 → `scripts/aggregate_baseline_report.py` 실행
- [ ] Ablation study 설계 (5D 보상 컴포넌트 × 전략 비교)
