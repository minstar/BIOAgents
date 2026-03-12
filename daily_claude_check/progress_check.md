# BIOAgents - Detailed Progress Check

> `summary_and_planning.md`와 연동. 수치 · 로그 · 실행 상태 · 실패 이력을 추적.
> Claude가 context 없이도 즉시 현황을 파악하기 위한 reference.

---

## Last Updated: 2026-03-12 21:00 KST

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
| EHR | MIMIC-III | action_score=**40.0%** | **830**/57,151 | 🔄 1.5% 진행 |
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
| MedLFQA | KQA Golden~KQA Silver | (진행 중) | — | 🔄 진행 중 |
| Vision QA | VQA-RAD | EM=**39.5%**, F1=43.9% | 451 | ✅ 완료 |
| | SLAKE | EM=**20.6%**, F1=23.7% | 1,061 | ✅ 완료 |
| | PathVQA | EM=**30.7%**, F1=31.7% | 6,719 | ✅ 완료 |
| | VQA-Med-2021 | EM=**0.0%**, F1=5.4% | 500 | ✅ ⚠️ 이슈 |
| | Quilt-VQA | EM=**0.0%**, F1=25.6% | 326 | ✅ 완료 |
| EHR | MIMIC-III | action_score=**49.4%** | **550**/57,151 | 🔄 1.0% 진행 |
| | eICU | — | — | ⏳ |

---

### 🔴 Step3-VL-10B (GPU 4+) — 심각한 품질 이슈

| 카테고리 | 벤치마크 | Score | N | 상태 |
|---------|---------|-------|---|------|
| Vision QA | VQA-RAD | EM=**0.000**, contains=27.9% | 451 | ✅ ❌ |
| | SLAKE | EM=**0.000**, contains=19.3% | 1,061 | ✅ ❌ |
| | PathVQA | EM=**0.000**, F1≈0.000 | ~1,050/6,719 | 🔄 15.6% |
| | VQA-Med-2021 | — | — | ⏳ |
| Text MC QA | (VQA 완료 후 시작) | — | — | ⏳ |
| EHR | — | — | — | ⏳ |

**✅ Step3 EM=0 원인 해결됨 (B024 + B025)**:
- **B024**: `_checkpoint_conversion_mapping`이 HF `VLMS` 리스트 미포함으로 미적용 → 가중치 랜덤 초기화
  - 수정: `key_mapping` 인자 명시적 전달
- **B025**: `<think>...</think>` 추론 텍스트가 예측에 포함 → EM 매칭 실패
  - 수정: `re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)` 후처리
- **결과**: VQA-RAD 50샘플 EM=0.200 확인 (0.000 → 0.200)
- v3 재실행 중 (PID 1744651, GPU 4)

---

## [Phase 2-B3] Autonomous GYM

### Dryrun (GPU 6) — ✅ 정상 동작 중
- Config: `configs/autonomous_gym_dryrun.yaml`
- Agent: `dryrun_qwen25vl` (Qwen2.5-VL-7B-Instruct)
- lease_0018+ 순환 중, 사이클 정상 완주 확인
- drug_interaction 도메인 improvement +5.2% 기록

### GPU 5 Lingshu Weakness Fixer — 🔄 4차 재시작 (03-12 20:20)
- Config: `configs/gym_gpu5_lingshu.yaml`
- PID: **1831616**
- **수정사항**: `eval_tasks_per_domain: 5`, `adaptive_quality: true` (B026 fix), `quality_threshold: 0.1` fallback
- 로그: `logs/autonomous_gym/gpu5_lingshu_restart_0312d.log`
- 현재: lease_0010 eval 진행 중, GPU 100% util
- Best score: 0.301 (clinical_diagnosis), 대부분 도메인 0.0 (premature_stop)

### GPU 7 Qwen2VL Weakness Fixer — 🔄 4차 재시작 (03-12 20:20)
- Config: `configs/gym_gpu7_qwen25vl.yaml`
- PID: **1832074**
- **수정사항**: `eval_tasks_per_domain: 5`, `adaptive_quality: true` (B026 fix), `quality_threshold: 0.1` fallback
- 로그: `logs/autonomous_gym/gpu7_qwen25vl_restart_0312d.log`
- 현재: lease_0008 eval 진행 중, GPU 100% util
- Best score: 0.286 (cross_domain)

---

## 코드 수정 이력

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
- [ ] Step3-VL VQA v3 완료 대기 (6 benchmarks, 로그: `logs/step3vl_vqa_rerun_v3_20260312/`)
- [ ] EHR 5K 진행 확인 (로그: `logs/ehr_baseline_20260312/`)
- [ ] Qwen MedLFQA 완료 확인 (기존 프로세스 killed — 별도 재실행 필요할 수 있음)
- [ ] Step3-VL TextQA + MedLFQA 실행 (VQA 완료 후)
- [ ] 전체 baseline 완료 후 → `scripts/aggregate_baseline_report.py` 실행
- [ ] Ablation study 설계 (5D 보상 컴포넌트 × 전략 비교)
