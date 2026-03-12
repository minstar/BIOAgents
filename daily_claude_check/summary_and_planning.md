# BIOAgents - Summary & Planning

> 전체 프로젝트 진행 요약, 컴포넌트 완성도, 연구 로드맵.
> Claude가 context 없이도 즉시 전략적 현황을 파악하기 위한 reference.
> 세부 수치/로그는 `progress_check.md` 참조.

---

## Last Updated: 2026-03-12 21:00 KST

---

## 프로젝트 개요

**목표**: Healthcare AI GYM — 의료 AI 에이전트가 자율적으로 self-improve하는 RL 훈련 환경 구축.
**논문 타겟**: NeurIPS 2026 (main track, ~June 2026) / ICLR 2026 Workshop (04/26-27, ~46일 남음)
**핵심 차별점**: 10 도메인 × 187 tools × 2,580+ tasks, 5D 보상, Autonomous self-evolving loop, FairGRPO, Dr. GRPO, Adaptive Quality Filtering

---

## Phase 현황

| Phase | 기간 | 상태 | 완성도 |
|-------|------|------|--------|
| Phase 1: 인프라 구축 | 02/12~02/15 | ✅ 완료 | 100% |
| Phase 1.5: 고도화 | 02/15~02/28 | ✅ 완료 | 95% |
| Phase 2-B2: Baseline 평가 (21 bench × 3 model) | 03/01~03/15 | 🔄 진행 중 | 65% |
| Phase 2-B3: Autonomous GYM (3 agents) | 03/10~ | 🔄 GPU5/6/7 사이클 순환 중 | 25% |
| Phase 3: Ablation + 결과 집계 | 03/15~03/25 | ⏳ 미시작 | 0% |
| Phase 4: 논문 작성 | 03/25~04/20 | ⏳ 미시작 | 0% |

---

## 컴포넌트 완성도

| 컴포넌트 | 상태 | 핵심 파일 |
|---------|------|----------|
| 10개 의료 도메인 (clinical_diagnosis, drug_interaction, ehr_management, medical_qa, obstetrics, psychiatry, radiology_report, triage_emergency, visual_diagnosis, cross_domain) | ✅ | `bioagents/domains/` |
| 171 clinical tools | ✅ | 각 도메인 `tools.py` |
| 2,580+ tasks (train/test split 완료) | ✅ | `data/domains/*/tasks*.json`, `split_tasks.json` |
| 5D Reward System (Accuracy 30%, Process 25%, Safety 20%, Format 15%, Coherence 10%) | ✅ | `bioagents/evaluation/rewards.py`, `grpo_rewards.py` |
| Reward Strategies (GRPO, MRPO, SARL, DRPO, CRPO, FAIR_GRPO, ADAPTIVE) | ✅ | `bioagents/evaluation/reward_strategies.py` |
| FairGRPO (demographic fairness-aware training) | ✅ | `bioagents/training/grpo_trainer.py` |
| Dr. GRPO (per-token length normalization bias 제거) | ✅ | `grpo_trainer.py` + `use_dr_grpo` flag |
| WebRL-style Eval→Train Replay (30% mix ratio) | ✅ | `grpo_trainer.py:eval_trajectories` |
| Agent0-style LLM Task Generation | ✅ | `bioagents/gym/gym_coach.py:CurriculumAgent.generate_tasks_from_specs()` |
| Anti-contamination Guard | ✅ | `autonomous_agent.py:_EXTERNAL_TEST_DOMAINS` frozenset |
| Patient Agent (30 cases, 12 personality, 13 bias) | ✅ | `bioagents/agents/patient_agent.py`, `data/clinical_cases_v2.json` |
| Autonomous GYM (8-agent, SelfAwareness, StrategySelector, SharedLogbook) | ✅ | `bioagents/gym/` |
| W&B 통합 | ✅ | 프로젝트명: `pt2-minstar-gym-rl` |
| 21 벤치마크 평가 인프라 | ✅ | `scripts/run_all_benchmarks_parallel.py` |
| EHR 평가 checkpoint/resume | ✅ NEW (03-11) | `bioagents/evaluation/ehr_benchmark_eval.py` |
| EHR stratified sampling (57K → 5K) | ✅ NEW (03-11) | `ehr_benchmark_eval.py:load_benchmark()` |
| Configurable eval/train timeout (dryrun용) | ✅ NEW (03-11) | `autonomous_gym.py:AutonomousGymConfig` |
| eval_tasks_per_domain 실제 적용 | ✅ NEW (03-11) | `autonomous_agent.py:_evaluate()` |
| Adaptive Quality Filtering (ZPD 기반 trajectory 선택) | ✅ NEW (03-12) | `autonomous_agent.py:_evaluate()` lines ~1159-1195 |
| Step3-VL key_mapping + think-strip | ✅ NEW (03-12) | `vqa_benchmark_eval.py` |
| EHR max_samples 기본값 5000 | ✅ NEW (03-12) | `ehr_benchmark_eval.py`, `baseline_eval.yaml` |

---

## 현재 실행 중인 실험 (2026-03-12 21:00 기준)

| 실험 | 모델 | GPU | 상태 | PID |
|------|------|-----|------|-----|
| EHR baseline (5K stratified) | Lingshu+Qwen+Step3 | 0-4 | 🔄 진행 중 (10h+) | 1727391 |
| Step3-VL VQA v3 (B024+B025 수정) | Step3-VL-10B | 4 | 🔄 실행 중 (9h+) | 1744651 |
| GYM Lingshu weakness_fixer | Lingshu-7B | 5 | 🔄 lease_0010 eval, adaptive_quality ON | **1831616** |
| GYM Dryrun | Qwen2.5-VL-7B | 6 | ✅ 정상 순환 (lease_0039+) | 1323866 |
| GYM Qwen2VL weakness_fixer | Qwen2.5-VL-7B | 7 | 🔄 lease_0008 eval, adaptive_quality ON | **1832074** |

---

## 알려진 미해결 이슈

| 이슈 | 영향 | 상태 |
|------|------|------|
| ~~Step3-VL EM=0~~ | ~~baseline 결과 신뢰도 저하~~ | ✅ B024+B025 FIXED — key_mapping 전달 + `<think>` 제거, VQA v3 실행 중 (EM=0.200@50) |
| ~~VQA-Med-2021 / Quilt-VQA EM=0~~ | ~~2 benchmark 결과 없음~~ | ℹ️ B020 WON'T FIX — open-ended VQA 특성, token_F1 사용 |
| ~~MIMIC-III 진행 느림~~ | ~~EHR baseline 몇 주 소요~~ | ✅ B016 FIXED — stratified 5K로 재시작됨 (03-12) |
| ~~GPU5/7 gym stuck~~ | ~~self-evolving loop 멈춤~~ | ✅ B021 FIXED — timeout 설정 후 재시작됨 (03-12) |
| ~~GPU5/7 학습 데이터 0건~~ | ~~quality_threshold 미달~~ | ✅ B026 FIXED — adaptive quality filtering 구현 (03-12 20:20) |
| Qwen MedLFQA 미완료 | 5개 중 일부 미완료 | ⚠️ 기존 프로세스 killed — 별도 재실행 필요할 수 있음 |
| Step3-VL VQA + EHR GPU 4 충돌 가능 | 동시 실행 시 OOM | ⚠️ 모니터링 필요 (A100 80GB이므로 가능할 수도 있음) |

---

## 다음 액션 (우선순위 순)

### [완료] 2026-03-12 실행한 작업
- ✅ GPU5/7 gym config에 `eval_timeout_seconds: 3600` 추가 + 재시작
- ✅ WORKER_SCRIPT `{EHR_MAX_SAMPLES}` 이스케이프 수정 (B022)
- ✅ 기존 baseline eval 프로세스 종료 (EHR만 남아있었음)
- ✅ EHR baseline 재시작 (stratified 5K, 3모델)
- ✅ Step3-VL VQA 재실행 (수정된 model_type 감지 코드 적용)

### [이번 주, ~3/15] Phase A 완료
- [ ] Dryrun 1사이클 완주 확인 → SharedLogbook 데이터 확인
- [ ] MIMIC-III / eICU stratified 5K 완료 확인
- [ ] Step3-VL EM=0 디버깅 (sample output 3-5개 출력 → 파싱 방식 확인)
- [ ] VQA-Med-2021 / Quilt-VQA EM=0 원인 파악

### [3/15~3/25] Phase B: 결과 집계 & Ablation
- [ ] `scripts/aggregate_baseline_report.py` 실행 → `results/baseline_20260315/summary_table.csv`
- [ ] Ablation configs 작성: `configs/ablation_5d_reward.yaml`, `configs/ablation_strategy.yaml`
- [ ] 5D 보상 ablation (5 variant × clinical_diagnosis × Lingshu × 100 steps)
- [ ] Reward strategy ablation (GRPO / MRPO / SARL / Dr.GRPO / FairGRPO)

### [3/25~4/20] Phase C: 논문 작성 (ICLR Workshop 타겟)
- [ ] Figure 1: GYM Architecture diagram
- [ ] Figure 2: Autonomous self-evolving loop
- [ ] Table 1: 21 benchmark baseline (3 models)
- [ ] Table 2: 5D reward ablation
- [ ] Abstract + Introduction (3 contributions 명확히)
- [ ] System Description + Experiments 섹션

### [4/1~5/1] Phase D: Full 30-day Autonomous GYM
- [ ] 8 GPUs × 8 agents × continuous=true
- [ ] 30일 자동 실행 (W&B 모니터링)
- [ ] 각 도메인 10%+ 성능 향상 타겟

---

## 핵심 아키텍처 결정 사항

1. **VL 모델 로딩**: `AutoModelForVision2Seq` (CausalLM 금지 — BUG-006)
2. **model_type 비교**: substring 매칭 `"vl" in model_type.lower()` (정확 매칭 금지 — BUG-007)
3. **GYM task**: `data/domains/*/tasks*.json`의 train split만 사용 (test split 완전 금지 — BUG-001~004)
4. **외부 벤치마크**: eval 전용, 학습/replay 절대 불가 (`_EXTERNAL_TEST_DOMAINS` 가드)
5. **Python 환경**: `.venv/bin/python` (conda base 절대 금지)
6. **MIMIC/eICU**: stratified 5K 샘플링 (전체 57K 실행 금지)
7. **EHR eval**: 100 task마다 checkpoint 저장 (crash recovery 보장)

---

## 논문 포지셔닝

| 비교 대상 | 차별점 |
|-----------|--------|
| MedAgentGym | tool-use + multimodal + cross-domain + safety/fairness RL |
| DiagGym | 10개 도메인, 171 tools, autonomous self-evolving loop |
| Agent Hospital | formal benchmark (21개), reproducible RL training |
| MedBench 류 | evaluation이 아닌 training + evaluation 통합 시스템 |

**3 Core Contributions**:
1. Healthcare AI GYM: 10 clinical domains × 171 tools × 2,580+ tasks (reproducible medical RL environment)
2. 5D Reward System with demographic fairness (FairGRPO): accuracy + process + safety + format + coherence
3. Autonomous self-evolving loop: evaluate → reflect → generate tasks → train → repeat (without human intervention)
