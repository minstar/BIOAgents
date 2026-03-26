# BIOAgents - Summary & Planning

> 전체 프로젝트 진행 요약, 컴포넌트 완성도, 연구 로드맵.
> Claude가 context 없이도 즉시 전략적 현황을 파악하기 위한 reference.
> 세부 수치/로그는 `progress_check.md` 참조.

---

## Last Updated: 2026-03-20 16:00 KST

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
| Phase 2-B3: Autonomous GYM (3 agents) | 03/10~ | ⏸️ 외부 벤치마크 검증 위해 일시 중단 | 25% |
| **Phase 2.5: GYM External Validation** | **03/20** | **⚠️ 완료 — mixed/negative 결과, 전략 결정 필요** | **80%** |
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
| **GRPO v2: KL penalty (reference model)** | ✅ NEW (03-14) | `grpo_trainer.py:_compute_ref_log_probs()`, `_grpo_policy_update()` |
| **GRPO v2: task cap (max_train_tasks=20)** | ✅ NEW (03-14) | `grpo_trainer.py:MultiTurnGRPOConfig` |
| **GRPO v2: cosine LR scheduler** | ✅ NEW (03-14) | `grpo_trainer.py:train_multiturn()` |
| **GRPO v2: pre/post eval 동일 task (eval_seed)** | ✅ NEW (03-14) | `autonomous_agent.py:_evaluate()` |
| **GRPO v2: accuracy-dominant reward (50%)** | ✅ NEW (03-14) | `autonomous_agent.py:_get_reward_weights_from_benchmarks()` |

---

## 현재 실행 중인 실험 (2026-03-20 15:30 기준)

| 실험 | 모델 | GPU | 상태 | PID |
|------|------|-----|------|-----|
| EHR baseline (5K stratified) | Lingshu+Qwen | 0-3 | 🔄 Lingshu 89%, Qwen 74% | 1727391 |
| Step3-VL TextQA 재실행 (B039 fix) | Step3-VL-10B | 4 | 🔄 진행 중 | **3305001** |
| GYM External Eval: TextQA | GYM-Lingshu-7B | 5 | ✅ 완료 — MedQA 56.9%, MedMCQA 60.8%, MMLU 66.8% | — |
| GYM External Eval: TextQA | GYM-Qwen2.5-VL-7B (late) | 6 | ❌ 중단 — MedQA 10.7% (모델 파괴, B041) | — |
| GYM External Eval: TextQA | GYM-Qwen2.5-VL-7B (early) | 7 | ❌ 중단 — MedQA 10.5% (첫 merge부터 파괴) | — |
| GPU 5/6/7 | — | 5,6,7 | ⏸️ 유휴 — 전략 결정 대기 중 | — |

---

## 알려진 미해결 이슈

| 이슈 | 영향 | 상태 |
|------|------|------|
| ~~GRPO degradation~~ | ~~자율 학습 루프 동작 불가~~ | ✅ B027 FIXED (03-14) |
| ~~GRPO training silent fail~~ | ~~03-14~16 training 0건 실행~~ | ✅ B028+B029 FIXED (03-16) |
| ~~Step3 TextQA/MedLFQA import 누락~~ | ~~NameError 크래시~~ | ✅ B034 FIXED (03-17) |
| ~~Domain herding (85% 단일 도메인)~~ | ~~학습 루프 다양성 상실~~ | ✅ B035 FIXED (03-17) |
| ~~Step3-VL BATCH_SIZE 크래시~~ | ~~4D tensor crash~~ | ✅ B036 FIXED (03-18) |
| ~~Step3-VL TextQA thinking 잘림~~ | ~~MedMCQA 21% < random~~ | ✅ B037 FIXED (03-18) |
| ~~model_profile Step3 key_mapping 누락~~ | ~~EHR action_score=0.000~~ | ✅ B038 FIXED (03-18) |
| ~~Step3-VL TextQA B037 regression~~ | ~~51%→10% 붕괴~~ | ✅ B039 FIXED (03-20) — `<think>\n` strip + max_new_tokens=64 + `</think>` fallback |
| ~~PMC-VQA HuggingFace 로딩 실패~~ | ~~Step3 VQA 1개 결과 없음~~ | ✅ B040 FIXED (03-20) — data_files 파라미터 추가 |
| Step3-VL TextQA 재실행 중 | B039 fix 적용 결과 확인 필요 | 🔄 GPU 4 실행 중 (PID 3305001) |
| Step3-VL EHR 미시작 | B038 fix 적용 후 실행 필요 | ⏳ TextQA 완료 후 |
| GPU7 Qwen v5 net negative | clinical_diagnosis 과다 선택 | ⚠️ 모니터링 중 |
| **B041: GRPO merge Qwen 모델 파괴** | **Qwen MedQA 54%→10%, early ckpt도 동일** | **🔴 CRITICAL — merge 프로세스 디버깅 필요** |
| **B041: Lingshu catastrophic forgetting** | **MedMCQA +7.6% 향상, MedQA -4.8%, MMLU -10.1%** | **⚠️ KL penalty 강화 또는 EWC 필요** |

---

## 다음 액션 (우선순위 순)

### [완료] 2026-03-12 실행한 작업
- ✅ GPU5/7 gym config에 `eval_timeout_seconds: 3600` 추가 + 재시작
- ✅ WORKER_SCRIPT `{EHR_MAX_SAMPLES}` 이스케이프 수정 (B022)
- ✅ 기존 baseline eval 프로세스 종료 (EHR만 남아있었음)
- ✅ EHR baseline 재시작 (stratified 5K, 3모델)
- ✅ Step3-VL VQA 재실행 (수정된 model_type 감지 코드 적용)

### [CRITICAL, 03-20] Phase 2.5: GYM External Validation — 완료, 전략 결정 필요

#### 결과 테이블

| 모델 | Benchmark | Baseline | GYM-trained | Delta | 판정 |
|------|-----------|----------|-------------|-------|------|
| Lingshu | MedQA (1273) | 61.7% | 56.9% | -4.8% | ❌ |
| Lingshu | MedMCQA (4183) | 53.2% | 60.8% | **+7.6%** | ✅ |
| Lingshu | MMLU 6-subset (1089) | 76.9% | 66.8% | -10.1% | ❌ |
| Qwen (late, 110 cycles) | MedQA | 54.0% | 10.7% | -43.3% | 💀 |
| Qwen (early, 1st merge) | MedQA | 54.0% | 10.5% | -43.5% | 💀 |

#### 핵심 진단
1. **Qwen GRPO merge 완전 파괴 (B041)**: 첫 번째 merge부터 10% → merge 프로세스 자체 버그
2. **Lingshu catastrophic forgetting**: MedMCQA만 +7.6%, MedQA/MMLU는 대폭 하락 → KL penalty 부족

#### 대안 (03-21 결정 예정)

**대안 A: GRPO merge 버그 수정 후 재실험 (2~3주)**
- Qwen LoRA target_modules 확인 → vision encoder에 LoRA 적용되고 있으면 제거
- merge alpha/scaling 검증, KL penalty 10x 강화 또는 EWC 추가
- 수정 후 짧은 GYM run (10 cycles) → 외부 벤치마크 즉시 검증
- 장점: self-evolving loop contribution 살릴 수 있음
- 단점: 시간 리스크, 수정해도 개선 보장 없음

**대안 B: 논문 scope 축소 — "환경 + 벤치마크" contribution만 (가장 안전)**
- Healthcare AI GYM (10 domains × 171 tools × 2,580+ tasks) + 5D Reward + FairGRPO
- 21 external benchmark × 3 model baseline 결과만으로 논문 작성
- self-evolving loop은 "future work"으로 → MedMCQA +7.6% 등 preliminary result만 언급
- 장점: 지금 데이터로 즉시 논문 작성 가능, deadline 안전
- 단점: contribution 약화 가능성 (환경 논문 = "incremental" 리스크)

**대안 C: LoRA adapter 원본으로 평가 (빠른 1시간 실험)**
- `best_merged` 대신 LoRA adapter 원본을 PeftModel로 로드하여 평가
- merge 과정 버그 vs 학습 자체 문제 1시간 안에 판별
- merge 버그 맞으면 → 수정해서 대안 A 진행
- 학습 자체 문제면 → 대안 B로 전환

**대안 D: 가중치 수정 없는 접근 — Few-shot / RAG (방향 전환)**
- GYM 성공 trajectory를 few-shot example로 활용 (ICL)
- catastrophic forgetting 원천 차단
- 단점: "self-evolving RL loop" narrative와 안 맞음

**추천 순서**: C(진단) → A 또는 B(결정) — 내일(03-21) 결정

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
