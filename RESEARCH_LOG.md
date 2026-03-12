# 최근 연구 로그 (AI Healthcare GYM / Agent GYM 관련)

> **목적**: Healthcare AI GYM 및 에이전트/벤치마크 관련 최신 논문 수집·정리  
> **최종 갱신**: 2026-03-10

---

## 1. 사용자 수집 논문 (2026년 2월)

### (1) CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion

- **URL**: https://arxiv.org/abs/2602.10999
- **요약**: 에이전트가 환경 히스토리를 시뮬레이션·탐색하고, 실행 피드백을 바탕으로 “정상 환경 히스토리”를 역추적해 런타임 실패가 있는 이전 상태로 역전시킨 뒤, 버그 상태와 에러 메시지를 결합해 **환경 집약적 CLI 태스크**를 대규모로 생성하는 파이프라인.
- **주요 결과**:
  - **1,655개** 환경 집약적 태스크 생성 (동종 최대 규모).
  - 파인튜닝 모델 **LiberCoder**: Terminal-Bench에서 **+21.1% (46.1%)** 달성.
- **Healthcare AI GYM 연관**:
  - **태스크 자동 생성**: Dockerfile·에이전트 태스크 유추 → 우리 GYM의 `auto_task_generator`, `generate_tasks_llm.py`와 유사한 “환경 역전” 아이디어 적용 가능 (의료 시나리오·실패 상태에서 복구 태스크 유도).
  - **실행 피드백 기반 탐색**: 에이전트가 도구 실행 결과를 보고 다음 행동을 정하는 구조와 맞닿음 (KnowledgeTools, EHR, triage 등).

---

### (2) Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization

- **URL**: https://arxiv.org/abs/2602.22675
- **요약**: Long-horizon 에이전트 검색에서 “더 많이 검색하고, 덜 생각한다”는 **SMTL** 프레임워크. 순차 추론 대신 **병렬 증거 수집**으로 컨텍스트 예산 내 효율적 관리, 이질적 연구 설정에서의 일반화 강화.
- **주요 결과**:
  - BrowseComp 48.6%, GAIA 75.7%, Xbench 82.0%, DeepResearch Bench 45.9%.
  - Mirothinker-v1.0 대비 BrowseComp에서 reasoning step **70.7% 감소**하면서 정확도 향상.
- **Healthcare AI GYM 연관**:
  - **멀티턴·도구 사용**: “검색(도구 호출) vs 추론(think)” 비율 조정 실험에 활용 가능 (이미 `RunConfig.no_think` 등으로 비슷한 ablation 있음).
  - **병렬 증거 수집**: PubMed/Evidence/가이드라인 검색을 병렬로 스케줄링하거나, 턴 수를 줄이는 전략 설계 참고.
  - **효율적 컨텍스트 관리**: 5D/6D 보상과 함께 “불필요한 think 호출 감소” 정책 실험에 활용.

---

## 2. 추가 수집 논문 (2026년 2월~3월)

### (3) ResearchGym: Evaluating Language Model Agents on Real-World AI Research

- **URL**: https://arxiv.org/abs/2602.15112
- **요약**: 가설 생성·실험·믿음 업데이트를 아우르는 **closed-loop 연구 태스크**용 벤치마크·실행 환경. ICML/ICLR/ACL 구두·스포트라이트 논문 5편의 레포를 활용해 39개 서브태스크, 단일 GPU·오염 방지 설계.
- **주요 결과**: GPT-5 기반 에이전트는 15개 평가 중 1곳에서만 11.5% 개선, 서브태스크 완료율 26.5%. 한 번은 human solution을 넘긴 경우도 있으나 **capability–reliability gap**이 큼.
- **Healthcare AI GYM 연관**:
  - Long-horizon 실패 모드(조급함, 가설 과신, 병렬 실험 조율, 컨텍스트 한계) → 우리 GYM의 premature_stop, tool_use_failure, reasoning_error 분석 및 보상 설계에 반영 가능.
  - “실제 연구 워크플로우”와 유사하게 “실제 진료 워크플로우”(cross-domain pathway) 평가 설계 참고.

---

### (4) LongCLI-Bench: Long-Horizon Agentic Programming in CLI

- **URL**: https://arxiv.org/abs/2602.14337
- **요약**: 1,000+ CS 과제·실제 워크플로우에서 선별한 **20개 long-horizon CLI 프로그래밍 태스크** 벤치마크. 요구사항 충족(fail-to-pass)과 회귀 방지(pass-to-pass) 이중 평가, 단계별 스코어링.
- **주요 결과**: SOTA 에이전트 pass rate <20%. 대부분 태스크가 30% 진행 전에 정체. Self-correction은 소폭 개선, **human–agent 협업**(플랜 주입·대화형 가이드)이 크게 유리.
- **Healthcare AI GYM 연관**:
  - **단계별 보상/진행도**: Process reward, SARL, 단계별 완료율 메트릭 확장에 참고.
  - **Human-in-the-loop**: SharedLogbook, GymCoach와 연계한 “플랜 주입·가이드” 실험 설계에 활용.

---

### (5) EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis

- **URL**: https://arxiv.org/abs/2601.05808
- **요약**: **프로그래밍적 합성**으로 도구-상호작용 환경을 대규모 생성. SkelBuilder(환경 골격), ScenGenerator(시나리오·규칙 기반 궤적 검증). 191 환경, ~7,000 시나리오, Qwen3 SFT/RL 학습.
- **주요 결과**: 환경 수 스케일링(0→20)에서 성능 향상; 훈련–테스트 환경 직접 유사성보다 **전이 가능한 문제 해결 패턴** 학습이 중요.
- **Healthcare AI GYM 연관**:
  - **도메인·태스크 스케일링**: 10개 의료 도메인 + cross_domain을 “환경 골격 + 시나리오 생성”으로 더 확장할 때 참고.
  - **규칙 기반 궤적 검증**: 우리의 evaluation_criteria, nl_assertions, safety violation taxonomy와 결합해 자동 검증 파이프라인 강화.

---

### (6) MedAgentGym (ICLR 2026 Oral)

- **URL**: OpenReview / ICLR 2026
- **요약**: **코드 중심 생의학 데이터 사이언스** 추론을 위한 스케일러블 에이전트 훈련 환경. 72,413 태스크 인스턴스, 129 카테고리, 12개 실세계 생의학 시나리오, 실행 가능 샌드박스·인터랙티브 피드백.
- **주요 결과**: Med-Copilot 오프라인 RL +43.02%, 온라인 RL +45.28% (GPT-4o 대비). 상용·오픈소스 간 성능 격차 큼.
- **Healthcare AI GYM 연관**:
  - README/PLANNING의 “vs. MedAgentGym” 비교 유지. 우리는 **텍스트+비전**, **126+ 임상 도구**, **cross-domain pathway**, **5D/6D 보상·안전·FairGRPO·자율 GYM**으로 차별화.
  - 코드 실행 샌드박스 설계는 우리의 tool 시뮬레이션·실행 피드백 루프와 비교하여 정리 가능.

---

### (7) MedAgentBench: A Realistic Virtual EHR Environment

- **URL**: https://arxiv.org/abs/2501.14654
- **요약**: FHIR 호환 가상 EHR로 의료 LLM 에이전트 벤치마크. 300개 임상 유래 태스크, 10 카테고리, 100명 환자 프로필, 70만+ 데이터 요소.
- **주요 결과**: Claude 3.5 Sonnet v2 69.67% 최고 성공률. 카테고리별 성능 편차 큼.
- **Healthcare AI GYM 연관**:
  - EHR 도메인(ehr_management), MIMIC 호환 구조, 14개 EHR 도구와의 정합성·확장 방향 참고.
  - FHIR·EHR API 표준화 시 참고.

---

### (8) AgentClinic (EMNLP 2024 Findings)

- **URL**: https://arxiv.org/abs/2405.07960
- **요약**: 멀티모달 에이전트 벤치마크 (이미지+대화). AgentClinic-NEJM, AgentClinic-MedQA. 인지·암시적 편향 도입 시 진단 정확도·환자 순응도 하락.
- **Healthcare AI GYM 연관**:
  - Patient Agent(12 성향, 13 편향), cognitive_bias 11 matched-pair 테스트와 직접 연관. 편향 유형·메트릭 확장 시 참고.

---

### (9) MedAgentsBench: Thinking Models & Agent Frameworks for Complex Medical Reasoning

- **URL**: arXiv:2503.07459 등
- **요약**: 다단계 임상 추론·진단·치료 계획이 필요한 복잡 의료 추론 벤치마크. 7개 의료 데이터셋, 성능·비용·추론 시간 분석.
- **주요 결과**: DeepSeek R1, OpenAI o3 등 thinking 모델이 복잡 의료 추론에서 우수; 검색 기반 에이전트는 비용 대비 성능이 유리한 경우 있음.
- **Healthcare AI GYM 연관**:
  - 6D 보상(reasoning quality, process), MRPO/CRPO/LLM-Judge 보상과의 비교·실험 설계에 활용.

---

## 6. Phase 2 반영 논문 (2025-2026, 2026-03-10 추가)

> 아래 논문들은 Track C (Eval→Train 피드백 루프) 및 알고리즘 강화에 직접 반영됨.

### 6.1 Training Algorithm

#### WebRL (ICLR 2025, arXiv:2411.02337)
- **제목**: WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
- **핵심**: 실패 trajectory에서 자동으로 curriculum task 생성 + online RL. 실패를 다음 훈련 task의 씨앗으로 활용.
- **GYM 반영 (C-1, C-2)**: `_evaluate()` failure_cases → LLM task generator 전달, eval 성공 trajectory → GRPO replay buffer (30% mix).
- **상태**: ✅ 구현 완료 (`autonomous_agent.py`, `gym_coach.py`)

#### Dr. GRPO (2025)
- **핵심**: GRPO의 per-token length normalization 제거. 짧은 의료 답변이 불필요하게 패널티 받는 문제 해결. Token-sum (not mean) loss 사용.
- **GYM 반영 (C-3)**: `BioAgentGRPOConfig.use_dr_grpo`, `_grpo_policy_update()` 분기.
- **상태**: ✅ 구현 완료 (`grpo_trainer.py`). `lingshu_weakness_fixer`, `qwen25vl_safety` 에이전트에 적용.

#### DAPO (NeurIPS 2025, arXiv:2412.18279)
- **제목**: DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **핵심**: Step-level dense reward, clip-higher, entropy bonus. AIME 2024에서 50점 달성 (SOTA). Token-level policy gradient.
- **GYM 반영**: step-level reward 아이디어 → C-4 `grpo_stepwise_medical_reward()` 설계 참고.
- **상태**: ✅ C-4 구현 완료 (`grpo_rewards.py::grpo_stepwise_medical_reward`)

#### VAPO (2025, arXiv:2504.05118)
- **제목**: VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning
- **핵심**: Value-based Augmented Proximal Policy Optimization. 5000 step에서 SOTA 달성. 학습 불안정성 해소.
- **GYM 반영**: 안정적 medical RL 학습 옵션으로 검토. `BioAgentGRPOConfig`에 향후 `use_vapo` 플래그 추가 가능.
- **상태**: 📋 참조 완료, 향후 ablation 대상

#### FairGRPO (arXiv:2510.19893)
- **핵심**: Demographic-aware reward reweighting. 인구집단 간 공평한 학습 보장.
- **GYM 반영**: ✅ 이미 구현 완료. `qwen25vl_safety` 에이전트에 적용 중.

### 6.2 Medical RL & Evaluation

#### Med-PRM (EMNLP 2025, arXiv:2506.11474)
- **제목**: Med-PRM: Medical Process Reward Model for Improved Reasoning
- **핵심**: 의료 추론 단계별(step-level) guideline 검증 보상. 최종 답변 reward만으로는 중간 추론 개선 어려움.
- **GYM 반영 (C-4)**: `grpo_stepwise_medical_reward()` — clinical step order (symptom→investigation→differential→treatment) 점수화, `step_reward_weight=0.1` 가중치로 기존 composite reward에 합산.
- **상태**: ✅ 구현 완료 (`grpo_rewards.py`, registry: `"stepwise_medical"`)

#### Medical VQA RL (2025, arXiv:2505.13973)
- **핵심**: GRPO가 SFT보다 의료 VQA에서 우수함을 실증. Visual reasoning + factual accuracy 동시 개선.
- **GYM 반영**: 우리 RL 접근법 정당화 근거. 논문 인용 예정.
- **상태**: 📋 참조

#### LingShu (arXiv:2506.07044)
- **제목**: LingShu: A Multilingual Generalist Medical Vision-Language Model
- **핵심**: 12+ 모달리티 지원. GPT-4.1, Claude 4 능가 (의료 VQA). 우리 baseline 모델.
- **GYM 반영**: baseline 모델로 사용 중. 논문 §4 Related Work에 인용.
- **상태**: ✅ 모델 사용 중 (`checkpoints/models/Lingshu-7B`)

### 6.3 Multi-Agent & GYM Framework

#### AgentGym-RL (ICLR 2026 Oral, arXiv:2509.08755)
- **제목**: AgentGym-RL: Scalable Multi-Turn RL for Agentic Tasks
- **핵심**: Multi-turn RL ScalingInter-RL, 다양한 환경에서의 에이전트 훈련.
- **GYM 반영**: GYM 논문 포지셔닝 reference (우리: 의료 도메인 특화 + 187 tools + safety/fairness).
- **상태**: 📋 포지셔닝 참조

#### MMedAgent-RL (ICLR 2026, arXiv:2506.00555)
- **제목**: MMedAgent-RL: Multi-modal Medical Agent with Reinforcement Learning
- **핵심**: Multi-agent 의료 추론. ICLR 2026 accept.
- **GYM 반영**: 차별점: 우리는 autonomous self-evolving (no human curriculum) + FairGRPO.
- **상태**: 📋 포지셔닝 참조

---

## 3. 요약 표 (Healthcare AI GYM 대비)

| 논문 | 초점 | GYM 연관 키워드 |
|------|------|-----------------|
| CLI-Gym | CLI 태스크 생성, 환경 역전 | 태스크 자동 생성, 실행 피드백 |
| SMTL | Long-horizon 검색, 병렬 증거 | think/도구 비율, 컨텍스트 효율 |
| ResearchGym | 실세계 연구 에이전트 평가 | long-horizon 실패 모드, reliability gap |
| LongCLI-Bench | Long-horizon CLI 벤치마크 | 단계별 보상, human–agent 협업 |
| EnvScaler | 도구 환경 프로그램적 합성 | 환경·시나리오 스케일링, 궤적 검증 |
| MedAgentGym | 생의학 코드 에이전트 훈련 | 코드 샌드박스, 72K 태스크 |
| MedAgentBench | 가상 EHR 벤치마크 | FHIR, EHR, 300 태스크 |
| AgentClinic | 멀티모달·편향 | Patient Agent, cognitive bias |
| MedAgentsBench | 복잡 의료 추론 | thinking 모델, 비용·성능 |

---

## 4. 다음 액션 (연구 반영)

- [ ] **CLI-Gym**: `auto_task_generator` 또는 별도 스크립트에 “실패 상태 역전” 아이디어 적용 검토 (의료 시나리오 버전).
- [ ] **SMTL**: `no_think` 및 “병렬 도구 호출” 실험 확대 (PubMed/Evidence 동시 검색 등).
- [ ] **ResearchGym**: 실패 모드(impatience, overconfidence, context limit)를 SharedLogbook/에러 분류에 메트릭으로 추가 검토.
- [ ] **LongCLI-Bench**: 단계별 완료율 메트릭(예: 30%/60%/100%)을 Process/assertion 보상에 반영 검토.
- [ ] **EnvScaler**: 신규 의료 도메인 추가 시 “골격 + 시나리오 생성” 템플릿 도입 검토.
- [ ] **MedAgentBench**: FHIR/EHR API 레이어가 필요해지면 해당 논문 스펙 참고.

---

## 5. 못다한 업무 정리 & 권장 다음 액션 (PLANNING.md §11 기준)

### 5.1 Phase 1.5 미완료

| 항목 | 상태 | 권장 액션 |
|------|------|-----------|
| **전체 Baseline 평가 완료** | 미완료 | `python scripts/run_full_baseline_eval.py --parallel` 실행 후 결과 확인 (3 VL 모델 × 21 벤치마크). 02/26 로그에 “실행 중” 기록됨. |
| **Autonomous GYM 1-Cycle Dry-run** | 미완료 | `configs/autonomous_gym_dryrun.yaml` 사용해 1-agent 1-GPU로 REFLECT→CHOOSE→TRAIN→RECORD 1사이클 검증. Baseline 완료 후 권장. |
| **Task Data** | 목표 초과 | 2,580+ tasks 이미 확보. 추가 확장은 선택. |
| **Clinician-Validated Evaluation** | 부분 반영 | nl_assertions, 22-category safety taxonomy 반영됨. MTBBench 스타일 클리니션 검증 프로토콜은 Phase 2에서 확장 가능. |

### 5.2 Phase 2 이후 (03/01~)

- GPU 확보 후 Autonomous GYM 본격 실행 (3 에이전트, 8 GPU).
- Cross-Domain Pathway RL 실험.
- Reward 전략 ablation (GRPO vs MRPO vs SARL vs DRPO vs Adaptive).
- FairGRPO·ScalingInter-RL·Safety 강화 실험.

### 5.3 연구 로그 반영 액션 (우선순위)

1. **SMTL**: `no_think` 및 병렬 도구 호출 실험 설계 (agent_runner / 보상).
2. **CLI-Gym**: auto_task_generator에 “실패 상태 역전” 의료 버전 검토.
3. **ResearchGym**: SharedLogbook/에러 분류에 long-horizon 실패 메트릭(impatience, overconfidence 등) 추가 검토.

---

*이 로그는 PLANNING.md §5 (Related Work) 및 §11 (Action Items)와 연동됩니다.*
