# 4-Persona NeurIPS 2026 Review: Healthcare AI GYM for Agents

**Paper**: "Healthcare AI GYM for Agents"
**Review Date**: 2026-04-21
**Review Version**: v2

---

## Persona 1: Graham Neubig
*Expert in: NLP systems, multilingual NLP, code generation, evaluation, open-source ML tools*

### 1. Summary
이 논문은 10개 임상 도메인, 195K+ 태스크, 187+ 도구를 포함하는 Gymnasium 호환 의료 AI 에이전트 훈련 환경(Healthcare AI GYM)과, multi-turn agentic RL에서 발생하는 세 가지 실패 모드(response explosion, multi-turn collapse, distillation instability)를 해결하는 TT-OPD 방법론을 제안한다. 환경 구축은 인상적이나, 실험 설계의 엄밀성과 재현성 측면에서 상당한 우려가 있다.

### 2. Score: 4/10

### 3. Strengths
- **S1 (Environment Scale)**: 10개 도메인, 195K+ 태스크, 187+ 도구, 828K passage KB를 갖춘 종합적 의료 에이전트 훈련 환경 구축은 커뮤니티에 실질적인 기여이다. Gymnasium API 호환은 기존 RL 파이프라인과의 통합을 용이하게 한다 (Section 3, Appendix A).
- **S2 (Failure Mode Analysis)**: multi-turn agentic RL에서의 세 가지 실패 모드를 체계적으로 식별하고 ablation을 통해 각 컴포넌트의 역할을 분리한 분석이 교육적으로 가치 있다 (Section 5.1, Figure 4).
- **S3 (5D Reward Design)**: 정확도만이 아닌 process quality, safety, format, coherence를 포함하는 5차원 보상 설계가 실제 임상 워크플로우를 반영하려는 시도로 의미 있다 (Eq. 1, Appendix A.5).
- **S4 (Gradient Signal Dilution Analysis)**: Proposition 2의 SNR 분석과 24:1 dilution ratio 발견은 multi-dimensional reward 설계에 대한 유용한 인사이트를 제공한다 (Section 5.2).

### 4. Weaknesses

- **W1 (CRITICAL: Single-Seed TT-OPD)**: TT-OPD의 모든 결과가 단일 seed에서 나온다 (Section 6, Limitations). NeurIPS 수준의 실험에서 이는 용납하기 어렵다. 61.1% 최종 결과가 noise인지 signal인지 판단할 수 없다. 특히 Figure 3a의 non-monotonic 패턴은 variance가 높을 수 있음을 시사한다.
  - **Fix**: 최소 3-seed 실험 필수. Mean +/- std 보고.

- **W2 (CRITICAL: Unfair MedQA Comparison)**: Table 1에서 TT-OPD (87.1%)와 base model (72.0%)의 비교는 근본적으로 불공정하다. TT-OPD는 828K passage KB + multi-turn tool-use를 활용하지만, base model은 single-turn direct prompting만 사용한다. 저자가 Limitations에서 이를 인정하지만 ("base model + multi-turn AgentRunner baseline without RL training would isolate the RL contribution"), 이 baseline이 없이는 RL의 기여를 전혀 측정할 수 없다. +15.1pp 중 얼마가 RAG이고 얼마가 RL인지 불명확하다.
  - **Fix**: RAG-only baseline (no RL, same tools/KB) 필수.

- **W3 (HIGH: Incomplete Evaluations)**: Table 1에서 다수 벤치마크가 "---" 또는 "$\dagger$ (partial)"로 표시되어 있다. PMC-VQA, MedicationQA, HealthSearchQA, KQA-Silver 등이 미완료 상태이다. 논문 제출 시점에 21개 벤치마크 중 상당수가 미완료인 것은 문제이다.
  - **Fix**: 모든 벤치마크 완료 또는 평가 가능한 것만 claim에 포함.

- **W4 (HIGH: Track 1/Track 2 Inconsistency)**: TT-OPD는 Qwen3.5-9B에서, transfer gap 분석은 Lingshu-7B (Qwen2.5-VL-7B)에서 수행되어 두 트랙의 결과를 직접 비교할 수 없다 (Section 4.1). "distinct objectives" 설명이 있지만, 이는 논문 전체의 narrative coherence를 해친다. TT-OPD가 transfer gap을 해결하는지 여부도 불명확하다.
  - **Fix**: 동일 모델에서 TT-OPD와 baseline 비교, 또는 두 트랙의 관계를 더 명확히 서술.

- **W5 (HIGH: Algorithm Pseudocode vs Text Inconsistency)**: Algorithm 1 (Appendix B)에서 $\lambda_{\text{KL}} = 0.01$인데, Section 3.2 본문에서 $\lambda_{\text{distill}} = 4.0$이라고 한다. Table 3에서도 distillation loss coef = 4.0으로 명시. 이 두 값이 같은 것인지 다른 것인지 혼란스럽다. Eq. 4에서도 $\lambda_{\text{distill}}$를 사용하는데 알고리즘에서는 $\lambda_{\text{KL}}$을 쓴다.
  - **Fix**: 표기 통일. Algorithm 1도 $\lambda_{\text{distill}} = 4.0$으로 수정하거나, 왜 다른 값인지 명확히 설명.

- **W6 (MEDIUM: Tool Usage Pareto Concern)**: 179개 도구 중 19개(10.6%)만 사용되고, 상위 3개가 72.9%를 차지한다 (Appendix G). 이는 187+ tools라는 환경의 contribution을 상당히 약화시킨다. "clinical practice mirrors this"라는 변명은 설득력이 부족하다---training data distribution이 Medical QA에 편중되어 있기 때문일 가능성이 높다.
  - **Fix**: 도메인별 균형 학습 데이터 구성 또는 tool diversity를 장려하는 reward component 추가.

- **W7 (MEDIUM: EHR 100% Accuracy on 50 Samples)**: MIMIC-III와 eICU에서 50개 샘플로 100% 정확도를 보고하는 것은 통계적으로 의미 없다 (Table 1). Full 5K task set 평가 없이 이를 테이블에 포함하는 것은 misleading이다.
  - **Fix**: 50개 파일럿 결과는 텍스트에서만 언급하고, 전체 데이터셋 결과 포함.

### 5. Questions for Authors
1. RAG-only baseline (RL 없이 동일 tool/KB 사용하는 multi-turn agent)의 MedQA 정확도는 얼마인가? +15.1pp 중 RAG 기여분과 RL 기여분을 분리할 수 있는가?
2. Algorithm 1의 $\lambda_{\text{KL}} = 0.01$과 본문의 $\lambda_{\text{distill}} = 4.0$은 같은 hyperparameter인가, 다른 것인가? 400배 차이를 어떻게 설명하는가?
3. 179개 도구 중 160개가 미사용인 상태에서, 환경의 도구 생태계가 실제로 훈련에 기여한다고 어떻게 주장할 수 있는가?

### 6. Missing References
- **AgentBench** (Liu et al., 2023): multi-turn agent 평가 프레임워크
- **WebArena** (Zhou et al., 2024): agentic environment design의 주요 참조
- **SWE-bench** (Jimenez et al., 2024): tool-use agent 평가의 gold standard
- **OpenDevin/SWE-Agent**: reproducible agent training frameworks
- **τ-bench** (Yao et al., 2024): multi-turn agent benchmark

---

## Persona 2: Jinhyuk Lee
*Expert in: Biomedical NLP, medical QA, knowledge-grounded generation, retrieval for biomedical text*

### 1. Summary
의료 AI 에이전트를 위한 대규모 훈련 환경과 self-distillation 기반 training stabilization을 제안한다. 10개 임상 도메인 설계와 안전성 기반 보상은 의료 도메인에 적합한 접근이나, biomedical 평가 방법론과 임상 타당성에 심각한 우려가 있다. 특히 MMLU-Med에서의 25pp 하락은 실제 의료 응용에서 치명적이다.

### 2. Score: 4/10

### 3. Strengths
- **S1 (Safety-Aware Reward)**: 5-level severity taxonomy (Appendix E)와 safety reward는 의료 AI에서 중요한 고려사항을 반영한다. Critical violation (severity 5)이 total reward를 0.1로 cap하는 설계는 적절하다.
- **S2 (Clinical Domain Breadth)**: 진단, 약물 상호작용, triage, 산과, 정신과 등 10개 도메인은 기존 medical agent 환경 대비 가장 포괄적이다 (Table 5).
- **S3 (Knowledge Base Design)**: MedCPT 기반 581K PubMed/PMC passage, BM25 + SQLite FTS5 구현은 실용적이고 재현 가능한 retrieval 시스템이다 (Appendix A.4).
- **S4 (Catastrophic Forgetting Documentation)**: MMLU-Med 84.9% -> 59.6% 하락을 솔직하게 보고하고 transfer gap으로 분석한 것은 연구적으로 정직한 태도이다 (Section 4.2, Section 5.3).

### 4. Weaknesses

- **W1 (CRITICAL: Catastrophic Forgetting Severity)**: TT-OPD 후 MMLU-Med가 84.9% -> 59.6% (-25.3pp), MedMCQA가 65.3% -> 54.2% (-11.1pp)로 대폭 하락한다 (Table 1). 의료 도메인에서 이 수준의 parametric knowledge 망각은 환자 안전에 직결되는 문제이다. 논문이 이를 "inherent limitation of the agentic paradigm"으로 dismissal하지만, 이는 method의 근본적 한계이다. Professional Medicine subtype은 83.5% -> 49.6% (-33.9pp)로 chance level에 근접한다 (Table 10).
  - **Fix**: EWC, replay buffer, 또는 multi-task training으로 catastrophic forgetting 완화 필수. 최소한 forgetting 없는 baseline과의 비교가 필요.

- **W2 (CRITICAL: Clinical Validity of Simulated Tools)**: 도구가 실제 임상 도구를 "시뮬레이션"하지만, 이 시뮬레이션의 임상 타당성 검증이 전혀 없다. `order_lab_test()`가 반환하는 결과가 실제 임상 시나리오의 lab 값 분포를 반영하는지, `calculate_gcs()`의 구현이 임상적으로 정확한지에 대한 validation이 없다. Medical AI 논문에서 이는 중대한 누락이다.
  - **Fix**: 임상 전문가의 tool fidelity 검증 또는 기존 validated clinical decision support tool과의 비교.

- **W3 (HIGH: MedQA Evaluation Protocol Issue)**: MedQA 87.1%는 873/1273 샘플에서 partial evaluation ($\dagger$)이다. 또한 multi-turn AgentRunner를 통한 평가이므로 기존 MedQA 리더보드의 결과와 직접 비교 불가능하다. MedQA는 표준화된 single-turn MCQ benchmark인데, multi-turn tool-use로 평가하면 benchmark의 원래 목적(parametric medical knowledge 측정)이 변질된다.
  - **Fix**: (1) 전체 1273 샘플 평가 완료, (2) 기존 SOTA (GPT-4, Med-PaLM 2 등)와 같은 조건에서의 비교 추가, (3) MedQA를 "retrieval-augmented agentic accuracy"로 명확히 재정의.

- **W4 (HIGH: Task Quality Validation Missing)**: 195K+ 태스크 중 상당수가 `AutoTaskGenerator` + LLM으로 생성되었다 (Appendix A.2). "Human validation"이 언급되지만 (1) 몇 명의 annotator가, (2) 어떤 기준으로, (3) 전체의 몇 %를 검증했는지 불명확하다. LLM-generated medical tasks의 품질은 환자 안전과 직결된다.
  - **Fix**: Inter-annotator agreement, validation coverage 비율, error rate 보고.

- **W5 (HIGH: BERTScore for Medical Accuracy)**: Accuracy reward에 BERTScore (BiomedBERT)를 사용하지만 (Section 3, Appendix A.5), BERTScore는 semantic similarity를 측정하지 medical correctness를 보장하지 않는다. "The patient has type 1 diabetes"와 "The patient has type 2 diabetes"는 BERTScore가 높지만 임상적으로 완전히 다른 진단이다.
  - **Fix**: Medical entity-level exact match 또는 UMLS concept matching 추가.

- **W6 (MEDIUM: Long-Form QA Metrics)**: LiveQA에서 ROUGE-L 0.083, hallucination 55.2%는 매우 낮은 성능이다 (Section 4.2). "Format mismatch" (short answers vs paragraph references)를 이유로 들지만, 이는 model이 task format을 이해하지 못한다는 의미이다. 의료 QA에서 55% hallucination rate는 심각하다.
  - **Fix**: Format-aware evaluation 또는 answer completeness를 장려하는 reward component.

- **W7 (MEDIUM: Obstetrics Zero Action Score)**: Table 6에서 산과 도메인의 action score가 0.000인 것은 해당 도메인에서 agent가 전혀 작동하지 않음을 의미한다. 10개 도메인 중 하나가 완전히 실패한 것을 contribution으로 주장하는 것은 부적절하다.
  - **Fix**: 실패 원인 분석 및 해결, 또는 해당 도메인을 current limitation으로 명시.

### 5. Questions for Authors
1. Professional Medicine MMLU에서 49.6% (chance level ~25% for 4-option MCQ이므로 chance 이상이긴 하나)는 USMLE 스타일 임상 의사결정 능력의 대폭 하락을 의미한다. 이 모델을 실제로 "의료 에이전트"로 배포하는 것이 윤리적으로 가능한가?
2. 195K+ 태스크의 human validation coverage는 정확히 몇 %이며, annotator의 medical expertise level은?
3. Safety violation detection이 rule-based (50+ patterns)인데, adversarial한 안전 위반이나 novel한 위반 패턴에 대한 coverage를 어떻게 보장하는가?
4. 산과 도메인에서 action score 0.0의 원인은 무엇인가?

### 6. Missing References
- **EHRSQL** (Lee et al., 2024): EHR 기반 structured query 평가
- **CLUE** (Wang et al., 2025): clinical language understanding evaluation
- **BioMedRAG** (Han et al., 2025): biomedical RAG baseline
- **MedPrompt** (Nori et al., 2024): prompting-based medical QA SOTA
- **ClinicalAgent Bench** (Huang et al., 2024): clinical agent evaluation framework
- **Almanac** (Zakka et al., 2024): retrieval-augmented clinical decision support

---

## Persona 3: Sewon Min
*Expert in: Question answering, in-context learning, retrieval-augmented generation, knowledge-intensive NLP*

### 1. Summary
Multi-turn agentic RL에서의 instability를 해결하기 위해 EMA teacher + outcome-conditioned hints + cosine length reward를 결합한 TT-OPD를 제안한다. Ablation study를 통한 failure mode 분석은 체계적이나, 실험 설계에서 중요한 confound가 해결되지 않았고, QA 평가 방법론에 심각한 문제가 있다.

### 2. Score: 5/10

### 3. Strengths
- **S1 (Systematic Ablation)**: Figure 4의 4-variant ablation은 각 컴포넌트(periodic reset -> EMA -> outcome hints -> cosine reward)의 효과를 명확히 분리한다. 특히 KL collapse, response explosion, multi-turn collapse의 progression을 시각적으로 보여주는 것이 효과적이다 (Section 5.1).
- **S2 (Gradient Signal Dilution Insight)**: Proposition 2의 SNR 분석은 multi-dimensional reward에서의 gradient dynamics를 이해하는 데 유용하며, 24:1 dilution ratio는 직관적으로 설득력 있다 (Section 5.2, Figure 5).
- **S3 (Transparent Negative Results)**: GRPO baseline이 "no improvement"을 보인다는 것, agentic RL이 text QA에 transfer되지 않는다는 것을 솔직하게 보고한다 (Section 5.3).
- **S4 (Non-Monotonic Convergence Analysis)**: Proposition 1의 EMA를 implicit learning rate annealing으로 해석하는 관점은 흥미롭다. Spring analogy로 설명하는 것도 접근성이 좋다 (Section 5.2).

### 4. Weaknesses

- **W1 (CRITICAL: Evaluation Protocol Confound)**: Section 4.2에서 "single-turn generation produces zero accuracy on all benchmarks"라고 하는데, 이는 모델이 multi-turn tool-use 없이는 전혀 작동하지 않는다는 의미이다. 이것은 심각한 문제이다: (1) 모델이 `submit_answer()` 호출 없이는 답을 출력할 수 없도록 과적합되었다는 것을 시사하며, (2) RAG 없이는 performance가 0%라면, 87.1% MedQA의 대부분은 RAG에 의한 것이지 RL에 의한 것이 아닐 수 있다. 이 confound를 해결하지 않으면 TT-OPD의 실제 기여를 측정할 수 없다.
  - **Fix**: (1) No-RL + same tools/KB baseline, (2) Random policy + tools baseline, (3) SFT-only + tools baseline. 이 세 가지 baseline으로 RL contribution을 분리.

- **W2 (CRITICAL: Proposition Rigor)**: Section 5.2의 세 propositions은 "proof sketches with stated assumptions"라고 하지만, 실제로는 well-known results의 재서술에 가까우다. Proposition 1의 Fisher information approximation은 standard natural gradient theory (Amari, 1998)의 직접 적용이고, Proposition 3의 KL boundedness는 EMA의 trivial property이다. "Proposition"이라는 라벨은 이들의 novelty를 과대표현한다. 또한 proof sketches (Appendix C)가 너무 간략하여 검증이 어렵다.
  - **Fix**: "Observations" 또는 "Remarks"로 다운그레이드하거나, full proofs 제공.

- **W3 (HIGH: No Comparison with Existing OPD Methods)**: OPSD, Self-Distilled RLVR, SRPO, CRISP, HiLL 등 다수의 OPD 방법이 Related Work에서 논의되지만 (Section 2), 실험에서 이들과의 직접 비교가 전혀 없다. "These methods focus on single-turn settings"라는 변명이 있지만, (1) 이 방법들을 multi-turn에 적용한 결과를 보여줘야 TT-OPD의 agentic-specific 기여가 증명되고, (2) 적어도 OPSD나 HiLL은 multi-turn adaptation이 가능할 것이다.
  - **Fix**: OPSD, Self-Distilled RLVR의 multi-turn adaptation baseline 추가.

- **W4 (HIGH: Outcome-Conditioned Hints Simplicity)**: Section 3.2에서 outcome hints가 template-based라고 명시한다 ("We use template-based hints for reproducibility"). "Reasoning appears sound, continue this clinical approach" (correct) vs "revisit the differential diagnosis" (incorrect)와 같은 단순 template가 정말로 의미 있는 KL regularization을 제공하는지 의심스럽다. Teacher의 logprob distribution이 이런 generic template에 얼마나 민감한지에 대한 ablation이 없다.
  - **Fix**: (1) Hint template 다양성에 대한 sensitivity analysis, (2) Random hints vs correct hints ablation, (3) 실제 template 전체 공개.

- **W5 (HIGH: Validation Set Details Missing)**: 61.1%라는 key metric이 어떤 validation set에서 측정되었는지 명확하지 않다. Section 4.1에서 "validation accuracy on Healthcare AI GYM tasks"라고만 언급하며, validation set의 크기, 도메인 분포, train set과의 overlap 여부가 불명확하다.
  - **Fix**: Validation set 구성 상세 기술 (크기, 도메인별 분포, 생성 방식).

- **W6 (MEDIUM: GRPO "No Improvement" Claim Overstated)**: Section 4.3에서 "GRPO baseline shows no improvement (±0.2 pp)"라고 하지만, Figure 3a에서 GRPO baseline은 단 하나의 configuration에서의 결과이다. GRPO에 cosine reward를 추가하면 어떤가? GRPO + length control (TT-OPD 없이)이 이미 상당 부분의 개선을 달성할 수 있는가?
  - **Fix**: GRPO + cosine reward (no distillation) ablation 추가.

- **W7 (MEDIUM: 60-Step Training Scale)**: 60 step의 training은 매우 짧다. 각 step에서 batch size 8, 3 generations per prompt이므로 총 60 * 8 * 3 = 1,440 rollouts밖에 되지 않는다. 이 scale에서 61.1% 결과의 generalizability를 어떻게 보장하는가?
  - **Fix**: 더 긴 training (200+ steps)에서의 behavior 보고, 또는 현재 scale이 충분하다는 convergence 증거.

### 5. Questions for Authors
1. TT-OPD trained model에서 tools/KB 없이 직접 답변하도록 하면 (e.g., "Just answer the question directly without using any tools") 정확도는 얼마인가? 0%라면, 이는 over-specialization 문제가 아닌가?
2. Outcome-conditioned hints에서 "reinforcing" vs "corrective" hint의 차이가 teacher logprob에 미치는 영향을 정량화할 수 있는가? (e.g., hint 유무에 따른 teacher perplexity 차이)
3. Validation set의 정확한 크기와 구성은? Train set과 동일한 task distribution에서 sampling한 것인가, held-out domain인가?
4. GRPO + cosine reward (distillation 없이)의 validation accuracy는?

### 6. Missing References
- **REST-EM** (Singh et al., 2024): reward-guided self-training
- **ReST** (Gulcehre et al., 2023): reinforced self-training for LLMs
- **V-STaR** (Hosseini et al., 2024): training verifiers for self-taught reasoning
- **STILL-1/2** (Ma et al., 2024): on-policy distillation for reasoning
- **Reward Soup** (Rame et al., 2024): multi-objective reward composition

---

## Persona 4: Akari Asai
*Expert in: Retrieval-augmented generation, self-reflective LLMs, multi-hop reasoning, knowledge-intensive tasks*

### 1. Summary
Healthcare AI GYM 환경에서 retrieval-augmented agentic reasoning을 학습하는 프레임워크를 제안하며, EMA teacher의 outcome-conditioned hints를 통해 multi-turn tool-use의 안정성을 유지하는 TT-OPD를 소개한다. Retrieval과 generation의 interaction을 agent training에 통합하는 시도는 흥미로우나, retrieval quality 분석과 self-reflection 메커니즘의 부재가 아쉽다.

### 2. Score: 5/10

### 3. Strengths
- **S1 (Retrieval-Augmented Agent Training)**: RAG를 training-time에 통합하여 retrieval-then-reason 패턴을 RL로 학습하는 접근은 Self-RAG과 complementary한 방향이다. MedQA에서 72% -> 87.1%의 개선은 retrieval augmentation의 가치를 보여준다 (Table 1).
- **S2 (Multi-Turn Tool-Use Stability)**: 7.0-7.4 turns를 유지하면서 accuracy를 개선하는 것은 multi-turn agent에서의 핵심 과제이며, Figure 3d에서 이를 설득력 있게 보여준다.
- **S3 (Outcome-Conditioned Teacher)**: Teacher에게 privileged information을 제공하여 outcome-aware logprob을 생성하는 아이디어는 Self-RAG의 self-reflection token과 유사하면서도 다른 접근이다. Privileged tokens이 student에게는 보이지 않는다는 설계가 깔끔하다 (Section 3.2).
- **S4 (Multi-Hop Reasoning Potential)**: Cross-domain pathways (chest pain -> labs -> imaging -> diagnosis)는 multi-hop clinical reasoning의 자연스러운 formulation이다 (Appendix A.2).

### 4. Weaknesses

- **W1 (CRITICAL: No Retrieval Quality Analysis)**: 논문의 핵심이 retrieval-augmented agentic reasoning임에도, retrieval quality에 대한 분석이 전혀 없다. (1) `search_evidence()`가 반환하는 passage의 relevance는? (2) Retrieved passage가 최종 답변에 얼마나 기여하는가? (3) Retrieval failure가 task failure의 원인인 비율은? BM25 + SQLite FTS5는 modern dense retrieval 대비 상당히 기본적인 시스템인데, retrieval bottleneck이 없는지 검증이 필요하다.
  - **Fix**: (1) Retrieval recall/precision 측정, (2) Retrieved passage vs final answer의 attribution analysis, (3) Oracle retrieval (gold passage 제공) 실험으로 retrieval ceiling 확인.

- **W2 (CRITICAL: No Adaptive Retrieval)**: Agent가 항상 search_evidence를 호출하는 것이 최적인지, 또는 parametric knowledge만으로 충분한 경우 retrieval을 skip하는 것이 더 나은지에 대한 분석이 없다. Self-RAG의 핵심 insight는 "when to retrieve"의 학습인데, 이 논문에서는 tool usage distribution (Appendix G)에서 search_evidence가 34.2%로 가장 많이 호출된다고만 보고한다. MMLU-Med에서의 성능 하락(-25.3pp)이 바로 "불필요한 retrieval overhead" 때문일 수 있다.
  - **Fix**: (1) Retrieval 호출 빈도와 task 성공률의 상관분석, (2) Retrieval이 답변을 개선한 경우 vs 악화시킨 경우 분류, (3) Adaptive retrieval 학습.

- **W3 (HIGH: No Self-Reflection Mechanism)**: TT-OPD의 outcome hints는 training-time에만 존재하며, inference-time에는 self-reflection 메커니즘이 없다. Agent가 잘못된 retrieval을 받았을 때 이를 인식하고 재검색하는 능력이 있는가? Figure 8의 trajectory example에서 2-turn만에 submit_answer를 호출하는 패턴은 self-reflection 없이 바로 답변하는 것처럼 보인다.
  - **Fix**: (1) Inference-time self-reflection 메커니즘 추가 (Self-RAG style reflection tokens), (2) Agent가 retrieval 결과를 평가하고 재검색하는 비율 분석.

- **W4 (HIGH: Knowledge Base Coverage Gaps)**: 828K passage KB가 모든 10개 도메인을 충분히 cover하는지 불명확하다. Section 6에서 "The knowledge base does not cover all subspecialties equally"라고 인정하지만, 어떤 도메인이 부족하고 이것이 성능에 어떤 영향을 미치는지 분석이 없다. 산과(obstetrics)에서 action score 0.0인 것이 KB coverage 부족 때문일 수 있다.
  - **Fix**: 도메인별 KB passage 수와 retrieval 성공률 보고.

- **W5 (HIGH: Trajectory Example Concern)**: Figure 8의 trajectory example에서 student가 "inferior STEMI"를 진단하고 바로 submit하지만, teacher의 corrective hint로 인해 "right ventricular involvement"을 고려하라는 신호를 받는다. 그런데 이 correction은 training-time에만 작동한다. Inference-time에 같은 실수를 하면 어떻게 되는가? TT-OPD가 이런 specific error pattern을 학습하여 prevention할 수 있다는 증거가 필요하다.
  - **Fix**: Training 중 corrective hint를 받은 error pattern이 이후 step에서 감소하는 longitudinal analysis.

- **W6 (MEDIUM: Multi-Hop Reasoning Evaluation Missing)**: Cross-domain pathways (6개)에 대한 평가 결과가 논문에 없다. Multi-hop clinical reasoning이 환경의 핵심 기능 중 하나로 소개되지만, 이에 대한 실험 결과가 없다.
  - **Fix**: Cross-domain pathway 정확도 및 stage-wise error 분석 추가.

- **W7 (MEDIUM: BM25 vs Dense Retrieval)**: 828K passage에 BM25만 사용하는 것은 MedCPT라는 dense retriever를 이미 가지고 있으면서도 활용하지 않는 것처럼 보인다. MedCPT가 passage sourcing에만 사용되고 retrieval에는 BM25가 사용되는 이유가 불명확하다.
  - **Fix**: BM25 vs MedCPT dense retrieval 비교 실험.

### 5. Questions for Authors
1. Agent가 retrieval을 skip하고 직접 답변하는 trajectory의 비율은? 그 경우의 accuracy는?
2. MedQA 87.1% 중, retrieved passage에 정답 정보가 포함된 비율은? (retrieval recall)
3. Inference-time에 agent가 자신의 답변을 re-evaluate하거나 re-retrieve하는 패턴이 관찰되는가?
4. MedCPT dense retrieval을 BM25 대신 사용하면 성능이 어떻게 변하는가?
5. Cross-domain pathway 태스크에서의 TT-OPD 성능은?

### 6. Missing References
- **CRAG** (Yan et al., 2024): Corrective RAG with adaptive retrieval
- **Active RAG** (Jiang et al., 2023): active retrieval augmented generation
- **FLARE** (Jiang et al., 2023): forward-looking active retrieval augmented generation
- **MedRAG** (Xiong et al., 2024): medical RAG framework
- **Self-RAG** (Asai et al., 2023) -- cited but not compared experimentally
- **Adaptive-RAG** (Jeong et al., 2024): complexity-adaptive retrieval

---

## Cross-Reviewer Consensus

### Issues Raised by 3+ Reviewers (CRITICAL)

1. **Unfair/Confounded MedQA Comparison** (Neubig W2, Min W1, Asai W1): TT-OPD의 87.1% MedQA 결과가 RL vs RAG 기여분을 분리하지 못함. RAG-only baseline 필수. 4명 모두 지적.

2. **Single-Seed TT-OPD Results** (Neubig W1, Min W7): 핵심 결과가 단일 seed에 의존. Statistical reliability 검증 불가.

3. **Catastrophic Forgetting / Over-Specialization** (Lee W1, Min W1, Asai W2): MMLU-Med -25.3pp 하락은 모델이 tool-use에 과도하게 특화됨을 시사. Single-turn에서 0% accuracy는 심각한 over-specialization.

4. **Incomplete Evaluations** (Neubig W3, Lee W3): 다수 벤치마크 미완료 ($\dagger$, ---). 논문 제출 수준의 완성도 미달.

### Issues Raised by 2 Reviewers (HIGH)

5. **No Comparison with Existing OPD Methods** (Min W3, implied by Neubig W4): OPSD, Self-Distilled RLVR 등과의 직접 비교 없음.

6. **Outcome Hint Effectiveness Unverified** (Min W4, Asai W5): Template-based hints의 실제 효과에 대한 ablation 부족.

7. **Track 1/Track 2 Model Inconsistency** (Neubig W4, implicit in Min's concerns): 두 트랙의 모델이 달라 결과 간 연결이 약함.

8. **Retrieval Quality Unanalyzed** (Asai W1, Asai W2): Retrieval-augmented 접근의 핵심인 retrieval 품질 분석 부재.

9. **Validation Set Details Missing** (Min W5, Lee W4): 핵심 metric인 validation accuracy의 dataset 구성이 불명확.

### Unique but Important Issues (MEDIUM)

10. **Algorithm Pseudocode vs Text Inconsistency** (Neubig W5): $\lambda_{\text{KL}}$ = 0.01 vs $\lambda_{\text{distill}}$ = 4.0 혼란.

11. **Tool Usage Pareto / 187 tools claim weakened** (Neubig W6): 10.6% tool usage rate가 환경 기여를 약화시킴.

12. **Propositions Overclaim Novelty** (Min W2): Well-known results의 재서술을 "Propositions"으로 과대표현.

13. **BERTScore for Medical Accuracy** (Lee W5): Semantic similarity가 medical correctness를 보장하지 않음.

14. **No Self-Reflection at Inference** (Asai W3): Training-time 교정만 존재, inference-time 자기 교정 없음.

15. **Obstetrics Domain Failure** (Lee W7): Action score 0.0인 도메인의 원인 미분석.

---

## Ranked Action Items

### 1. RAG-only baseline 추가 (MedQA RL contribution 분리)
- **Severity**: CRITICAL
- **Fixable without new experiments?**: NO (새 실험 필요)
- **Fix**: Base model + same multi-turn AgentRunner + tools/KB, but NO RL training. 이 baseline의 MedQA 정확도로 RL vs RAG 기여분 분리.

### 2. Multi-seed TT-OPD 실험 (최소 3 seeds)
- **Severity**: CRITICAL
- **Fixable without new experiments?**: NO
- **Fix**: 3 seeds로 TT-OPD 실험 반복. Mean ± std 보고. 61.1%의 reproducibility 검증.

### 3. Incomplete benchmark 결과 완성
- **Severity**: CRITICAL
- **Fixable without new experiments?**: NO (추가 평가 필요)
- **Fix**: Table 1의 모든 $\dagger$와 --- 항목 완료. 불가능한 벤치마크는 제거하고 claim 범위 축소.

### 4. Catastrophic forgetting 완화 또는 제대로 된 분석
- **Severity**: CRITICAL
- **Fixable without new experiments?**: NO
- **Fix**: (1) EWC/replay buffer 적용, 또는 (2) Forgetting이 불가피하다면, tool access가 없는 상황에서의 안전성 분석 추가. 최소한 "without tools, this model should not be deployed"라는 명확한 경고.

### 5. Algorithm pseudocode-text 불일치 수정 ($\lambda$ 값)
- **Severity**: HIGH
- **Fixable without new experiments?**: YES
- **Fix**: Algorithm 1의 $\lambda_{\text{KL}} = 0.01$을 $\lambda_{\text{distill}} = 4.0$으로 수정하거나, GRPO KL penalty $\beta = 0.01$과의 차이를 명확히 서술.

### 6. 기존 OPD 방법 (OPSD, Self-Distilled RLVR) baseline 추가
- **Severity**: HIGH
- **Fixable without new experiments?**: NO
- **Fix**: OPSD를 multi-turn setting에 적용한 baseline 실험. TT-OPD의 agentic-specific 기여 증명.

### 7. GRPO + cosine reward (no distillation) ablation 추가
- **Severity**: HIGH
- **Fixable without new experiments?**: NO
- **Fix**: Cosine length reward만 추가한 GRPO의 validation accuracy 측정. Distillation 없이도 length control만으로 상당한 개선이 있는지 확인.

### 8. Validation set 구성 상세 기술
- **Severity**: HIGH
- **Fixable without new experiments?**: YES
- **Fix**: Validation set 크기, 도메인별 분포, train set과의 overlap 여부, sampling 방법 등을 Appendix에 추가.

### 9. Outcome hint ablation (random hint vs no hint vs correct hint)
- **Severity**: HIGH
- **Fixable without new experiments?**: NO (부분적으로 가능)
- **Fix**: (1) Random hint, (2) Reversed hint (correct에 corrective, incorrect에 reinforcing), (3) No hint (EMA only). 최소 (3)은 이미 ablation에 있으므로, (1)과 (2) 추가.

### 10. Retrieval quality 분석 추가
- **Severity**: HIGH
- **Fixable without new experiments?**: PARTIALLY (기존 trajectory 분석으로 가능)
- **Fix**: Training trajectory에서 retrieved passage의 relevance 측정, retrieval 성공/실패와 task 성공/실패의 상관분석.

### 11. Proposition 표현 수정 ("Proposition" -> "Remark" or "Observation")
- **Severity**: MEDIUM
- **Fixable without new experiments?**: YES
- **Fix**: Well-known results의 application인 경우 "Observation"으로 변경. 진정한 novel result인 경우에만 "Proposition" 유지.

### 12. Track 1/Track 2 narrative coherence 강화
- **Severity**: MEDIUM
- **Fixable without new experiments?**: YES (writing)
- **Fix**: 두 트랙의 관계를 Introduction에서 더 명확히 서술. "TT-OPD는 transfer gap을 해결하는가?"라는 질문에 답변.

### 13. Tool usage 편중 문제 논의 강화
- **Severity**: MEDIUM
- **Fixable without new experiments?**: YES (writing)
- **Fix**: 현재의 Pareto concentration을 limitation으로 더 솔직하게 인정. "187+ tools" claim을 "187+ tools available, 19 actively used"로 수정.

### 14. BERTScore 한계 인정 및 보완 metric 추가
- **Severity**: MEDIUM
- **Fixable without new experiments?**: YES (기존 결과에 UMLS matching 추가 가능)
- **Fix**: UMLS concept matching 또는 medical entity exact match를 보조 metric으로 추가.

### 15. Cross-domain pathway 평가 결과 추가
- **Severity**: MEDIUM
- **Fixable without new experiments?**: NO
- **Fix**: 6개 cross-domain pathway에서의 TT-OPD 성능 평가 및 stage-wise error analysis.

---

## Overall Assessment

**Average Score: 4.5/10** (Neubig: 4, Lee: 4, Min: 5, Asai: 5)

**Verdict**: **Reject** in current form. Resubmit after addressing Critical items 1-4.

이 논문은 두 가지 substantial한 기여(Healthcare AI GYM 환경, TT-OPD 방법론)를 시도하지만, 실험적 엄밀성에서 NeurIPS 기준을 충족하지 못한다. 가장 심각한 문제는 (1) RAG vs RL contribution 미분리, (2) single-seed 결과, (3) 다수 벤치마크 미완료, (4) catastrophic forgetting의 심각성이다.

긍정적으로, failure mode 분석의 체계성과 환경 구축의 규모는 높이 평가할 수 있으며, 위의 Critical/High 이슈들이 해결되면 strong contribution이 될 잠재력이 있다. 특히 환경 구축만으로도 별도 systems/datasets paper로의 가치가 있을 수 있다.

**추천**: (1) GYM environment와 TT-OPD를 별도 논문으로 분리하는 것을 고려하거나, (2) Critical items 1-4를 모두 해결한 후 resubmit.
