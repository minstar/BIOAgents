# BIOAgents NeurIPS 2026 논문 리뷰: 4인 전문가 페르소나 리뷰

---

## 리뷰어 1: Graham Neubig (CMU) -- NLP 시스템, 코드 생성, LLM 에이전트 전문가

### Overall Assessment
- **점수: 5/10**
- **추천: Weak Reject**

Neubig 교수는 SWE-bench, CodeAgent, OpenHands 등 에이전트 시스템의 설계와 평가 엄밀성에 깊은 관심을 갖고 있으며, 특히 재현 가능성과 공정한 비교를 중시한다. 이 논문은 흥미로운 환경(Healthcare AI GYM)과 방법론(TT-OPD)을 제안하지만, 실험 설계에 근본적인 결함이 있다.

### Detailed Comments

**1. 불명확한 문장/표현**

- "single-turn generation produces zero accuracy on all benchmarks" (Section 4.2): 이 문장은 극단적이며 의심스럽다. 모델이 정말 모든 벤치마크에서 0%를 기록한다면, 이는 TT-OPD 학습이 모델의 일반적 능력을 완전히 파괴했다는 뜻이다. 이것을 "artifact가 아니라 feature"라고 주장하는 것은 설득력이 없다. 실제로는 모델이 `submit_answer()` 호출 없이는 답변을 생성할 수 없게 과적합(overfit)된 것으로 보인다.

- "the model learns to maximize token-level coverage as a proxy for task completion" (Section 1): 이 인과관계 주장에 대한 증거가 부족하다. response explosion이 coverage maximization 때문인지, 단순히 KL penalty 부재 때문인지 구분되지 않는다.

- "passively absorbing the student's learned weights" (Section 3.2): EMA를 "passively absorbing"으로 표현하는 것은 부정확하다. EMA는 명시적인 수학적 연산이지 "수동적 흡수"가 아니다.

**2. 불명확한 실험**

- **Track 1 vs Track 2 분리**: TT-OPD (Qwen3.5-9B, full-param, from scratch)와 transfer gap 분석 (Lingshu-7B, LoRA, from SFT)이 완전히 다른 모델, 다른 학습 방식, 다른 시작점을 사용한다. 이 두 track의 결과를 하나의 논문에서 통합된 스토리로 제시하는 것은 **심각한 실험 설계 문제**이다.

- **MedQA 87.1% 결과**: 873/1273 샘플만 평가된 partial 결과($\dagger$)이다. 완전한 결과 없이 +15.1pp 개선을 주장하는 것은 불완전하다.

- **EHR 100% accuracy**: 50개 샘플에서 100%는 통계적으로 무의미하다.

**3. 누락된 실험**

- **동일 모델에서의 공정한 비교**: Qwen3.5-9B에서 (a) base GRPO, (b) EMA-only, (c) TT-OPD, (d) SFT+GRPO를 모두 비교해야 한다.
- **다른 모델에서의 검증**: Llama 3, Mistral 등 다른 base model에서의 결과가 없다.
- **Multi-seed 실험**: TT-OPD는 single seed만 보고한다.
- **Ablation에서 cosine reward만 사용 (outcome hint 없이)**: Full ablation matrix가 불완전하다.

**4. 실험 방향**

- 실험 스토리가 두 개의 분리된 이야기(TT-OPD 학습 역학 + agentic-textual transfer gap)로 나뉘어 있으며, 이 둘이 유기적으로 연결되지 않는다.

**5. 논문 방향/내러티브**

- 논문의 두 가지 기여(환경 + 방법론)가 모두 중요하지만, 각각이 절반만 완성된 느낌이다.

**6. 기술적 건전성**

- Proposition 1-3은 "proof sketches"이며, 엄밀한 증명이 아니다.
- Eq. 3의 TT-OPD loss에서 $s_t^+$의 정의가 모호하다.

**7. 새로움 평가**

- TT-OPD의 핵심 새로움은 "bidirectional outcome-privileged conditioning"이다. 하지만 이것은 OPSD의 확장 + EMA + cosine reward 결합이며, 각 구성요소가 기존 방법이다.

**8. 관련 연구 누락**

- OpenHands/SWE-Agent, AgentTrek, AgentQ, WebArena, OSWorld, veRL

**9. 프레젠테이션 품질**

- Figure 3과 Figure 4에 거의 동일한 데이터가 중복 표시된다.

**10. 재현 가능성**

- Anonymous repository URL 제공 (좋음)
- 하이퍼파라미터 표 제공 (좋음)
- 학습에 사용된 실제 task subset이 불명확

### Specific Line-by-Line Comments

- **L57 (abstract)**: "195K+ tasks" -- 실제 학습에 사용된 task 수는?
- **L84**: "standard GRPO produces no improvement" -- GRPO의 문제인지 환경 설계의 문제인지 구분 필요
- **L387**: "single-turn generation produces zero accuracy" -- 치명적 결함임을 인정해야 함
- **L409**: MIMIC-III/eICU 100% on 50 samples -- 통계적 의미 없음
- **L466 (MMLU 59.6%)**: 25.3pp 하락은 catastrophic forgetting

---

## 리뷰어 2: Jinhyuk Lee (Google DeepMind) -- 생의학 NLP, PubMedBERT, BioBERT 전문가

### Overall Assessment
- **점수: 4/10**
- **추천: Reject**

Lee 박사는 의료 AI 벤치마크의 엄밀성과 임상적 타당성에 매우 높은 기준을 갖고 있다.

### Detailed Comments

**1. 불명확한 문장/표현**

- "ecological validity" (Section 1): MedAgentGym이 ecological validity가 낮다고 비판하면서, 본 논문의 Healthcare AI GYM이 높다는 근거를 제시하지 않는다.
- "knowledge base of 828K medical passages": quality와 coverage에 대한 검증이 없다.

**2. 불명확한 실험**

- **MedQA 87.1% via multi-turn**: base model에게도 동일한 tool과 KB 접근 권한을 주고 multi-turn으로 평가해야 한다. RAG + base model의 성능이 baseline으로 필요.
- **MMLU 84.9% -> 59.6% 하락**: catastrophic forgetting. 의료 AI에서 기존 지식의 25% 손실은 안전성 관점에서 심각.
- **5D Reward의 임상적 타당성**: weight 설정 근거 불명확. Safety가 0.2 weight인 것이 정당한가?

**3. 누락된 실험**

- **임상의 평가 (Human evaluation)**: 의료 AI 논문에서 자동화 메트릭만으로는 불충분
- **Safety evaluation 세부 결과**: violation rate 비교 필수
- **Hallucination rate 상세 분석**: 55.2%는 매우 높음
- **외부 의료 LLM과의 비교**: Med-PaLM 2, MedGemma 등

**4. 실험 방향**

- RL 학습 역학에 과도하게 집중, 임상적 영향에 더 중점을 둬야 함

**5. 논문 방향/내러티브**

- Table 4에서 Obstetrics action score = 0.000, Psychiatry = 0.421로, 다수 도메인에서 실패. 이를 솔직히 논의하지 않는다.

**6. 기술적 건전성**

- KB가 BM25 + SQLite FTS5 사용. 2026년 기준으로 dense retrieval을 사용하지 않는 것은 뒤처져 있다.

**8. 관련 연구 누락**

- AMIE (Tu et al., 2024), MedGemma 1.5, BioMedGPT, PMC-LLaMA, MedPrompt, ClinicalBench

**9. 프레젠테이션 품질**

- Table 1에 "---" 항목이 너무 많다. "$\dagger$ partial" 결과가 3개.

**10. 재현 가능성**

- KB에 test set 답변이 포함되어 있지 않다는 보장이 없다.

### Specific Line-by-Line Comments

- **L228**: "chosen to balance" → "manually set"으로 변경 권장
- **L398**: partial result를 main table에 넣는 것은 부적절
- **L430**: 55.2% hallucination은 위험 수준
- **Appendix Table 4**: Obstetrics action score 0.000 설명 없음

---

## 리뷰어 3: Sewon Min (UC Berkeley/Meta) -- Open-domain QA, 실험 설계, 통계적 엄밀성 전문가

### Overall Assessment
- **점수: 5/10**
- **추천: Borderline Reject**

### Detailed Comments

**1. 불명확한 문장/표현**

- "standard GRPO produces no improvement": GRPO의 한계인지 환경/reward 설계의 문제인지 구분 안됨
- "renders DAPO's clipping and GSPO's importance sampling vacuous": 강한 표현. 이들은 다른 setting을 위해 설계된 것
- "oscillating recovery": 불안정한 학습을 positive framing하는 것은 문제

**2. 불명확한 실험**

- **Validation set 구성**: val accuracy 61.1%가 어떤 set에서 측정되었는지 불명확
- **G=3**: advantage estimation에 매우 불안정. Track 2는 G=8인데 Track 1은 왜 3인가?
- **Step 단위 보고**: 각 step이 정확히 무엇인지 불명확

**3. 누락된 실험**

- **Confidence intervals / error bars**: Multi-seed 실험 필수
- **Base model + RAG baseline**: retrieval 효과 분리
- **Training data 규모 효과**: 195K 중 실제 사용된 subset 크기
- **더 긴 학습**: 60 step 이후 결과

**4. 실험 방향**

- Ablation 구조는 좋으나, TT-OPD vs GRPO가 cross-model 비교 (apple-to-orange)

**5. 논문 방향/내러티브**

- 3개의 independent contribution이 모두 깊이 부족. 하나의 깊은 기여가 낫다.

**6. 기술적 건전성**

- Eq. 4 cosine reward (binary) vs 5D reward (continuous) 결합 방식 불명확
- "dynamic sampling: keep only prompts with mixed outcomes" — 중요한데 본문에서 불충분하게 논의

**7. 새로움 평가**

- TT-OPD는 EMA teacher + privileged conditioning + cosine reward의 결합. NeurIPS 수준의 novelty인지 의문.

**8. 관련 연구 누락**

- STaR, ReST, STILL-2, Process reward 연구 (Math-Shepherd, OmegaPRM)

**9. 프레젠테이션 품질**

- Figure 중복. "Formal Analysis"라고 부르는 것은 과장 → "Analytical Insights"가 적절

### ⚠️ Specific: Hyperparameter 불일치

- **본문 Algorithm 1에서 $\lambda_{\text{KL}} = 0.01$, Appendix Table 5에서 $\lambda_{\text{KL}} = 4.0$ — 400배 차이! 심각한 불일치**

---

## 리뷰어 4: Akari Asai (UW/Meta) -- RAG, Multi-hop Reasoning, Self-RAG 전문가

### Overall Assessment
- **점수: 5/10**
- **추천: Weak Reject**

### Detailed Comments

**1. 불명확한 문장/표현**

- "bidirectional outcome-privileged conditioning": "bidirectional" 용어 혼란. "outcome-dependent" 또는 "outcome-conditioned"가 더 정확
- "outcome-conditioned process supervision": Process supervision은 step-level이며 TT-OPD는 trajectory-level. "outcome-conditioned regularization"이 정확

**2. 불명확한 실험**

- **Retrieval 품질 분석 부재**: search_evidence를 34.2%로 가장 많이 사용하는데, retrieved passages의 quality 분석 없음
- **Multi-hop reasoning 분석 부재**: turn별 reasoning의 질적 분석이 Figure 8 하나뿐
- **Knowledge base dependence**: KB 없이 0%는 완전한 KB 의존 → selective retrieval 능력 부재

**3. 누락된 실험**

- **Retrieval ablation**: (a) KB 없는 모델, (b) gold passages, (c) 전체 KB 비교
- **Knowledge grounding 분석**: attribution score 필요
- **Error analysis by turn**: cascading failure 분석
- **Self-reflection / self-correction 능력**: multi-turn의 핵심 장점인데 분석 부재

**6. 기술적 건전성**

- Outcome hint가 teacher logprob에 미치는 영향의 구현 불명확
- Turn-level truncation이 bias 도입 가능성

**7. 새로움 평가**

- Self-RAG와의 관계 미논의. HiLL과의 차이 불충분.

**8. 관련 연구 누락**

- **Self-RAG** (Asai et al., 2023) — 가장 관련성 높은데 인용 없음!
- CRAG, Chain-of-Verification, ToolkenGPT, Gorilla

**9. 프레젠테이션 품질**

- TODO comment 남아있음: Table 8의 "% TODO: verify base model per-subtype scores" (line 1466)
- Failure case 예시 필요

---

## 공통 지적사항 (Common Issues) -- 심각도 순위

### 1. [CRITICAL] 불공정한 실험 비교 (4인 모두 지적)

Track 1 (TT-OPD, Qwen3.5-9B, full-param) vs Track 2 (GRPO baselines, Lingshu-7B, LoRA)는 **완전히 다른 실험 설정**이다. 동일 모델에서의 공정한 비교 없이는 TT-OPD의 효과를 입증할 수 없다.

**권장 해결책**: Qwen3.5-9B에서 (a) base model + same tools/KB (multi-turn), (b) GRPO only, (c) TT-OPD를 모두 비교. 최소 3 seed.

### 2. [CRITICAL] Multi-seed 부재 및 통계적 유의성 (Neubig, Min)

TT-OPD 결과가 **single seed**이다. oscillating pattern을 보이는 불안정한 학습에서 single seed는 매우 불충분하다.

**권장 해결책**: 최소 3 seed로 TT-OPD를 재실행하고 mean ± std 보고.

### 3. [HIGH] Catastrophic Forgetting 미해결 (Lee, Min, Asai)

MMLU 84.9% -> 59.6%는 25pp 하락이며, single-turn 0%는 모델의 일반적 능력 파괴. 이를 "feature"로 framing하는 것은 문제.

**권장 해결책**: Multi-task learning이나 replay 기법. 또는 limitation에서 솔직하게 논의.

### 4. [HIGH] Partial / 미완성 결과 (4인 모두)

Table 1에서 $\dagger$ (partial) 결과 3개, "---" 결과 6개. TODO 주석이 본문에 남아있음.

**권장 해결책**: 모든 벤치마크 완전 평가 완료. TODO 제거.

### 5. [HIGH] Retrieval 효과 vs RL 효과 분리 불가 (Asai, Lee)

MedQA 87.1%가 RL 학습 결과인지, 828K passage KB 접근 효과인지 구분 불가.

**권장 해결책**: Base model (no RL) + multi-turn AgentRunner + 동일 KB baseline 추가.

### 6. [MEDIUM] Figure 중복 (Neubig, Min)

Figure 3과 Figure 4에서 TT-OPD curve가 동일 데이터로 반복.

**권장 해결책**: Figure 통합.

### 7. [MEDIUM] "Formal Analysis"의 비형식성 (Min, Neubig)

Proposition 1-3이 proof sketch만 제공.

**권장 해결책**: Section 제목을 "Analytical Insights"로 변경.

### 8. [MEDIUM] Hyperparameter 불일치 (Min)

본문 $\lambda_{\text{KL}} = 0.01$, Appendix Table 5에서 $\lambda_{\text{KL}} = 4.0$. **400배 차이**.

**권장 해결책**: 정확한 값 확인 및 통일.

### 9. [MEDIUM] Human evaluation 부재 (Lee)

의료 AI 논문에서 자동화 메트릭만으로는 불충분.

### 10. [LOW] 용어 혼란 (Asai)

"Bidirectional" → "outcome-conditioned", "process supervision" → "trajectory-level regularization".

---

### 최종 종합 평가

이 논문은 **흥미로운 문제 정의**(multi-turn agentic RL의 failure modes)와 **야심찬 환경 구축**(Healthcare AI GYM)을 제시하지만, 실험적 엄밀성에서 NeurIPS 기준에 미달한다. 가장 심각한 문제는: (1) 불공정한 cross-model 비교, (2) single-seed 결과, (3) retrieval 효과와 RL 효과의 미분리, (4) 다수의 미완성 결과이다. **이 네 가지를 해결하면 strong accept 수준의 논문이 될 잠재력이 있다.**
