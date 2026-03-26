# BIOAgents - Feedback Loop & Self-Check Prompts

> 작업 완료 후 이 문서의 프로토콜을 순서대로 통과해야 작업 완료.
> 오류 발생 시 해당 Pass를 다시 실행.
> Claude가 매 작업 세션 종료 시 반드시 참조해야 하는 문서.

---

## Last Updated: 2026-03-20 16:00 KST

---

## 5-Pass Self-Review Protocol

작업을 마무리하기 전에 아래 5개 Pass를 순서대로 통과해야 한다.
각 Pass는 독립적인 관점에서 검토하며, 이전 Pass의 맥락 없이 "신선한 눈"으로 본다.

---

### Pass 1: Logic & Correctness (코드 논리 검증)

> "내가 작성한 코드가 실제로 의도한 대로 동작하는가?"

```
1. 수정한 함수/메서드가 올바른 입력/출력 타입을 가지는가?
   → 함수 signature와 실제 caller를 대조 확인

2. None, empty list, 0 등 edge case에서 올바르게 동작하는가?
   → 특히: eval_tasks_per_domain=0 또는 None일 때 샘플링 건너뛰는가?
   → max_samples=0이 "샘플링 없음"인지 "0개 샘플"인지 확인

3. 조건 분기가 올바른가?
   → if/else 논리가 뒤집히지 않았는가?
   → None 체크와 0 체크를 혼동하지 않았는가? (0은 falsy지만 "설정 없음"과 다름)

4. 루프 범위가 올바른가?
   → off-by-one 오류 확인
   → list slicing 범위 확인

5. random.sample() 사용 시:
   → population 크기 < k이면 ValueError → min(k, len(population)) 처리 확인
   → 재현성이 필요하면 seed 설정 확인

체크 통과 여부: [ ] Pass / [ ] Fail (재작업 필요)
```

---

### Pass 2: Integration & Side Effects (통합 및 부작용 검증)

> "내 변경이 다른 코드와 올바르게 연결되는가? 의도치 않은 부작용은 없는가?"

```
1. 수정한 함수를 호출하는 모든 곳(callers)에서 변경이 안전한가?
   → 새 매개변수 추가 시 기본값이 backward-compatible한가?
   → 반환값 타입이 변경되었다면 모든 caller를 업데이트했는가?

2. Config 변경 사항이 실제로 로드되는 경로인가?
   → AutonomousGymConfig, AgentWorker, _dispatch_workers 등 chain 전체 확인
   → YAML 필드명과 Python dataclass 필드명이 정확히 일치하는가?

3. 현재 실행 중인 프로세스에 변경이 적용되었는가?
   → Python 프로세스는 소스 변경 즉시 적용 안됨 → 재시작 필요 여부 확인
   → 재시작이 필요하면 restart 방법을 명확히 기록

4. 파일 I/O 관련 변경:
   → checkpoint 경로가 존재하는가? (mkdir -p 필요한가?)
   → 파일 권한 문제는 없는가?
   → 기존 checkpoint와 새 형식이 호환되는가?

5. 병렬 실행 안전성:
   → 여러 프로세스가 같은 checkpoint 파일에 동시 쓰기하는가?
   → race condition 가능성 확인

체크 통과 여부: [ ] Pass / [ ] Fail (재작업 필요)
```

---

### Pass 3: Data Safety & Anti-Contamination (데이터 안전성 검증)

> "내 변경이 data contamination을 유발하거나 방지를 우회하지 않는가?"

```
1. 새로운 데이터 파일을 로딩하는가?
   → 파일명에 "test"가 포함되는가? → 포함 시 STOP, 사용 금지
   → train/test split이 명확히 분리되어 있는가?

2. GYM task 생성 시:
   → `*_test.jsonl`, `*_test.json` 파일 사용 금지 확인
   → `split_tasks.json`에서 `split="train"` 사용 확인
   → `data/medical_qa_200/` 경로 사용 여부 확인 (DEPRECATED, 오염됨)

3. Trajectory 추출/사용 시:
   → test task ID가 포함되지 않는지 확인
   → `_EXTERNAL_TEST_DOMAINS` frozenset 우회하는 코드 없는지 확인

4. 새 벤치마크/평가 추가 시:
   → train split이 있다면 train split만 사용
   → 외부 벤치마크 test set은 eval-only (학습/GYM 데이터 절대 불가)

5. Replay buffer에 외부 test 데이터가 포함될 경로는 없는가?
   → WebRL replay: GYM internal task trajectory만 포함 확인
   → `agent_runner.py`의 replay mix 코드에서 source 필터링 확인

체크 통과 여부: [ ] Pass / [ ] Fail (재작업 필요)
```

---

### Pass 4: Resource & Performance (리소스 및 성능 검증)

> "내 변경이 GPU/메모리/시간을 안전하게 사용하는가?"

```
1. 새 모델을 사용하거나 모델 설정을 변경하는가?
   → 10B+ 모델: gpu_memory_utilization ≤ 0.70 확인
   → 10B+ 모델: inference_batch_size = 1로 시작
   → VL 모델: AutoModelForVision2Seq 사용 확인 (CausalLM 금지)
   → model_type 비교: substring 매칭 확인 ("vl" in model_type.lower())

2. EHR benchmark 실행 시:
   → max_samples 설정 확인 (0 = 전체 57K → 금지)
   → 예상 완료 시간: tasks × 160s ÷ 3600 = N시간
   → checkpoint가 활성화되어 있는가?

3. Autonomous GYM eval_tasks_per_domain 설정 시:
   → RunConfig(task_ids=sampled_task_ids)에 실제로 전달되는지 확인
   → eval 예상 시간: tasks_per_domain × domains × 30s = N분

4. 타임아웃 설정 확인:
   → eval_timeout_seconds가 eval 예상 시간 × 2 이상인가?
   → train_timeout_seconds가 training 예상 시간 × 2 이상인가?
   → 너무 짧으면 조기 종료, 너무 길면 hang 감지 불가

5. 병렬 처리 설정:
   → GPU 할당이 겹치지 않는가? (각 모델이 다른 GPU 사용)
   → CUDA_VISIBLE_DEVICES 설정 확인

체크 통과 여부: [ ] Pass / [ ] Fail (재작업 필요)
```

---

### Pass 5: Documentation & Deployment (문서화 및 배포 검증)

> "내 변경이 문서화되었고, 다음 세션에서도 이해 가능한가?"

```
1. progress_check.md 업데이트 여부:
   → 코드 수정 이력 (파일, 라인, 내용) 추가했는가?
   → 현재 실행 상태 (진행률, score) 업데이트했는가?
   → 재시작 필요한 프로세스 기록했는가?

2. bug_report.md 업데이트 여부:
   → 새로 발견한 버그 추가했는가?
   → 수정한 버그의 상태를 FIXED로 변경했는가?

3. 재시작 필요한 프로세스 체크:
   → 수정된 코드가 현재 실행 중인 프로세스에 적용되는가?
   → 재시작 커맨드가 progress_check.md에 기록되었는가?

4. 다음 세션을 위한 TODO 업데이트:
   → summary_and_planning.md의 TODO 리스트 업데이트했는가?
   → 완료된 항목 체크 표시했는가?
   → 새로 발견한 작업 항목 추가했는가?

5. 환경 변수 / 경로 변경 시:
   → 환경 정보 섹션 업데이트했는가?
   → MEMORY.md 업데이트가 필요한가?

체크 통과 여부: [ ] Pass / [ ] Fail (재작업 필요)
```

---

## 작업 유형별 추가 체크리스트

### Autonomous GYM 코드 수정 후

```
Q1. eval_tasks_per_domain 값이 RunConfig(task_ids=...)에 실제로 전달되는가?
    → autonomous_agent.py:_evaluate() 내 RunConfig 생성 코드 확인

Q2. AgentWorker에 전달되는 timeout이 AutonomousGymConfig에서 오는가?
    → autonomous_gym.py:_dispatch_workers() → AgentWorker 생성 코드 확인

Q3. 수정 사항이 현재 실행 중인 프로세스에 적용되는가?
    → 적용 안된다면: 프로세스 종료 후 재시작 방법 기록

Q4. Dryrun config의 timeout이 eval_tasks_per_domain 기준 충분한가?
    → eval_tasks_per_domain=2, 3 domains → ~10-30분 → eval_timeout=3600으로 충분
```

### 벤치마크 평가 코드 수정 후

```
Q1. evaluator가 기대하는 답변 포맷이 모델 출력 포맷과 일치하는가?
    → VQA: single-word answer vs 설명형 답변 → EM 기준으로 0점 가능
    → 포맷 불일치 시 flexible evaluator 작성 (contains, F1 등 병용)

Q2. 결과 파일이 올바른 경로에 저장되는가?
    → logs/ 경로 확인, disk space 확인

Q3. N개 샘플 평가 후 전체 추론 가능한가?
    → evaluation result가 N/total 기준 representative sample인지 확인
```

### Config 파일 수정 후

```
Q1. YAML 필드명과 Python dataclass 필드명이 정확히 일치하는가?
    → snakecase vs camelcase 혼용 없는가?
    → 오타 없는가?

Q2. 수정한 YAML이 실제로 로드되는 경로인가?
    → load_config() 또는 dataclasses.fields() 방식으로 어떻게 파싱되는지 확인

Q3. 기존 config와의 backward compatibility:
    → 새 필드에 기본값 설정했는가?
    → 기존 YAML 파일들이 새 schema에서도 동작하는가?
```

---

## 긴급 재시작 프로토콜

현재 실행 중인 프로세스가 수정 전 코드라면 반드시 재시작해야 한다.

### Dryrun 재시작
```bash
# 1. 현재 프로세스 확인
ps aux | grep run_autonomous_gym | grep -v grep

# 2. 종료
ps aux | grep run_autonomous_gym | grep -v grep | awk '{print $2}' | xargs kill -9

# 3. 재시작
cd /data/project/private/minstar/workspace/BIOAgents
.venv/bin/python scripts/run_autonomous_gym.py \
  --config configs/autonomous_gym_dryrun.yaml \
  > logs/autonomous_gym_dryrun/launcher_restart.log 2>&1 &

# 4. 검증 (1시간 내)
tail -f logs/autonomous_gym_dryrun/launcher_restart.log | grep -E "cycle|domain|task_ids|completed|error"
# 기대: task_ids=[...] 에 eval_tasks_per_domain 개수만큼만 있는지 확인
```

### EHR Baseline Eval 재시작 (stratified 5K)
```bash
# 1. 현재 benchmark 프로세스 확인
ps aux | grep run_all_benchmarks | grep -v grep

# 2. EHR 부분만 재실행하거나 전체 재시작 (결과 있는 부분은 자동 skip)
cd /data/project/private/minstar/workspace/BIOAgents
.venv/bin/python scripts/run_all_benchmarks_parallel.py \
  --models lingshu qwen2vl step3vl \
  --benchmarks ehr \
  > logs/full_baseline_20260309/ehr_restart.log 2>&1 &
```

---

## Self-Review 완료 기록

| 날짜 | 작업 내용 | 통과한 Pass | 발견된 문제 | 조치 |
|------|-----------|-------------|------------|------|
| 2026-03-11 | Dryrun 타임아웃 수정 (A-1a~e) | 1~5 통과 | EHRTaskResult vars() 사용 발견 (B017) | dataclasses.asdict()로 수정 |
| 2026-03-11 | EHR stratified sampling 추가 (A-3a~c) | 1~5 통과 | max_samples=0이 기본이었음 (B016) | EHR_MAX_SAMPLES=5000으로 수정 |
| 2026-03-11 | Step3 OOM 방지 (A-2) | 1~4 통과 | gpu_memory_utilization 미설정 (B018) | 0.70으로 설정 |
| 2026-03-12 | GPU5/7 gym timeout 설정 (B-1,2) | 1~5 통과 | eval_timeout 미설정 → 24h 기본값 (B021) | 3600초로 설정, 프로세스 재시작 |
| 2026-03-12 | WORKER_SCRIPT 이스케이프 수정 (B-3) | 1~5 통과 | {EHR_MAX_SAMPLES} 미이스케이프 (B022) | {{}} 이스케이프 + dry-run 테스트 확인 |
| 2026-03-12 | EHR/Step3-VL 재실행 | 1~4 통과 | GPU 4 충돌 가능성 확인 | 모니터링 중 |
| 2026-03-12 | Step3-VL EM=0 근본 원인 수정 (B-5,6) | 1~5 통과 | B024: key_mapping 미적용, B025: `<think>` 미제거 | key_mapping 전달 + think-strip, EM=0.200 확인 |
| 2026-03-16 | GRPO training silent fail 수정 (B028/B029) | 1~5 통과 | B028: max_train_tasks 필드 누락→TypeError, B029: stderr 무시 | 필드 추가 + stderr 로깅 + dry-run 테스트 + 재시작 |
| 2026-03-16 | B030~B033 일괄 수정 | 1~5 통과 | B030: worker config 7필드 누락, B031: dryrun timeout, B032: Step3 TextQA/MedLFQA key_mapping 누락, B033: 8-agent GYM 충돌 | config 필드 추가 + timeout 증가 + key_mapping/think-strip + 프로세스 정리 + 재시작 |
| 2026-03-16 | B032 최종 해결: Qwen3 init hang | 1~5 통과 | `from_pretrained()` 자체가 Qwen3Model full-size init에서 hang (transformers 4.57.6). CPU/GPU 무관. safetensors 파일 정상. | `_load_step3vl_manual()` 헬퍼: `no_init_weights()` + 수동 safetensors 로딩. 테스트: 14s load + 정상 생성. TextQA/MedLFQA 양쪽 적용 |
| 2026-03-17 | B034 수정: Step3 import 누락 | 1~5 통과 | `_load_step3vl_manual()`에서 `AutoConfig` 사용하지만 함수 내 import 없음 | 함수 내 import 추가. 구문 검증 + TextQA 실행 성공 확인 |
| 2026-03-17 | B035 수정: Domain herding 근본 수정 | 1~5 통과 | 3가지 동시 원인: 절대 threshold(0.1), 동일 weakness 점수(0.8), recency penalty 없음 | 상대 threshold(avg*0.15), 연속 weakness 점수, recency penalty 추가. math 검증 통과. GPU 5/6/7 v5 재시작 |
| 2026-03-18 | B036 수정: Step3-VL batch size 크래시 | 1~5 통과 | `get_input_embeddings()` squeeze/unsqueeze가 batch>1에서 4D tensor 생성 → apply_rotary_pos_emb 크래시 | `BATCH_SIZE = 1 if is_step3 else 8` (TextQA+MedLFQA 두 곳). 구문 검증 + batch=1 shapes 정상 확인 |
| 2026-03-18 | B037 수정: Step3 TextQA thinking 잘림 | 1~5 통과 | `<think>\n` generation prompt + max_new_tokens=32 → thinking 잘림 → MedMCQA 21%/MMLU 31.5% (below random) | `<think>\n` 제거 + max_new_tokens=64 (Step3만). MedLFQA에도 thinking 비활성화. 구문 검증 통과 |
| 2026-03-18 | B038 수정: model_profile Step3 manual loading | 1~5 통과 | `load_model()` → `from_pretrained()` → Step3 key_mapping 미적용 + Qwen3 init hang → EHR action_score=0.000 | `_load_step3vl_manual()` 메서드 추가, `device_map="cpu"` 지원(LoRA merge). ModelProfile 테스트 + 구문 검증 통과 |
| 2026-03-20 | B039 수정: Step3 TextQA B037 regression | 1~5 통과 | B037 fix 잘못 적용: `<think>\n` strip 누락 + max_new_tokens=512 → 생성 thinking text에서 잘못된 답 추출 → ~10% | `<think>\n` strip(TextQA+MedLFQA) + max_new_tokens 64 + `</think>` fallback strip. 구문 검증 + 10-sample test (MedMCQA/MMLU 30% > random 25%) |
| 2026-03-20 | B040 수정: PMC-VQA HF 로딩 | 1~5 통과 | train_2.csv 스키마 불일치 (extra columns, missing Answer_label) | `data_files` 파라미터로 train.csv/test.csv만 지정 |
| 2026-03-20 | GYM External Validation (B041 발견) | N/A (eval only) | **Qwen GRPO merge 모델 완전 파괴** (MedQA 54%→10%, early/late 모두 동일). **Lingshu catastrophic forgetting** (MedMCQA +7.6% 향상, MedQA -4.8%, MMLU -10.1% 하락). | B041 bug report 작성. 대안 A(merge 수정)/B(scope 축소)/C(LoRA adapter 진단)/D(Few-shot/RAG) 정리. 03-21 결정 예정 |
