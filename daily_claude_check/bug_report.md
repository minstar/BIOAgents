# BIOAgents - Claude Bug Report & 재발 방지 기록

> Bug 발생 → 원인 분석 → 해결 → 재발 방지 패턴 정리.
> 같은 실수를 두 번 반복하지 않기 위한 참조 문서.
> 상세 원본 기록: `BUGLOG.md`

---

## Last Updated: 2026-03-20 14:00 KST

---

## Bug Index

| ID | 카테고리 | 요약 | 발생일 | 상태 |
|----|---------|------|--------|------|
| B001 | Data Contamination | 외부 벤치마크 TEST 파일로 GYM task 생성 | 02-12 | FIXED |
| B002 | Data Contamination | SFT 데이터에 test split task 포함 (32.5% 오염!) | 02-12 | FIXED |
| B003 | Data Contamination | Trajectory에서 test task 미필터링 | 02-13 | FIXED |
| B004 | Data Contamination | medical_qa_200 오염 데이터 사용 | 02-13 | FIXED |
| B005 | Data Pipeline | Contamination checker nested JSON 미파싱 | 02-12 | FIXED |
| B006 | Model Loading | VL 모델을 AutoModelForCausalLM으로 로딩 시도 | 02-13 | FIXED |
| B007 | Model Loading | model_type "qwen2_5_vl_text" 미인식 | 02-13 | FIXED |
| B008 | GYM Env | env.reset() 반환값 str vs dict 불일치 | 02-13 | FIXED |
| B009 | GYM Env | Gymnasium 환경 ID 불일치 | 02-12 | FIXED |
| B010 | Dependencies | prometheus_client 미설치 | 02-12 | FIXED |
| B011 | Config | SelfPlayConfig 인자명 불일치 | 02-12 | FIXED |
| B012 | Config | SFT config 모델 경로 오류 | 02-12 | FIXED |
| B013 | Data Pipeline | MedMCQA train 파일 nested list 포맷 | 02-13 | FIXED |
| B014 | Autonomous GYM | eval_tasks_per_domain이 RunConfig에 전달 안됨 → 전체 task 실행 | 03-11 | FIXED |
| B015 | Autonomous GYM | AgentWorker timeout 하드코딩 (86400) → YAML 설정 불가 | 03-11 | FIXED |
| B016 | EHR Evaluation | MIMIC-III max_samples=0 → 57,151 tasks 전체 실행 (수 주 소요) | 03-11 | FIXED |
| B017 | EHR Evaluation | EHRTaskResult checkpoint에서 vars() 사용 → 직렬화 불안정 | 03-11 | FIXED |
| B018 | Model Loading | Step3-VL-10B gpu_memory_utilization 미설정 → OOM/SIGTERM | 03-11 | FIXED |
| B019 | Evaluation | Step3-VL EM=0 on all VQA benchmarks (model_type 미인식) | 03-11 | ✅ FIXED (03-12) |
| B020 | Evaluation | VQA-Med-2021 / Quilt-VQA EM=0 (모든 모델) | 03-11 | ℹ️ WON'T FIX |
| B021 | Autonomous GYM | gym_gpu5/gpu7 config에 eval_timeout_seconds 미설정 → 24h 기본값 | 03-12 | ✅ FIXED |
| B022 | Script | WORKER_SCRIPT 내 {EHR_MAX_SAMPLES} 이스케이프 누락 → KeyError | 03-12 | ✅ FIXED |
| B023 | EHR Evaluation | stratified sampling에서 dict를 set()에 넣어 unhashable type 에러 | 03-12 | ✅ FIXED |
| B024 | Model Loading | Step3-VL _checkpoint_conversion_mapping 미적용 → 가중치 랜덤 초기화 → EM=0 | 03-12 | ✅ FIXED |
| B025 | Evaluation | Step3-VL `<think>...</think>` 추론 텍스트가 예측에 포함되어 EM=0 | 03-12 | ✅ FIXED |
| B026 | Autonomous GYM | quality_threshold 고정값 → 학습 데이터 0건 (점수 미달) | 03-12 | ✅ FIXED |
| B027 | GRPO Training | GRPO 학습 후 성능 degradation (7개 원인 동시 수정) | 03-14 | ✅ FIXED |
| B028 | GRPO Training | `max_train_tasks` 필드 누락 → BioAgentGRPOConfig TypeError → training 100% 실패 | 03-16 | ✅ FIXED |
| B029 | Subprocess | `_run_subprocess` stderr 무시 → training 실패 silent (로그 없음) | 03-16 | ✅ FIXED |
| B030 | Autonomous GYM | `_build_cycle_script()`에 max_train_tasks/adaptive_quality/use_dr_grpo 등 7개 필드 누락 → worker에 기본값만 전달 | 03-16 | ✅ FIXED |
| B031 | Config | Dryrun eval_timeout_seconds=3600 → 전체 사이클(~100분)보다 짧아 매시간 timeout → 결과 0건 | 03-16 | ✅ FIXED |
| B032 | Evaluation | Step3-VL TextQA/MedLFQA: Qwen3 init hang (transformers 4.57.x) → `_load_step3vl_manual()` workaround | 03-16 | ✅ FIXED (final) |
| B033 | Resource Conflict | autonomous_gym.yaml (8-agent)이 per-GPU config과 동시 실행 → GPU 0-3 EHR baseline과 충돌, GPU 1 OOM 위험 | 03-16 | ✅ FIXED (프로세스 종료) |
| B034 | Model Loading | `_load_step3vl_manual()` 내 `AutoConfig`, `AutoModelForCausalLM`, `torch` import 누락 → NameError | 03-17 | ✅ FIXED |
| B035 | Autonomous GYM | Domain herding: 절대적 weakness threshold(0.1) + 동일 weakness 점수(0.8) + recency penalty 없음 → 85% 단일 도메인 반복 | 03-17 | ✅ FIXED |
| B036 | Evaluation | Step3-VL TextQA/MedLFQA BATCH_SIZE=8 → 4D tensor crash in apply_rotary_pos_emb (get_input_embeddings batch_size=1 전용) | 03-18 | ✅ FIXED |
| B037 | Evaluation | Step3-VL TextQA `max_new_tokens=32` + `<think>` generation prompt → thinking 잘림 → below-random MedMCQA(21%)/MMLU(31.5%) | 03-18 | ✅ FIXED |
| B038 | Model Loading | `model_profile.py:load_model()` Step3 key_mapping + manual loading 누락 → EHR eval action_score=0.000 | 03-18 | ✅ FIXED |
| B039 | Evaluation | Step3-VL TextQA B037 regression: `add_generation_prompt=True`의 `<think>\n` 미제거 + max_new_tokens=512 → thinking text에서 잘못된 답 추출 → 10% (51%에서 붕괴) | 03-20 | ✅ FIXED |
| B040 | Data Loading | PMC-VQA HuggingFace train_2.csv 스키마 불일치 → dataset generation 에러 | 03-20 | ✅ FIXED |
| B041 | GRPO Training | GRPO merge가 Qwen2.5-VL 모델을 완전히 파괴 → 외부 벤치마크 10% (baseline 54%) | 03-20 | 🔴 CRITICAL |

---

## 🔴 CRITICAL: Data Contamination 관련

### B001: 외부 벤치마크 TEST 파일로 GYM task 생성
**심각도**: CRITICAL

**증상**: MedQA/MedMCQA/MMLU 평가 점수가 비정상적으로 높을 수 있음 (테스트 문제를 학습에 사용)

**근본 원인**: `scripts/generate_gym_data.py`가 `*_test.jsonl` 파일에서 GYM task 생성.
이 task들로 SFT/GRPO 학습하면 evaluation에서 이미 본 문제를 풀게 됨.

**수정**: `*_train_gpt4.jsonl` 파일만 사용. MMLU는 train split 없으므로 GYM에서 완전 제외.

**재발 방지**:
> **절대 `*_test.jsonl`, `*_test.json` 파일에서 학습/GYM 데이터를 만들지 말 것.**
> GYM task 생성 코드에 CONTAMINATION GUARD assertion 필수:
```python
assert "test" not in str(benchmark_path).lower(), f"CONTAMINATION: {benchmark_path} is a test file!"
```

---

### B002: SFT 데이터에 test split task 포함 (32.5% 오염!)
**심각도**: CRITICAL

**증상**: p2 SFT 데이터 228/701 샘플이 test task ID 포함 → 평가 결과 신뢰 불가

**근본 원인**: `load_domain_tasks()`가 `split_tasks.json`의 train/test 구분 없이 전체 task 로딩.

**재발 방지**:
> SFT 데이터 생성 시 반드시 `split="train"` 명시적 인자 전달.
> 생성 후 `scripts/check_contamination.py` 자동 실행 필수.

---

### B003/B004: Trajectory/medical_qa_200 오염
**심각도**: CRITICAL / HIGH

**재발 방지**:
> Trajectory 추출 시 모든 test task ID와 대조 필수 (`_load_all_test_ids()` 사용).
> `data/medical_qa_200/` 디렉토리는 DEPRECATED (오염됨) — 절대 사용 금지.

---

## 🟡 HIGH: Model Loading 관련

### B006: VL 모델을 AutoModelForCausalLM으로 로딩
**심각도**: HIGH

**증상**: `ValueError: Unrecognized configuration class Qwen2_5_VLConfig for AutoModelForCausalLM`

**근본 원인**: VL 모델 (Qwen2.5-VL, Lingshu)은 `Qwen2_5_VLForConditionalGeneration` 필요.
`AutoModelForCausalLM`으로는 로딩 불가.

**재발 방지**:
> 모든 모델 로딩 코드에서 `AutoModelForVision2Seq` 또는 모델 타입 감지 후 분기 필수.
```python
# 올바른 패턴:
from transformers import AutoConfig, AutoModelForVision2Seq, AutoModelForCausalLM
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model_type = getattr(config, "model_type", "")
is_vl = "qwen2" in model_type.lower() and "vl" in model_type.lower()
if is_vl:
    model = AutoModelForVision2Seq.from_pretrained(model_path, ...)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, ...)
```

---

### B007: model_type "qwen2_5_vl_text" 미인식
**심각도**: MEDIUM

**증상**: SFT 후 저장된 체크포인트 model_type이 `qwen2_5_vl_text`인데, 정확 매칭 실패.

**재발 방지**:
> model_type 비교는 반드시 **substring 매칭** 사용. 정확한 문자열 리스트 매칭 금지.
```python
# 올바름: "vl" in model_type.lower()
# 잘못됨: model_type in ("qwen2_5_vl", "qwen2_vl")
```

---

### B018: Step3-VL-10B gpu_memory_utilization 미설정 → OOM/SIGTERM
**심각도**: HIGH

**증상**: baseline eval 및 Autonomous GYM 실행 중 process crash (code=-15, SIGTERM).

**근본 원인**: 10B 모델에 explicit `gpu_memory_utilization` 설정 없이 기본값 0.85 적용.
Step3-VL-10B는 메모리 요구가 높아 OOM 발생 → kernel이 SIGTERM으로 강제 종료.

**수정**: `configs/autonomous_gym.yaml`의 Step3 에이전트들에 추가:
```yaml
gpu_memory_utilization: 0.70   # 10B 모델 보수적 VRAM 설정
inference_batch_size: 1        # 단일 샘플 추론
```

**재발 방지**:
> 10B+ 모델 사용 시 반드시 `gpu_memory_utilization: 0.70` 이하로 설정.
> `inference_batch_size: 1` 기본값으로 시작 후 점진적으로 늘릴 것.
> Crash 후 로그에서 signal -15 확인되면 OOM을 먼저 의심할 것.

---

## 🟡 HIGH: GYM Environment 관련

### B008: env.reset() 반환값 str vs dict 불일치
**심각도**: HIGH

**증상**: `AttributeError: 'str' object has no attribute 'get'`

**근본 원인**: `BioAgent-v0`의 `reset()`이 `(str, dict)` 튜플을 반환하는데, `obs.get()` 호출.

**재발 방지**:
> `env.reset()` → `(obs_str, info_dict)` 구조. obs는 str이므로 `.get()` 호출 불가.
```python
obs, info = env.reset()
# obs = str (observation text)
# info = dict (metadata)
```

---

### B009: Gymnasium 환경 ID 불일치
**심각도**: HIGH

**증상**: `gymnasium.error.NameNotFound: Environment 'bioagent-gym' doesn't exist`

**수정**: `gym.make("BioAgent-v0")` (정확한 대소문자 포함)

**재발 방지**:
> 환경 ID 상수 사용: `GYM_ENV_ID = "BioAgent-v0"`
> 직접 문자열 사용 금지.

---

## 🟠 MEDIUM: Autonomous GYM 관련 (2026-03-11 신규)

### B014: eval_tasks_per_domain이 RunConfig에 전달 안됨 → 전체 task 실행
**심각도**: HIGH (dryrun 24시간 타임아웃의 근본 원인)

**증상**: `eval_tasks_per_domain: 2`로 설정했는데 수백 개 전체 task 실행 → 24시간 타임아웃.

**근본 원인**: `autonomous_agent.py:_evaluate()`에서 `RunConfig(task_ids=None)` 생성.
`agent_runner.py:run_all_tasks()`는 `task_ids=None`이면 모든 task 실행.
`eval_tasks_per_domain` 값이 `RunConfig`에 전혀 반영되지 않았음.

**수정**: `_evaluate()` 내 `RunConfig` 생성 전 task sampling 추가:
```python
max_eval = self.config.eval_tasks_per_domain or 0
if max_eval > 0:
    all_ids = [t["id"] for t in env._tasks]
    sampled_task_ids = random.sample(all_ids, min(max_eval, len(all_ids)))
    run_config = RunConfig(..., task_ids=sampled_task_ids)
```

**재발 방지**:
> `RunConfig(task_ids=None)` 생성 전 반드시 `eval_tasks_per_domain` 값을 확인할 것.
> `task_ids=None`의 의미는 "모든 task 실행"임을 항상 기억할 것.
> Dryrun config에서 `eval_tasks_per_domain` 값이 실제로 적용되는지 첫 실행 로그 확인 필수.

---

### B015: AgentWorker timeout 하드코딩 (86400초) → YAML 설정 불가
**심각도**: MEDIUM

**증상**: `autonomous_gym_dryrun.yaml`에서 timeout 설정을 변경해도 실제로 무시됨.
Dryrun이 항상 24시간 대기.

**근본 원인**: `autonomous_gym.py:AgentWorker.run()`의 `_run_subprocess()` 호출에
`timeout=86400`이 하드코딩되어 있음. `AutonomousGymConfig`에 해당 필드 없었음.

**수정**:
- `AutonomousGymConfig`에 `eval_timeout_seconds: int = 86400`, `train_timeout_seconds: int = 7200` 추가
- `AgentWorker.__init__()`에 `eval_timeout`, `train_timeout` 인자 추가
- `_dispatch_workers()`에서 config 값 전달

**재발 방지**:
> 모든 subprocess timeout은 YAML config로 제어 가능해야 함.
> Hardcoded timeout 값 금지 — 새로 추가할 때도 config field로 노출할 것.

---

## 🟠 MEDIUM: EHR Evaluation 관련 (2026-03-11 신규)

### B016: MIMIC-III max_samples=0 → 57,151 tasks 전체 실행
**심각도**: HIGH (완료까지 수 주 소요)

**증상**: MIMIC-III baseline eval이 1.5% (830/57K) 이후 진전 없음 — 실제론 수 주 예상.

**근본 원인**: `run_all_benchmarks_parallel.py:run_ehr()`에서 `max_samples=0`으로 설정.
`ehr_benchmark_eval.py`에서 `max_samples=0`이면 샘플링 없이 전체 실행.

**수정**: `EHR_MAX_SAMPLES = 5000` (stratified sampling)으로 변경.
현재 실행 중인 프로세스는 수정 전 코드 → **재시작 필요**.

**재발 방지**:
> EHR benchmark (`max_samples=0`)는 절대 production 실행에 사용하지 말 것.
> 기본값을 `max_samples=5000`으로 변경하여 실수로 전체 실행되는 것 방지.
> 새 EHR 실험 시작 전 `max_samples` 값 반드시 확인.

---

### B017: EHRTaskResult checkpoint에서 vars() 사용 → 직렬화 불안정
**심각도**: MEDIUM

**증상**: checkpoint 저장/로드 시 dataclass field 불일치로 에러 발생 가능성.

**근본 원인**: `vars(result)`는 dataclass의 `__dict__`를 반환하는데, private field나
property가 있으면 예상치 못한 값 포함. `dataclasses.asdict()`가 올바른 방법.

**수정**:
```python
# 잘못됨:
json.dump(vars(r), f)
EHRTaskResult(**json_data)

# 올바름:
import dataclasses
json.dump(dataclasses.asdict(r), f)
known_fields = {f.name for f in dataclasses.fields(EHRTaskResult)}
EHRTaskResult(**{k: v for k, v in json_data.items() if k in known_fields})
```

**재발 방지**:
> dataclass 직렬화는 항상 `dataclasses.asdict()` 사용.
> 역직렬화 시 알려진 field만 필터링하여 forward-compatible하게 작성.

---

## ✅ RESOLVED: 평가 이슈

### B019: Step3-VL-10B EM=0 on all VQA benchmarks → FIXED (03-12)
**심각도**: HIGH → **RESOLVED**

**증상**: VQA-RAD, SLAKE, PathVQA 모두 EM=0.000. `contains` 지표는 20-28%.

**근본 원인**: `vqa_benchmark_eval.py`의 VL 모델 감지 리스트에 `"step_robotics"` (Step3의 model_type)이 없었음.
→ `_is_vl_model=False` → text-only 생성 → `<im_patch>` placeholder가 토크나이저에서 garbage 출력 ("!!!...")

**수정**:
- `vqa_benchmark_eval.py` line 209: `"step_robotics"` 추가
- `_generate_step3vl()` 메서드 추가 (Step3VLProcessor 사용)
- VQA 재실행 (03-12): `logs/step3vl_vqa_rerun_20260312/`

**재발 방지**:
> 새 VL 모델 추가 시 반드시 `_is_vl_model` 감지 리스트에 model_type 추가.
> model_type은 `config.json`의 `"model_type"` 필드에서 확인.

---

### B020: VQA-Med-2021 / Quilt-VQA EM=0 → WON'T FIX (open-ended VQA 특성)
**심각도**: MEDIUM → **WON'T FIX**

**증상**: Lingshu와 Qwen 모두 두 benchmark에서만 EM=0. token_F1은 4-27%.

**분석**: 두 벤치마크는 open-ended 답변 형식 → EM(정확 매칭) 기준 0점은 정상.
모델은 의미 있는 응답 생성 중 (token_F1 > 0).

**조치**: 논문 Table에서 token_F1을 primary metric으로 사용. 별도 코드 수정 불필요.

---

### B021: gym_gpu5/gpu7 config에 eval_timeout_seconds 미설정 → FIXED (03-12)
**심각도**: HIGH (GPU 5,7이 24시간 단위로만 사이클 돌 수 있었음)

**증상**: GPU5 lingshu_weakness_fixer와 GPU7 qwen25vl_weakness_fixer가 eval lease에서 수시간째 stuck.
dryrun(GPU 6)은 정상 동작하는데 GPU5/7만 문제.

**근본 원인**: `configs/gym_gpu5_lingshu.yaml`과 `configs/gym_gpu7_qwen25vl.yaml`에 `eval_timeout_seconds` / `train_timeout_seconds` 미설정.
→ `AutonomousGymConfig` 기본값 86400초(24h) 적용 → task당 ~2.5시간이면 3 tasks = 7.5시간 대기.

**수정**: 두 config에 `eval_timeout_seconds: 3600`, `train_timeout_seconds: 3600` 추가 → 프로세스 재시작.

**재발 방지**:
> 새 gym config 작성 시 반드시 `eval_timeout_seconds`와 `train_timeout_seconds` 명시적 설정.
> 기본값 86400초는 production full eval에만 적합.

---

### B022: WORKER_SCRIPT 내 {EHR_MAX_SAMPLES} 이스케이프 누락 → FIXED (03-12)
**심각도**: MEDIUM (Step3-VL VQA/EHR 재실행 차단)

**증상**: `scripts/run_all_benchmarks_parallel.py` 실행 시 `KeyError: 'EHR_MAX_SAMPLES'`

**근본 원인**: WORKER_SCRIPT 템플릿(triple-quoted string) 안의 f-string에서 `{EHR_MAX_SAMPLES}` 변수가
Python `.format()` 호출 시 템플릿 변수로 해석됨. `{{EHR_MAX_SAMPLES}}`로 이스케이프 필요.

**수정**: line 533의 `{EHR_MAX_SAMPLES}` → `{{EHR_MAX_SAMPLES}}`

**재발 방지**:
> WORKER_SCRIPT 템플릿 수정 시 모든 중괄호가 `{{` `}}`로 이스케이프되어 있는지 확인.
> 수정 후 반드시 `WORKER_SCRIPT.format(...)` dry-run 테스트 실행.

### B023: EHR stratified sampling에서 unhashable type: 'dict' → FIXED (03-12)
**심각도**: HIGH

**증상**: EHR eval 시작 즉시 `TypeError: unhashable type: 'dict'` → 3모델 모두 실패.

**근본 원인**: `ehr_benchmark_eval.py:191`에서 `set(sampled)` — tasks가 dict이므로 해싱 불가.

**수정**: `sampled_ids = {id(t) for t in sampled}` + `id(t) not in sampled_ids` 비교.

**재발 방지**:
> dict 리스트의 중복 제거 시 `set()` 사용 금지. `id()` 비교 또는 고유 키 기반 비교 사용.

---

### B024: Step3-VL _checkpoint_conversion_mapping 미적용 → 가중치 랜덤 초기화 → FIXED (03-12)
**심각도**: CRITICAL (모든 VQA 결과 EM=0의 진짜 원인)

**증상**: Step3-VL VQA에서 EM=0.000, 모델 로딩 시 "weights were not initialized" 경고.

**근본 원인**: `Step3VL10BForCausalLM`이 HuggingFace `VLMS` 리스트에 없어서
`_checkpoint_conversion_mapping`이 자동 적용되지 않음. Checkpoint 키 (`model.layers.*`)와
모델 키 (`model.language_model.layers.*`)가 불일치 → 언어 모델 가중치 전체 랜덤 초기화.

**수정**: `from_pretrained()`에 `key_mapping` 인자를 명시적으로 전달:
```python
step3_key_mapping = {
    "^vision_model": "model.vision_model",
    r"^model(?!\.(language_model|vision_model))": "model.language_model",
    "^vit_large_projector": "model.vit_large_projector",
}
AutoModelForCausalLM.from_pretrained(..., key_mapping=step3_key_mapping, ...)
```

**재발 방지**:
> Custom 모델 로딩 시 반드시 "weights not initialized" 경고 확인.
> `_checkpoint_conversion_mapping`이 자동 적용되려면 모델 클래스명이 `VLMS` 리스트에 있어야 함.
> 없는 경우 `key_mapping` 인자를 명시적으로 전달.

---

### B025: Step3-VL `<think>...</think>` 추론 텍스트 미제거 → FIXED (03-12)
**심각도**: HIGH (EM=0의 보조 원인)

**증상**: 가중치 수정 후에도 F1=0.052이고 EM=0. 모델 출력에 `<think>` 추론 체인 포함.

**근본 원인**: Step3-VL chat_template가 `add_generation_prompt=True`일 때 `<think>\n`을 삽입.
모델이 `<think>reasoning...</think>answer` 형식으로 출력 → EM 매칭 실패.

**수정**: `_generate_step3vl()` 반환 전 `<think>...</think>` 제거:
```python
response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
```

**결과**: 수정 후 VQA-RAD 50샘플에서 EM=0.200 확인 (0.000 → 0.200).

**재발 방지**:
> reasoning 모델 (Qwen3, Step3 등) 사용 시 `<think>` 태그 제거 후처리 필수.
> 새 모델 추가 시 chat_template의 generation prompt 확인.

---

### B026: quality_threshold 고정값 → adaptive quality filtering 도입 → FIXED (03-12)
**심각도**: HIGH (GPU5/7에서 학습 데이터 0건 — 학습 자체가 안 됨)

**증상**: GPU5 Lingshu (best 0.301), GPU7 Qwen2VL (best 0.286)의 eval 점수가 `quality_threshold=0.4`보다 낮아
replay buffer에 trajectory 0건 → GRPO 학습이 warm-start demonstration 없이 실행.
Dryrun(GPU6)은 `quality_threshold=0.3`이라 정상 동작.

**근본 원인**: 고정 `quality_threshold`는 모델의 현재 수준을 반영하지 못함.
새 모델이나 약한 도메인에서는 모든 점수가 threshold 미만 → 유효 trajectory 0건.

**수정**: Adaptive quality filtering 구현 (`autonomous_agent.py` lines ~1159-1195):
```python
# 점수 분포를 [0,1]로 정규화 후 [lo, hi] 범위만 선택
# "Zone of Proximal Development" — 너무 쉬운 것(이미 마스터)과 너무 어려운 것(학습 신호 없음) 제외
r_min, r_max = min(all_rewards), max(all_rewards)
lo_cutoff = r_min + adaptive_quality_lo * (r_max - r_min)  # default 0.2
hi_cutoff = r_min + adaptive_quality_hi * (r_max - r_min)  # default 0.8
```

**Config 변경** (`gym_gpu5_lingshu.yaml`, `gym_gpu7_qwen25vl.yaml`):
```yaml
quality_threshold: 0.1         # fallback (adaptive_quality가 우선)
adaptive_quality: true          # 점수 분포 기반 적응적 필터링
adaptive_quality_lo: 0.2        # 하위 20% 제외
adaptive_quality_hi: 0.8        # 상위 20% 제외
```

**재발 방지**:
> 새 gym config에 `adaptive_quality: true` 기본 설정.
> 고정 `quality_threshold`만 쓰면 새 모델에서 학습 불가할 수 있음.
> `quality_threshold`는 adaptive 비활성 시 fallback으로만 사용.

---

### B027: GRPO 학습 후 성능 degradation (7개 근본 원인 동시 수정) → FIXED (03-14)
**심각도**: CRITICAL (자율 학습 루프 자체가 동작하지 않음 — 논문 핵심 contribution)

**증상**: 모든 GYM agent (dryrun, GPU5, GPU7)에서 GRPO 학습 후 post_score < pre_score.
Dryrun lease 55~60에서 100% improvement 음수. GPU7은 post_score=0.0 빈발.

**근본 원인 (7개)**:
1. **KL penalty 부재**: `_grpo_policy_update()`에서 `beta*loss`만 하고 reference model 없음 → catastrophic forgetting
2. **전체 task 학습**: 도메인 모든 task를 매 epoch 학습 → 소수 데이터에 overfitting
3. **Pre/post eval 다른 task**: 랜덤 5개 task 비교 → 샘플링 분산이 improvement로 오인
4. **Reward hacking**: accuracy 25~30%뿐, format/process/safety가 70%+ → 길고 의료 용어 많은 응답 학습
5. **Replay advantage=1.0 고정**: GRPO rollout advantage 분포와 불일치
6. **LR scheduler 없음**: 상수 LR로 full gradient step → 불안정
7. **LR 2e-5 과다**: KL anchor 없이 높은 LR → 빠른 drift

**수정**:
- `_compute_ref_log_probs()`: LoRA disable → reference log-probs 계산
- `_grpo_policy_update()`: `L = -adv * log_π + β * (log_π - log_π_ref)` 실제 KL 항
- `max_train_tasks=20`: task cap으로 overfitting 방지
- `eval_seed`: pre/post eval 동일 task 보장
- accuracy reward: 0.30→0.50, format: 0.15→0.05
- replay advantage: rollout positive advantage 평균으로 정규화
- cosine LR scheduler (warmup 10% + decay)
- LR 2e-5→5e-6, training_epochs 2→1

**재발 방지**:
> GRPO/RL 학습 시 반드시 reference model KL penalty 포함.
> `beta * loss`는 KL penalty가 아님 — 반드시 `log π_θ - log π_ref` 계산.
> Pre/post eval은 반드시 동일 task set으로 비교.
> Reward에서 accuracy가 최소 50% 이상 차지하도록 설정.
> 새 RL 실험 시작 전 첫 사이클 improvement 부호 반드시 확인.

---

### B028: `max_train_tasks` 필드 BioAgentGRPOConfig에 누락 → TypeError → training 100% silent fail → FIXED (03-16)
**심각도**: CRITICAL (03-14부터 모든 GYM agent의 training이 0% 실행됨 — GPU 5/6/7 2일간 eval만 반복)

**증상**: 3개 GYM agent (dryrun lease_0001~0467, GPU5 lease_0001~0164, GPU7 lease_0001~0114) 모두
`improvement: 0.0`, `pre_score == post_score` 100% 발생. Eval만 반복, 학습 진행 없음.

**근본 원인**: 03-14 GRPO v2 작업(B027)에서 `max_train_tasks` 필드를 `MultiTurnGRPOConfig`(line 817)에만 추가.
`_train_grpo()` (autonomous_agent.py line 1772)에서 `_common_kwargs`에 `max_train_tasks=20`을 넣고
`BioAgentGRPOConfig(**_common_kwargs)` (line 1806) 호출 시 `TypeError: __init__() got an unexpected keyword argument 'max_train_tasks'` 발생.
Exception은 line 1831의 `except Exception` 에서 catch되어 `None` 반환 → training skip → `post_score = pre_score`.

**추가 원인 (B029)**: `_run_subprocess()` (autonomous_gym.py)가 `returncode==0`일 때 stderr를 무시.
Training exception의 traceback이 stderr로 출력되지만 부모 프로세스가 이를 읽지 않아 로그에 전혀 기록되지 않음.

**수정**:
- `BioAgentGRPOConfig` (grpo_trainer.py line 109)에 `max_train_tasks: int = 20` 추가
- `_run_subprocess()` (autonomous_gym.py)에 returncode==0일 때도 stderr 마지막 10줄 로깅 추가

**검증**: dry-run test 통과 — `BioAgentGRPOConfig(**_common_kwargs)` 정상 생성 확인.
GPU 5/6/7 프로세스 재시작 (PID 2514243/2514552/2514932).

**재발 방지**:
> 부모 dataclass에 필드를 추가할 때, 해당 필드를 kwargs로 전달하는 모든 caller 확인.
> `@dataclass` 상속 체인에서 새 필드는 반드시 **base class**에 추가 (child에만 추가하면 base 생성 시 TypeError).
> Subprocess의 stderr는 항상 로깅 (returncode 무관) — silent failure 방지.
> `_train_grpo()`의 except block에서 `logger.error()` + traceback 사용 (warning → error 격상).

---

### B029: `_run_subprocess` stderr 무시 → FIXED (03-16)
**심각도**: HIGH (B028의 2일간 미발견 직접 원인)

**증상**: GYM subprocess의 training exception traceback이 로그에 전혀 나타나지 않음.

**근본 원인**: `_run_subprocess()` (autonomous_gym.py line 797-817)에서 `proc.returncode == 0`일 때
stdout 마지막 20줄만 로깅하고 stderr를 완전 무시. subprocess 내 `_train_grpo()` exception은
`except Exception`으로 catch되어 정상 종료(exit code 0)하므로 stderr 로깅 경로를 타지 않음.

**수정**: returncode==0일 때도 stderr 마지막 10줄을 `logger.warning`으로 출력.

**재발 방지**:
> Subprocess stderr는 항상 로깅 — exit code가 0이어도 경고/에러가 stderr에 출력될 수 있음.
> `capture_output=True` 사용 시 proc.stderr를 반드시 확인하는 패턴 적용.

---

### B030: `_build_cycle_script()` 7개 config 필드 누락 → FIXED (03-16)
**심각도**: HIGH (use_dr_grpo=True가 GPU5/7에서 무시됨 — 2일간 Dr.GRPO 미적용)

**증상**: GPU5 lingshu의 `use_dr_grpo: true` (YAML) 설정이 worker script에 반영되지 않음 → 기본값 `False`로 실행.
`max_train_tasks`, `adaptive_quality`, `adaptive_quality_lo/hi`, `use_fair_grpo`, `fairness_weight`, `max_fairness_gap` 동일 현상.

**근본 원인**: `autonomous_gym.py:_build_cycle_script()` (line 832-910)에서 `AutonomousAgentConfig` 생성 시
03-12~14에 추가된 7개 필드가 worker script 템플릿에 포함되지 않음.
기본값이 reasonably safe하여 (max_train_tasks=20, adaptive_quality=True) 동작은 했으나 `use_dr_grpo` 같은 필드는 효과 없었음.

**수정**: `_build_cycle_script()` 템플릿에 7개 필드 추가. GPU 5/6/7 프로세스 재시작.

**재발 방지**:
> `AutonomousAgentConfig`에 새 필드 추가 시 반드시 `_build_cycle_script()` 템플릿도 함께 업데이트.
> Worker script 생성 후 `/tmp/gym_worker_*.py`에서 새 필드 존재 확인.

---

### B031: Dryrun eval_timeout_seconds=3600 → timeout으로 결과 0건 → FIXED (03-16)
**심각도**: HIGH (dryrun이 5시간 동안 결과 0건 생산)

**증상**: Dryrun(GPU 6)이 lease_0005까지 진행했지만 `_run_subprocess` 완료 결과 0건.
매 1시간마다 subprocess가 timeout으로 kill되어 새 lease 할당.

**근본 원인**: `eval_timeout_seconds: 3600`(1시간)이 전체 사이클 소요 시간보다 짧음.
3 domains × 2 tasks/domain × ~7min/task × 2(pre+post eval) + training = ~100분.
`_run_subprocess(timeout=3600)` → 60분에 subprocess kill → JSON 출력 없음 → 결과 손실.

**수정**: `eval_timeout_seconds: 7200` (2시간)으로 변경.

**재발 방지**:
> 전체 사이클 소요 시간 계산: `domains × tasks/domain × min_per_task × 2 + training_min`
> `eval_timeout_seconds`는 예상 소요 시간의 최소 1.5배 이상으로 설정.
> 새 dryrun config 작성 시 eval_tasks_per_domain과 timeout의 관계 반드시 확인.

---

### B032: Step3-VL TextQA/MedLFQA에 key_mapping + think-strip 누락 → FIXED (03-16)
**심각도**: HIGH (B024/B025의 반복 — run_full_benchmark_suite.py에 미적용)

**증상**: Step3-VL TextQA 실행 시 "weights were not initialized" 경고 후
`RuntimeError: tensor size mismatch (32 vs 370)` 크래시.

**근본 원인**: `run_full_benchmark_suite.py`의 TextQA/MedLFQA 모델 로딩 코드에
B024(key_mapping)와 B025(think-strip) 수정이 적용되지 않음.
VQA 코드(`vqa_benchmark_eval.py`)에만 수정되어 있었음.

**수정**:
1. TextQA/MedLFQA 모델 로딩에 `is_step3` 분기 추가 + key_mapping 전달
2. `<think>...</think>` 제거 후처리 추가
3. Step3-VL에 `attn_implementation="sdpa"` 제외 — `__lock_page_killable`에서 hang 발생
4. **최종 root cause (03-16 16:30)**: `from_pretrained()` 자체가 hang — Qwen3Model의 random weight 초기화(`__init__` → `post_init()`)가 full-size(4096 hidden, 36 layers)에서 무한 hang (transformers 4.57.6). CPU-only, device_map=None/auto/cpu 모두 동일. safetensors 파일 자체는 정상 접근 가능.
5. **최종 해결**: `_load_step3vl_manual()` 헬퍼 함수 — `no_init_weights()` + 수동 safetensors 로딩 + GPU 이동. 로딩 시간: ~14s (기존 from_pretrained: 무한 hang). `run_full_benchmark_suite.py`에 공통 함수로 추출 완료.

**재발 방지**:
> Step3-VL(Qwen3 backbone) 모델은 `from_pretrained()` 사용 금지 — `_load_step3vl_manual()` 사용.
> transformers 업데이트 시 Qwen3Model full-size init hang 해결 여부 재확인.
> 모델 로딩 코드를 공통 유틸 함수로 추출 완료 (`_load_step3vl_manual()`).
> Step3-VL은 `attn_implementation="sdpa"` 비호환 — custom attention 구현 사용.

---

### B033: autonomous_gym.yaml (8-agent) + per-GPU config 동시 실행 → 리소스 충돌 → FIXED (03-16)
**심각도**: CRITICAL (GPU 1 OOM 위험, EHR baseline 불안정)

**증상**: GPU 1 메모리 81092/81920 MiB (99.0%). GPU 0-3에서 EHR baseline + GYM worker 이중 실행.

**근본 원인**: `autonomous_gym.yaml` (8-agent, 8 GPU) 프로세스(PID 2510494)가 per-GPU config 프로세스와 동시에 실행됨.
메인 GYM이 GPU 0-7에 worker를 spawn하면서 EHR baseline(GPU 0-3), per-GPU GYM(GPU 5/6/7)과 충돌.

**수정**: 메인 GYM 프로세스 및 모든 orphaned worker 종료. per-GPU config만 유지.

**재발 방지**:
> `autonomous_gym.yaml` (전체 GPU)과 per-GPU config는 절대 동시 실행 금지.
> 프로세스 시작 전 `ps aux | grep run_autonomous_gym`으로 기존 프로세스 확인 필수.
> 8-agent 풀 GYM은 Phase D (30-day run)에서만 사용.

---

### B034: `_load_step3vl_manual()` import 누락 → NameError → Step3 TextQA/MedLFQA 실행 불가 → FIXED (03-17)
**심각도**: HIGH (Step3-VL baseline 평가 차단)

**증상**: Step3-VL TextQA 및 MedLFQA 실행 시 `NameError: name 'AutoConfig' is not defined` 즉시 크래시.

**근본 원인**: B032 최종 수정 시 `_load_step3vl_manual()` 함수(line 91-134)에 `AutoConfig`, `AutoModelForCausalLM`, `torch`를 사용하지만 함수 내 import를 누락. 이 함수를 호출하는 `evaluate_medlfqa()`와 `evaluate_textqa()`는 각자 함수 로컬에서 import하지만, 이는 `_load_step3vl_manual` scope에서 접근 불가.

**수정**: 함수 내에 `import torch`, `from transformers import AutoConfig, AutoModelForCausalLM` 추가.

**재발 방지**:
> 공통 유틸 함수 추출 시 해당 함수 내에서 필요한 모든 import를 명시적으로 포함할 것.
> 단순 syntax import test: `python -c "from module import func"` 로는 runtime import 누락 발견 불가 — 실제 호출 테스트 필수.

---

### B035: Domain herding — weakness threshold + 차별화 부재 + recency penalty 없음 → 85% 단일 도메인 반복 → FIXED (03-17)
**심각도**: HIGH (GYM 자율 학습 루프의 도메인 다양성 완전 상실 — 논문 실험 유효성 저하)

**증상**:
- Dryrun (GPU 6): lease_0001~0040 중 85%가 drug_interaction
- GPU5 Lingshu: triage_emergency 연속 10+ cycle
- GPU7 Qwen: triage_emergency 연속 7+ cycle
- `HERDING WARNING` 로그 지속 출력되지만 실제 행동 변화 없음

**근본 원인 (3가지 동시)**:
1. **절대적 weakness threshold**: `shared_logbook.py:276` — `threshold = 0.1` 고정. 모든 도메인 점수가 0.30~0.45 범위일 때, avg - 0.1 = 0.31~0.35로 대부분 도메인이 weakness로 분류 안됨 → weakness 리스트 빈 상태
2. **동일 weakness 점수 fallback**: `autonomous_agent.py:441` — weakness 리스트에 없으면 `my_score < 0.5` → `weakness = 0.8` 일괄 적용. 모든 도메인이 score < 0.5이면 차별화 불가
3. **recency penalty 없음**: 최근 연속 선택된 도메인에 대한 페널티가 없어 한번 선택되면 계속 반복

**수정 (3가지)**:
1. `shared_logbook.py:276`: 절대 threshold 0.1 → 상대 threshold `max(avg * 0.15, 0.02)` 변경
2. `autonomous_agent.py:_score_domain()`: 연속 weakness 점수 → `0.5 + (avg - my_score) / avg` 비례 점수로 교체
3. `autonomous_agent.py:_score_domain()`: 최근 5회 중 동일 도메인 3회 이상 → 0.3배 penalty, 2회 → 0.6배 penalty 추가

**GPU 5/6/7 프로세스 재시작**: PID 2759172/2759173/2759174 (v5)

**재발 방지**:
> Threshold 설정은 항상 **상대적** (비율 기반)으로 — 절대값은 점수 분포 변화에 취약.
> StrategySelector에 도메인 선택 분포를 로그로 출력하여 herding 조기 감지.
> `detect_herding()` 결과를 경고만이 아닌 domain selection에 직접 반영할 것.

### B036: Step3-VL TextQA/MedLFQA BATCH_SIZE=8 → 4D tensor crash in apply_rotary_pos_emb → FIXED (03-18)
**심각도**: HIGH (Step3-VL TextQA/MedLFQA baseline 평가 완전 차단 — GPU 4 죽은 상태)

**증상**: Step3-VL TextQA 및 MedLFQA 실행 시 `RuntimeError: The size of tensor a (32) must match the size of tensor b (370) at non-singleton dimension 3` 크래시.
모델 로딩은 성공하지만 첫 번째 batch 추론에서 `apply_rotary_pos_emb()`에서 실패.

**근본 원인**: `StepRoboticsModel.get_input_embeddings()` (modeling_step_vl.py line 199-221)이 batch_size=1 전용으로 설계됨.
`input_ids.squeeze(0)` + `inputs_embeds.unsqueeze(0)` 패턴이 batch_size>1일 때 4D tensor를 생성.
- batch_size=8, seq_len=370 → squeeze(0) 효과 없음 → embed_tokens → (8, 370, 4096) → unsqueeze(0) → **(1, 8, 370, 4096)** 4D
- Qwen3Model이 3D tensor를 기대하므로 attention reshape에서 dim mismatch
- q 텐서 dim3=32 (num_heads가 head_dim 위치에), cos dim3=370 (seq_len이 head_dim 위치에) → 크래시

`run_full_benchmark_suite.py`의 TextQA (line 499)와 MedLFQA (line 257) 모두 `BATCH_SIZE=8` 사용.

**수정**: `BATCH_SIZE = 1 if is_step3 else 8` — Step3-VL 모델일 때만 batch_size=1 적용 (두 곳 모두).

**재발 방지**:
> Step3-VL 모델의 get_input_embeddings()는 squeeze(0)/unsqueeze(0)로 batch_size=1 전용.
> Step3-VL 사용하는 모든 eval 코드에서 BATCH_SIZE=1 강제.
> 새 VL 모델 추가 시 get_input_embeddings()의 batch 처리 방식 반드시 확인.

### B037: Step3-VL TextQA `max_new_tokens=32` + `<think>` generation prompt → below-random MedMCQA/MMLU → FIXED (03-18)
**심각도**: HIGH (Step3-VL TextQA 결과 전체 무효 — MedMCQA 21% < random 25%, MMLU 31.5%)

**증상**: Step3-VL TextQA에서 MedQA=51.5% (합리적)이지만 MedMCQA=21.0% (random 25% 이하), MMLU=31.5% (타 모델 65~87% 대비 극히 낮음).

**근본 원인**: Step3 chat template의 `add_generation_prompt=True`가 `<think>\n`을 generation prompt 끝에 삽입.
`max_new_tokens=32`로는 thinking이 끝나기 전에 잘림 → `</think>` 없이 truncate → think-strip regex 미매칭.
첫 번째 A/B/C/D 문자가 thinking text에서 추출되어 잘못된 답 반환.
MedQA는 thinking이 짧거나 우연히 맞는 답이 나와 51.5% 기록.

**수정**:
1. Step3에서 `<think>\n`을 prompt 끝에서 제거 (`text[:-len("<think>\n")]`) — thinking 비활성화
2. `max_new_tokens`를 32→64로 증가 (안전 마진, Step3만)
3. MedLFQA에도 동일한 thinking 비활성화 적용 (max_new_tokens=512는 유지)

**재발 방지**:
> Step3-VL(Qwen3 계열)의 chat template은 `<think>` 태그를 자동 삽입함.
> MC 답변 등 짧은 생성이 필요한 경우 반드시 `<think>\n`을 제거하거나 max_new_tokens를 충분히 늘릴 것.
> 새 reasoning 모델 추가 시 chat template의 generation prompt 확인 필수.

---

### B038: `model_profile.py:load_model()` — Step3 key_mapping + manual loading 누락 → EHR eval 가중치 랜덤 초기화 → FIXED (03-18)
**심각도**: CRITICAL (Step3-VL EHR baseline 결과 전체 무효 — action_score=0.000)

**증상**: `agent_runner.py`가 `profile.load_model()`을 통해 Step3 로딩 시 `Some weights were not initialized` 경고 후 action_score=0.000.

**근본 원인**: `model_profile.py:load_model()`이 `from_pretrained()` 직접 호출.
Step3-VL은 두 가지 이유로 `from_pretrained()` 사용 불가:
1. B032: Qwen3Model `__init__` hang (transformers 4.57.x)
2. B024: `_checkpoint_conversion_mapping` 미자동 적용 → key mismatch → random weights
VQA/TextQA/MedLFQA는 `run_full_benchmark_suite.py`의 `_load_step3vl_manual()` 사용하지만,
EHR eval은 `agent_runner.py` → `model_profile.py:load_model()` 경로를 타므로 수정 누락.

**수정**: `model_profile.py:load_model()`에 `model_type == "step_robotics"` 분기 추가.
`_load_step3vl_manual()` 메서드를 `ModelProfile` 클래스 내에 구현 (key_mapping + no_init_weights + safetensors 수동 로딩).
`device_map="cpu"` 호출 시 `cpu_only=True` 파라미터로 GPU 이동 방지 (LoRA merge 경로).

**재발 방지**:
> Step3-VL 관련 수정 시 모든 모델 로딩 경로 확인: `run_full_benchmark_suite.py`, `vqa_benchmark_eval.py`, `agent_runner.py` → `model_profile.py`.
> `model_profile.py:load_model()`은 모든 eval 경로의 공통 진입점 — 여기에 모델별 workaround 반드시 적용.

### B039: Step3-VL TextQA B037 regression — `<think>\n` 미제거 + max_new_tokens=512 → ~10% 정확도 → FIXED (03-20)
**심각도**: CRITICAL (Step3-VL TextQA 결과 전체 무효 — MedQA 51%→10%, MedMCQA 21%→10%, MMLU 31%→10%)

**증상**: B037 수정 후 재실행한 03-19 TextQA 결과가 모든 벤치마크에서 ~10%로 붕괴.
Random 25% (4지선다)보다 낮음.

**근본 원인**: B037 fix가 잘못 적용됨 (3가지 동시 원인):
1. `add_generation_prompt=True`가 Step3 chat template에서 `<think>\n`을 prompt 끝에 삽입 — 이를 strip하는 코드가 누락됨
2. `gen_max_tokens`를 계획(64)과 달리 512로 설정 — 512 토큰 thinking 생성
3. `outputs[j][input_len:]`은 생성 부분만 추출하므로 opening `<think>` 태그가 없음 → `re.sub(r"<think>.*?</think>", ...)` regex 불매칭 → thinking text 전체가 answer extraction 대상
4. `for ch in response: if ch in "ABCD"` — thinking text의 첫 A/B/C/D를 답으로 잘못 추출 → ~10% (random 이하)

**수정** (`scripts/run_full_benchmark_suite.py`):
1. TextQA line 525-527: `<think>\n` prompt strip 추가
2. TextQA line 540: `gen_max_tokens = 64 if is_step3 else 32`
3. TextQA line 550-552: `</think>` fallback strip 추가
4. MedLFQA line 278-280: 동일 `<think>\n` strip
5. MedLFQA line 303-305: 동일 `</think>` fallback

**재발 방지**:
> Step3-VL(Qwen3 계열) `add_generation_prompt=True`는 항상 `<think>\n`을 삽입함.
> MC QA 등 짧은 답변 시 반드시 prompt에서 `<think>\n` strip할 것.
> 생성 텍스트의 `</think>` fallback strip을 항상 적용 (regex만으로 불충분 — opening tag가 prompt 측에 있을 수 있음).
> RL fix 시 max_new_tokens 변경은 계획 값과 정확히 일치시킬 것 (512 vs 64 같은 임의 변경 금지).

---

### B040: PMC-VQA HuggingFace train_2.csv 스키마 불일치 → dataset generation 에러 → FIXED (03-20)
**심각도**: MEDIUM (Step3 VQA 6개 중 1개 결과 없음)

**증상**: `load_dataset("RadGenome/PMC-VQA")` 호출 시 `All the data files must have the same columns` 에러.

**근본 원인**: HF dataset의 `train_2.csv`가 `train.csv`와 다른 스키마 사용.
- `train_2.csv`에 `{'index', 'split', 'Caption'}` 추가 컬럼, `{'Answer_label'}` 누락
- `load_dataset`이 모든 CSV를 합치려다 실패

**수정**: `bioagents/data_pipeline/vqa_loader.py` line 370에 `data_files` 파라미터 추가:
```python
ds = load_dataset("RadGenome/PMC-VQA", split=split,
                  data_files={"test": "test.csv", "train": "train.csv"})
```

**재발 방지**:
> HuggingFace dataset 로딩 시 multi-file dataset는 스키마 일관성 확인.
> 불일치 시 `data_files` 파라미터로 특정 파일만 로드.

---

### B041: GRPO merge가 Qwen2.5-VL 모델 완전 파괴 + Lingshu catastrophic forgetting → CRITICAL (03-20)
**심각도**: CRITICAL (논문 핵심 claim 무력화)

**증상**: GYM 학습 후 외부 벤치마크(TextQA) 평가 결과:

| 모델 | Benchmark | Baseline | GYM-trained | Delta |
|------|-----------|----------|-------------|-------|
| **Lingshu** | MedQA | 61.7% | 56.9% | **-4.8%** |
| **Lingshu** | MedMCQA | 53.2% | 60.8% | **+7.6%** ✅ |
| **Lingshu** | MMLU(6) | 76.9% | 66.8% | **-10.1%** |
| **Qwen2.5-VL** | MedQA | 54.0% | 10.5% | **-43.5%** 💀 |
| **Qwen2.5-VL (early ckpt)** | MedQA | 54.0% | 12.5% | **-41.5%** 💀 |

**근본 원인** (추정):
1. **Qwen 모델 파괴**: GRPO LoRA merge 과정에서 모델 가중치가 손상됨. Early checkpoint(첫 GRPO merge)부터 이미 10% → merge 프로세스 자체의 문제.
   - 가능한 원인: LoRA rank 불일치, merge alpha 설정 오류, VL model의 vision encoder까지 LoRA 적용
2. **Lingshu catastrophic forgetting**: MedMCQA는 +7.6% 향상되었지만, MedQA(-4.8%)와 MMLU(-10.1%)에서 대폭 하락.
   - GYM internal task가 특정 도메인(clinical_diagnosis, drug_interaction)에 편향
   - KL penalty가 forgetting을 충분히 방지하지 못함

**영향**:
- 논문 "Autonomous self-evolving loop" contribution의 실증적 근거 약화
- 현재 GYM 학습은 외부 벤치마크 기준으로 net negative (Qwen) 또는 mixed (Lingshu)

**수정 방향** (미수정):
1. GRPO merge 코드 디버깅 — Qwen LoRA target_modules 확인, vision encoder 제외 여부
2. KL penalty 강화 또는 EWC(Elastic Weight Consolidation) 추가
3. GYM internal eval에 외부 벤치마크 subset을 validation set으로 추가 (anti-contamination 유지)
4. 학습 빈도 감소 (매 cycle → 매 5 cycles)

**재발 방지**:
> GYM 학습 후 반드시 외부 벤치마크로 검증 (internal eval만으로 판단 금지).
> LoRA merge 후 간단한 sanity check (MedQA 100개) 실행.
> 모든 GRPO checkpoint에 대해 merge 전후 weight norm 비교.

---

## Prevention Checklist

Claude가 코드를 작성/수정할 때 반드시 확인하는 체크리스트.

### Data Contamination 방지
- [ ] `*_test.jsonl`, `*_test.json` 파일에서 학습/GYM 데이터 생성 금지
- [ ] `split_tasks.json`에서 반드시 `split="train"` 사용
- [ ] 생성 후 `scripts/check_contamination.py` 실행
- [ ] Trajectory 추출 시 test task ID 필터링
- [ ] `data/medical_qa_200/` 절대 사용 금지 (DEPRECATED/오염)
- [ ] 새 벤치마크 추가 시 train/test 분리 문서화

### Model Loading 방지
- [ ] VL 모델 감지: `"qwen2" in model_type.lower() and "vl" in model_type.lower()` (substring)
- [ ] VL 모델 로딩: `AutoModelForVision2Seq` (CausalLM 금지)
- [ ] model_type 비교: substring 매칭만 사용 (정확 매칭 리스트 금지)
- [ ] 10B+ 모델: `gpu_memory_utilization: 0.70` 이하, `inference_batch_size: 1`
- [ ] **Custom 모델 로딩 시 "weights not initialized" 경고 반드시 확인** (B024)
- [ ] **`_checkpoint_conversion_mapping` 자동 미적용 모델: `key_mapping` 명시적 전달** (B024)
- [ ] **Reasoning 모델 (`<think>` 태그): 출력 후처리에서 `<think>...</think>` 제거** (B025)

### Model Loading 방지 (추가)
- [ ] **Step3-VL key_mapping/think-strip은 모든 eval 스크립트에 적용** (B032 — `grep -r "from_pretrained" scripts/ bioagents/evaluation/`)
- [ ] **모델 로딩 수정 시 vqa_benchmark_eval.py + run_full_benchmark_suite.py + 기타 모두 확인**
- [ ] **Step3-VL(Qwen3 backbone): `from_pretrained()` 사용 금지 → `_load_step3vl_manual()` 사용** (B032 final — transformers 4.57.x Qwen3 init hang)
- [ ] **Step3-VL 추론 시 BATCH_SIZE=1 강제** (B036 — get_input_embeddings squeeze(0)/unsqueeze(0)가 batch>1에서 4D tensor 생성)
- [ ] **Step3-VL MC 평가 시 `<think>` generation prompt 반드시 strip** (B039 — `add_generation_prompt=True`가 자동 삽입, strip 안하면 thinking text에서 잘못된 답 추출)
- [ ] **Step3-VL 생성 후 `</think>` fallback strip 적용** (B039 — `<think>`가 prompt 측에 있으면 regex 불매칭)
- [ ] **model_profile.py:load_model() 경로도 Step3 manual loading 적용 확인** (B038 — EHR eval 경로)

### Autonomous GYM 방지
- [ ] `RunConfig` 생성 전 `task_ids` 값 확인 (None = 전체 실행)
- [ ] `eval_tasks_per_domain` 값이 실제로 `RunConfig(task_ids=...)` 에 전달되는지 확인
- [ ] 모든 subprocess timeout은 YAML config로 노출
- [ ] Dryrun 재시작 후 첫 사이클이 설정된 task 수만큼만 실행되는지 로그 확인
- [ ] **`_build_cycle_script()` 필드 동기화**: `AutonomousAgentConfig` 새 필드 추가 시 반드시 포함 (B030)
- [ ] **`eval_timeout_seconds` = 예상 사이클 소요 시간 × 1.5 이상** (B031)
- [ ] **autonomous_gym.yaml(전체 GPU)과 per-GPU config 절대 동시 실행 금지** (B033)
- [ ] **새 gym config: `adaptive_quality: true` 기본 설정, 고정 threshold만 사용 금지** (B026)
- [ ] **새 gym config 작성 시 `eval_timeout_seconds`, `train_timeout_seconds` 반드시 명시** (B021)

### GRPO/RL Training 방지
- [ ] `_grpo_policy_update()`에 `ref_log_probs` 전달 여부 확인 (None이면 KL 없음!)
- [ ] `beta * loss`는 KL penalty가 아님 — 반드시 `log π_θ - log π_ref` 사용
- [ ] Pre/post eval에 동일 `eval_seed` 전달 여부 확인
- [ ] `max_train_tasks > 0`으로 task cap 설정 (0=전체 학습 → overfitting 위험)
- [ ] Reward weights에서 accuracy ≥ 50% 확인
- [ ] `training_epochs ≤ 1` 확인 (LoRA + 소량 데이터에서 2+ epoch은 overfitting)
- [ ] `learning_rate ≤ 1e-5` 확인 (KL penalty 있어도 높은 LR은 drift 유발)
- [ ] 첫 사이클 improvement 부호 반드시 확인 (음수면 즉시 중단)

### Dataclass / Config 방지
- [ ] 새 필드 추가 시 **base class**에 추가 (child class에만 추가하면 base 생성 시 TypeError)
- [ ] `**kwargs` 또는 `**_common_kwargs`로 전달되는 모든 키가 target dataclass에 존재하는지 확인
- [ ] Subprocess stderr는 항상 로깅 (returncode 무관)

### EHR Evaluation 방지
- [ ] `max_samples` 반드시 설정 (0은 전체 57K 실행 — 금지)
- [ ] EHR eval 시작 전 예상 완료 시간 계산: `tasks × 160s ÷ 3600 = hours`
- [ ] checkpoint 경로가 쓰기 가능한지 확인
- [ ] dataclass 직렬화: `dataclasses.asdict()` 사용

### GYM Environment 방지
- [ ] 환경 ID: `"BioAgent-v0"` 상수 사용
- [ ] `env.reset()` 반환: `(str, dict)` — obs는 str
- [ ] Python 환경: `.venv/bin/python` (conda base 금지)

---

## History

| 날짜 | 작업 | 수정 Bug |
|------|------|---------|
| 2026-02-12 | 초기 GPU 가동 | B009, B010, B011, B012 |
| 2026-02-12 | Data contamination 감사 | B001, B002 |
| 2026-02-13 | Contamination 완전 해결 + 모델 로딩 수정 | B003, B004, B005, B006, B007, B008, B013 |
| 2026-03-11 | Dryrun 타임아웃 + EHR 최적화 + Step3 OOM 수정 | B014, B015, B016, B017, B018 |
| 2026-03-12 | Step3 VL model_type 수정 + gym timeout 설정 + WORKER_SCRIPT 이스케이프 | B019, B020(WF), B021, B022 |
| 2026-03-12 | EHR stratified sampling dict unhashable 수정 | B023 |
| 2026-03-12 | Step3-VL 가중치 매핑 + think-strip 수정 | B024, B025 |
| 2026-03-12 | Adaptive quality filtering 구현 (quality_threshold 고정값 대체) | B026 |
| 2026-03-14 | GRPO degradation 근본 수정 — KL penalty, task cap, LR 감소, reward rebalance 등 | B027 |
| 2026-03-16 | max_train_tasks 필드 누락 수정 + subprocess stderr 로깅 추가 | B028, B029 |
| 2026-03-16 | worker config 필드 누락 + dryrun timeout + Step3 TextQA/MedLFQA key_mapping + 8-agent 충돌 | B030, B031, B032, B033 |
| 2026-03-17 | Step3 import 누락 + domain herding 근본 수정 (threshold + recency penalty) | B034, B035 |
| 2026-03-18 | Step3-VL TextQA/MedLFQA batch size 크래시 수정 | B036 |
| 2026-03-18 | Step3 TextQA thinking 비활성화 + model_profile Step3 manual loading | B037, B038 |
| 2026-03-20 | Step3 TextQA B037 regression 수정 + PMC-VQA 로딩 수정 | B039, B040 |
| 2026-03-20 | GYM External Validation → Qwen GRPO merge 모델 파괴 발견 + Lingshu 부분 향상 확인 | B041 |
