# BIOAgents - Claude Bug Report & 재발 방지 기록

> Bug 발생 → 원인 분석 → 해결 → 재발 방지 패턴 정리.
> 같은 실수를 두 번 반복하지 않기 위한 참조 문서.
> 상세 원본 기록: `BUGLOG.md`

---

## Last Updated: 2026-03-12 21:00 KST

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

### Autonomous GYM 방지
- [ ] `RunConfig` 생성 전 `task_ids` 값 확인 (None = 전체 실행)
- [ ] `eval_tasks_per_domain` 값이 실제로 `RunConfig(task_ids=...)` 에 전달되는지 확인
- [ ] 모든 subprocess timeout은 YAML config로 노출
- [ ] Dryrun 재시작 후 첫 사이클이 설정된 task 수만큼만 실행되는지 로그 확인
- [ ] **새 gym config: `adaptive_quality: true` 기본 설정, 고정 threshold만 사용 금지** (B026)
- [ ] **새 gym config 작성 시 `eval_timeout_seconds`, `train_timeout_seconds` 반드시 명시** (B021)

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
