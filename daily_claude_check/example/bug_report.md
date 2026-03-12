# Claude Bug Report - 재발 방지 기록

> Bug 발생 → 원인 분석 → 해결 → 재발 방지 패턴 정리.
> 같은 실수를 두 번 반복하지 않기 위한 참조 문서.

---

## Bug Index

| ID | 카테고리 | 요약 | 발생일 | 재발 횟수 |
|----|---------|------|--------|----------|
| B001 | Config | pt-eval config 경로 포맷 오류 | 2026-03-11 | 4회 (1차~4차 제출 모두) |
| B002 | Environment | vLLM reasoning parser 미등록 | 2026-03-11 | 2회 |
| B003 | Environment | `--trust-remote-code` 누락 | 2026-03-11 | 2회 |
| B004 | Dependency | `pyjson5` 미설치 | 2026-03-11 | 1회 |
| B005 | Environment | `wandb` ModuleNotFoundError | 2026-03-11 | 1회 |
| B006 | Slurm | `SLURM_JOB_ID` 환경변수 오염 | 2026-03-11 | 1회 |
| B007 | Docker | Docker 이미지 미존재/unauthorized | 2026-03-11 | 1회 |
| B008 | Serve | `serve_type` 자동 감지 오류 | 2026-03-11 | 잠재적 |

---

## B001: pt-eval config 경로 포맷 오류

**심각도**: CRITICAL (모든 커스텀 옵션이 무시됨)

**증상**: config에 설정한 `trust_remote_code`, `tool_call_parser`, `reasoning_parser`, `chat_template` 등이 전혀 적용되지 않고 default 값으로 실행됨.

**원인**:
- `run_slurm.py`에서 `--config config/models/solar_open_sft.py`로 전달
- `run.py`의 `load_configs()`가 `os.path.join('config', f'{config_path}.py')`로 경로 조합
- 결과: `config/config/models/solar_open_sft.py.py` → 파일 없음 → "Config file not found. Skipping..." → 조용히 무시

**해결**: `--config models/solar_open_sft` (prefix `config/` 제거, 확장자 `.py` 제거)

**재발 방지 규칙**:
> pt-eval `--config` 인자는 반드시 `config/` 디렉토리 기준 상대경로, 확장자 없이 전달한다.
> 예: `--config models/solar_open_sft` (O) / `--config config/models/solar_open_sft.py` (X)

---

## B002: vLLM reasoning parser 미등록 (`solar_wbl`)

**심각도**: HIGH

**증상**: `vllm serve: error: argument --reasoning-parser: invalid choice: 'solar_wbl'`

**원인**:
- vLLM 0.11.1: `--reasoning-parser`의 valid choices가 `_REASONING_PARSERS_TO_REGISTER` dict에서 정적으로 결정됨
- `--reasoning-parser-plugin`으로 plugin을 로드해도, argument validation이 plugin 로드보다 먼저 실행됨
- vLLM 0.9.0.1: `__init__.py`에 직접 import 하는 방식 → import 없으면 미등록

**해결**:
- vLLM 0.11.1 (`pt-eval.browsecomp_plus` conda): `__init__.py`의 `_REASONING_PARSERS_TO_REGISTER`에 `solar_wbl` 엔트리 추가 + parser 파일 복사
- vLLM 0.9.0.1 (`browsecomp_plus/.venv`): `__init__.py`에 `from .solarwbl_reasoning_parser import SolarWBLReasoningParser` 추가 + **해당 버전 API에 맞게 parser 재작성** (base class가 다름: `ReasoningParser` vs `BaseThinkingReasoningParser`)

**재발 방지 규칙**:
> custom parser를 사용할 때는 반드시 해당 vLLM의 `reasoning/__init__.py`에 등록한다. plugin 방식만으로는 argument validation을 통과하지 못한다.
> vLLM 버전별로 base class와 method signature가 다르므로 반드시 해당 버전의 기존 parser (예: `deepseek_r1_reasoning_parser.py`)를 참고하여 작성한다.

---

## B003: `--trust-remote-code` 누락

**심각도**: HIGH

**증상**: `ValueError: model type 'solar_open' but Transformers does not recognize this architecture`

**원인**:
- `solar_open`은 HuggingFace transformers에 등록되지 않은 커스텀 아키텍처
- `--trust-remote-code` 없이는 checkpoint 디렉토리의 custom `configuration_*.py`를 로드하지 않음
- `serve_vllm_0_11_1.py`에 해당 옵션 지원이 없었음

**해결**:
- `serve_vllm_0_11_1.py`에 `trust_remote_code` 옵션 지원 코드 추가
- `solar_open_sft.py` config에 `trust_remote_code: True` 설정

**재발 방지 규칙**:
> Solar Open 계열 모델 서빙 시 `trust_remote_code: True`는 필수. 다른 커스텀 아키텍처 모델도 동일.

---

## B004: `pyjson5` 미설치

**심각도**: MEDIUM

**증상**: `ModuleNotFoundError: No module named 'pyjson5'` → tool parser plugin 로드 실패 → `invalid tool call parser: solar_wbl`

**원인**:
- `solarwbl_tool_parser.py`가 `import pyjson5`를 사용
- `pt-eval.browsecomp_plus` conda env에 `pyjson5`가 설치되어 있지 않았음

**해결**: `pip install pyjson5` (in `pt-eval.browsecomp_plus`)

**재발 방지 규칙**:
> 새 plugin 파일을 사용할 때는 해당 파일의 import를 전부 확인하고 대상 환경에 dependency가 설치되어 있는지 체크한다.

---

## B005: `wandb` ModuleNotFoundError

**심각도**: MEDIUM

**증상**: Slurm job이 시작 즉시 `ModuleNotFoundError: No module named 'wandb'`로 실패

**원인**:
- `run_slurm.py`가 생성하는 Slurm 스크립트에서 conda env를 활성화하지 않아 base env의 Python이 사용됨

**해결**: `run_slurm.py`의 Slurm 스크립트 템플릿에 `source conda.sh && conda activate pt-eval` 추가

**재발 방지 규칙**:
> Slurm 스크립트에서 Python을 실행할 때는 반드시 올바른 conda env가 활성화되어 있는지 확인한다.

---

## B006: `SLURM_JOB_ID` 환경변수 오염

**심각도**: MEDIUM

**증상**: wbl-eval `run.sh`가 새 sbatch를 제출하지 않고 현재 노드에서 직접 실행됨 (기존 Slurm job 안에서 실행된 것으로 오판)

**원인**:
- 현재 shell 세션이 이미 Slurm allocation (interactive job) 안에서 실행 중
- `SLURM_JOB_ID`가 설정되어 있어 `run.sh`가 `slurm-job` 환경으로 감지 → sbatch 대신 직접 실행

**해결**: `env -u SLURM_JOB_ID -u SLURM_NODELIST -u SLURM_JOB_NODELIST -u SLURM_NNODES bash run.sh ...`

**재발 방지 규칙**:
> wbl-eval `run.sh`를 login shell에서 실행할 때, interactive job 안이라면 반드시 `env -u SLURM_JOB_ID ...`로 Slurm 환경변수를 해제한다.

---

## B007: Docker 이미지 미존재/unauthorized

**심각도**: LOW (native mode로 우회)

**증상**: `Unauthorized` (ghcr.io pull) 또는 `no such file or directory` (local image path)

**원인**:
- `ghcr.io/upstageai/solar-open-vllm:solar-open-1.0.0.dev10` 이미지에 인증 필요
- `/data/project/private/christopher/resources/solar_open_docker/` 경로가 compute node에서 접근 불가

**해결**: Docker mode 대신 native mode로 전환 (vLLM을 conda env에서 직접 실행)

**재발 방지 규칙**:
> `omni-preemptible` 노드에서 Solar Open Docker 이미지는 사용 불가. 항상 native mode를 사용한다.

---

## B008: `serve_type` 자동 감지 오류 (잠재적)

**심각도**: LOW (config에서 명시하면 회피 가능)

**증상**: model path에 "solar-open"이 포함되면 `solar_docker`로 감지되어 Docker 서빙 시도

**원인**:
- `_detect_serve_type()`이 `model_path.lower()`에서 "solar-open" 문자열 매칭
- `/mnt/weka/.../Solar-Open-100B` → lowercase 시 `solar-open` 매칭 → `solar_docker`

**해결**: config에 `serve_type: "vllm_0_11_1"` 명시적 지정 (현재 `solar_open_sft.py`에는 미설정이지만, `_global` options에 포함시키면 됨)

**재발 방지 규칙**:
> Solar Open 모델을 native mode로 서빙할 때는 config에 `serve_type` 을 명시한다. 자동 감지에 의존하지 않는다.
