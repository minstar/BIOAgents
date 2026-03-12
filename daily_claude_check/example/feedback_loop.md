# Feedback Loop - 작업 재점검 & Self-Check Prompts

> 작업 완료 후 한 번씩 이 문서의 체크리스트를 돌면서 실수를 조기에 발견한다.
> Claude가 매 작업 세션 종료 시 참조해야 하는 문서.

---

## Self-Check Protocol

작업을 마무리하기 전에 아래 체크리스트를 순서대로 확인한다.

### 1. Slurm Job 제출 시

- [ ] `--config` 경로 포맷이 올바른가? (prefix `config/` 제거, `.py` 확장자 제거 → `models/solar_open_sft`)
- [ ] `--nodelist` 가 `Slurm-GPU-Node-[75-90]` 으로 제한되어 있는가?
- [ ] `--partition` 이 올바른가? (`omni-preemptible`)
- [ ] 현재 shell에 `SLURM_JOB_ID` 가 설정되어 있지 않은가? (wbl-eval 제출 시 `env -u SLURM_JOB_ID` 필수)
- [ ] 제출 후 `squeue`로 job이 큐에 정상 등록되었는지 확인했는가?

### 2. vLLM / 모델 서빙 관련

- [ ] `solar_open` 아키텍처 모델에 `--trust-remote-code` 가 포함되어 있는가?
- [ ] config 옵션이 실제 vLLM 명령어에 반영되는지 로그로 검증했는가? (첫 제출 시 반드시 로그 확인)
- [ ] 어떤 vLLM이 실행되는지 확인했는가? (conda env vLLM vs `.venv` vLLM — 버전과 설치된 plugin이 다름)
  - `pt-eval.browsecomp_plus` conda → vLLM 0.11.1
  - `browsecomp_plus/submodule/.venv` → vLLM 0.9.0.1
- [ ] custom parser (tool/reasoning)가 해당 vLLM 버전에 설치되어 있는가?
- [ ] 필요한 Python 패키지가 해당 환경에 설치되어 있는가? (예: `pyjson5`)

### 3. Config / 설정 파일 수정 시

- [ ] 수정한 config가 실제로 로드되는 경로인가? (`load_configs()` 동작 방식 재확인)
- [ ] `_detect_serve_type()`에 의해 의도치 않은 serve type으로 분기되지 않는가? (model path에 "solar-open" 포함 시 `solar_docker`로 감지됨 → `serve_type` 명시 필요)
- [ ] `_global` options가 benchmark-specific options에 정상 merge 되는 구조인가?

### 4. wbl-eval 제출 시

- [ ] native mode로 제출하는가? (Docker mode는 이미지 접근 불가로 실패함)
- [ ] `CONDA_ENV`과 `CONDA_BASE`가 config에 설정되어 있는가?
- [ ] `VLLM_EXTRA_ARGS`에 필요한 플래그가 전부 포함되어 있는가?
- [ ] Docker 그룹 권한이 compute node에 적용되어 있는가? (MCP-Atlas는 Docker 필요)

### 5. 결과 확인 시

- [ ] 결과 디렉토리에 실제 output이 생성되었는가?
- [ ] `planning.md`에 job 상태를 업데이트했는가?
- [ ] 실패한 경우 `claude_bug_report.md`에 기록했는가?

---

## 작업별 재점검 Prompts

아래는 특정 유형의 작업 완료 후 스스로에게 던지는 질문들이다.

### SFT 모델 평가 제출 후

```
1. 방금 제출한 job의 config가 정상 로드되는지 확인했는가?
   → `squeue`로 큐 등록 확인 + 이전 실패 패턴 반복 여부 점검
2. vLLM serve 명령에 --trust-remote-code, --chat-template, --tool-call-parser, --reasoning-parser 가 모두 포함되는가?
3. serve_type이 의도한 대로 vllm_0_11_1 인가? (solar_docker로 잘못 감지되지 않는가?)
4. browsecomp_plus 의 경우 .venv vLLM과 conda vLLM 중 어디서 실행되는가?
```

### 데이터 생성 Job 제출 후

```
1. 데이터 출력 경로가 올바르게 설정되어 있는가?
2. GPU 메모리 요구사항에 맞는 nodelist를 지정했는가?
3. 이전에 같은 job이 실패한 적이 있다면 원인이 해결되었는가?
```

### 환경 수정 (pip install, vLLM 패치 등) 후

```
1. 수정한 환경이 compute node에서도 접근 가능한 경로에 있는가? (shared filesystem)
2. 동일 패키지를 사용하는 다른 vLLM 환경에도 적용이 필요하지 않은가?
3. __pycache__ 때문에 이전 버전이 캐싱되어 있지 않은가?
```

---

## History

| 날짜 | 작업 | 재점검 결과 | 발견된 문제 |
|------|------|------------|------------|
| 2026-03-11 | pt-eval browsecomp-plus 제출 (5차) | config 포맷 수정 후 재제출 | 이전 4차까지 config가 미적용되고 있었음 (경로 포맷 오류) |
| 2026-03-11 | wbl-eval MCP-Atlas 제출 (3차) | pyjson5 설치 후 재제출 | tool parser plugin이 pyjson5 의존성 누락으로 로드 실패 |
