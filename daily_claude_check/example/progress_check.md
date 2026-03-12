# Daily Claude Check - Detailed Job Tracking & Progress

> `progress_summary_and_planning.md`와 연동되는 상세 기록.
> Slurm job ID, 실행 환경, 실패/해결 이력, 구체적 수치를 추적합니다.
> Claude가 context를 빠르게 파악하기 위한 reference 문서입니다.

---

## Last Updated: 2026-03-11 17:50 KST

---

## [SFT] Solar Open 102B SFT 학습 & 평가

### SFT Training Detail

| 항목 | 값 |
|------|-----|
| Base Model | Solar Open 100B (`solar_wbl` / `100B_A12B_SFT`) |
| Training Framework | `torchtitan_sft` |
| Config | `minstar_trajectory_sft_v1.toml` |
| Initial Checkpoint | `/home/minstar/checkpoints/Solar-Open-100B-DCP` |
| Dump Folder | `/mnt/weka/post_training/checkpoints/minstar_trajectory_sft_v1` |
| Optimizer | AdamW (lr=1e-6, β1=0.9, β2=0.999, ε=1e-8, wd=0.0) |
| LR Scheduler | warmup 50 steps |
| Sequence Length | 65536 |
| Epochs | 3 |
| Sample Packing | Enabled |
| Precision | Float8 (FSDP float8 all-gather) |
| Activation Checkpoint | Full |
| Parallelism | FSDP (shard=-1), EP=8 |
| Checkpoint Strategy | epoch-based (interval=1 epoch, also every 500 steps) |

**Dataset (22개, ~106K samples)**:
- Open-source 19개 (max 10K random sample/dataset, license-clean)
  - MCP류: toucan-1.5m (719K), nemotron-agentic (219K), toolbench (187K), glaive-fc (113K), tool-calling-mix (34K), smolagents (33K), when2call (15K), agent-flan (12K), hermes-fc (1.9K), owl-sft (1.6K)
  - Web/Search류: browseragent (44K), afm-webagent (7.6K), eto-traj (4.9K), react-llama (3.5K), deepresearch-sft (2.5K), agenttuning (1.9K), mind2web (1K), simple-deep-searcher (871), deepdive-traj (858)
- Synthesized 3개 (all pass + judge_correct):
  - `asearcher_traj_wiki2018_fromqa_think`
  - `synth_wiki2026_all_pass`
  - `synth_wiki2026_judge_correct`

**Think Mode 분포**: `[lastthink=0.3, nothink=0.3, think=0.4]`

**저장된 Checkpoint Steps**: `step-200` (epoch 1 끝), `step-390` (epoch 2 끝), `step-590` (epoch 3 끝)

### DCP → HuggingFace 변환: ✅ 완료

| Source (DCP) | Target (HF) | Status |
|-------------|------------|--------|
| `.../checkpoint/step-200` | `.../checkpoint/step-200-hf/` | ✅ 완료 (42 safetensors + 6 JSON) |
| `.../checkpoint/step-390` | `.../checkpoint/step-390-hf/` | ✅ 완료 |
| `.../checkpoint/step-590` | `.../checkpoint/step-590-hf/` | ✅ 완료 |

변환 스크립트: `/home/minstar/workspace/torchtitan_sft/run_convert_all.sh`

---

### SFT 평가 Job 현황

#### A. pt-eval BrowseComp-Plus

**Motivation**: SFT 학습 step별 (epoch 1/2/3) 성능 변화 추적 및 base 모델 대비 browsecomp-plus 향상도 측정

| Job ID | Model | Benchmark | Status | Submitted | Notes |
|--------|-------|-----------|--------|-----------|-------|
| 54370 | step-200-hf (epoch 1) | browsecomp_plus | ⏳ PENDING | 03-11 17:41 | config 포맷 수정 후 재제출 |
| 54371 | step-390-hf (epoch 2) | browsecomp_plus | ⏳ PENDING | 03-11 17:41 | |
| 54372 | step-590-hf (epoch 3) | browsecomp_plus | ⏳ PENDING | 03-11 17:41 | |
| 54373 | Solar-Open-100B (base) | browsecomp_plus | ⏳ PENDING | 03-11 17:41 | 비교 baseline |

**Framework**: `pt-eval` / vLLM native serving
**Config**: `models/solar_open_sft` → `/home/minstar/workspace/pt-eval/config/models/solar_open_sft.py`
**Partition**: `omni-preemptible`, Nodelist: `Slurm-GPU-Node-[75-90]`

**Key Config Options**:
```
trust_remote_code: True
chat_template: /mnt/weka/post_training/checkpoints/Solar-Open-100B/chat_template.jinja
tool_call_parser: solar_wbl (plugin)
reasoning_parser: solar_wbl (native 등록)
enable_reasoning: True
serve_type: vllm_0_11_1
client_type: oss
temperature: 0.8, top_p: 0.95, max_tokens: 32768
```

**이전 실패 이력 (총 5회 재제출)**:

| 시도 | Job IDs | 원인 | 해결 |
|------|---------|------|------|
| 1차 | 54198-54200 | `wandb` ModuleNotFoundError | `run_slurm.py`에 conda activate 추가 |
| 2차 | 54220-54222 | `--reasoning-parser solar_wbl` invalid choice | vLLM reasoning 모듈에 직접 등록 |
| 3차 | 54231-54235 | reasoning parser 제거 후 재제출 (임시 우회) | - |
| 4차 | 54269-54272 | `model type solar_open not recognized` | `--trust-remote-code` 추가 + config 경로 포맷 수정 (`config/models/x.py` → `models/x`) |
| 5차 | 54370-54373 | (현재 PENDING) | config 포맷 최종 수정 후 제출 |

**수정한 파일들**:
- `/home/minstar/workspace/pt-eval/config/models/solar_open_sft.py`
- `/home/minstar/workspace/pt-eval/utils/serve_vllm_0_11_1.py` (`trust_remote_code` 옵션)
- `/home/minstar/workspace/pt-eval/run_slurm.py` (conda activate)
- vLLM reasoning parser (2곳):
  - `miniconda3/envs/pt-eval.browsecomp_plus/.../vllm/reasoning/` (0.11.1 - lazy register)
  - `framework/browsecomp_plus/submodule/.venv/.../vllm/reasoning/` (0.9.0.1 - direct import)

#### B. wbl-eval MCP-Atlas

**Motivation**: SFT 모델의 MCP server 활용 능력 평가. API key 없이 사용 가능한 20개 MCP server (전체 task의 18%, 89개 task) 기준 평가. (ref: GLM 4.7 reported=52.0, 재현=56.17)

| Job ID | Model | Benchmark | Status | Submitted | Notes |
|--------|-------|-----------|--------|-----------|-------|
| 54375 | step-200-hf (epoch 1) | custom/mcp_atlas | ⏳ PENDING | 03-11 17:41 | pyjson5 설치 후 재제출 |
| 54376 | step-390-hf (epoch 2) | custom/mcp_atlas | ⏳ PENDING | 03-11 17:41 | |
| 54377 | step-590-hf (epoch 3) | custom/mcp_atlas | ⏳ PENDING | 03-11 17:41 | |
| 54378 | Solar-Open-100B (base) | custom/mcp_atlas | ⏳ PENDING | 03-11 17:41 | 비교 baseline |

**Framework**: `wbl-eval` / native mode (NOT Docker for vLLM)
**Config 위치**: `/home/minstar/workspace/solar-system/eval/wbl-eval/scripts/user/configs/`
**Partition**: `omni-preemptible`, Nodelist: `Slurm-GPU-Node-[75-90]`
**Conda Env** (vLLM serving): `pt-eval.browsecomp_plus` (vLLM 0.11.1)

**주의사항**:
- MCP-Atlas 벤치마크 자체가 Docker 사용 (`ghcr.io/scaleapi/mcp-atlas:1.2.5`) → MCP 서버 컨테이너
- Docker 그룹 추가 완료 (`sudo usermod -aG docker minstar`)
- PR #88 (`eval/feat/add-mcp_atlas` 브랜치) 기반

**이전 실패 이력 (총 3회 재제출)**:

| 시도 | Job IDs | 원인 | 해결 |
|------|---------|------|------|
| 1차 | 54206-54209 | Docker 이미지 (`solar-open-vllm`) 미존재/unauthorized | native mode로 전환 |
| 2차 | 54258-54261 | `pyjson5` ModuleNotFoundError → tool parser 로드 실패 | pip install pyjson5 |
| 3차 | 54375-54378 | (현재 PENDING) | pyjson5 설치 + 재제출 |

---

## [Evaluation] 기타 평가 Job 현황

### BrowseComp Variants (deepresearch pipeline)

**Motivation**: BM25 setting, get_document (full content browsing) 등 다양한 평가 방식 중 최적 파악

| Job ID | Name | Status | Runtime | Node |
|--------|------|--------|---------|------|
| 54336 | browsecomp-variants | RUNNING | ~37m+ | Node-75 |

### Eval-Closedbook (Solar Open)

**Motivation**: 기본 closedbook 성능 평가

| Job ID | Status | Runtime | Node |
|--------|--------|---------|------|
| 54284 | RUNNING | ~2h | Node-88 |
| 54286 | RUNNING | ~1h50m | Node-76 |
| 54287 | RUNNING | ~1h42m | Node-89 |
| 54288 | RUNNING | ~1h05m | Node-82 |
| 54289 | RUNNING | ~1h05m | Node-87 |
| 54346 | RUNNING | ~20m | Node-85 |
| 53655 | RUNNING | ~5m | Node-75 |
| 53653, 53654, 54283 | PENDING | - | - |

### Eval-Toolbased

| Job ID | Status | Notes |
|--------|--------|-------|
| 53631 | PENDING (Priority) | |
| 53632 | PENDING (Resources) | |
| 53633 | PENDING (BeginTime) | |
| 53634 | PENDING (BeginTime) | |

---

## [Data Generation] 관련 Job 현황

### Data Generation Dense (CommonCrawl E5 Indexing)

**Motivation**: CommonCrawl (6T) retrieved corpus (e5-base-v2) 구축

| Job ID | Name | Status | Notes |
|--------|------|--------|-------|
| 53300, 53301 | data-generation-dense-omni | PENDING | |
| 53463, 53464 | data-generation-dense-omni | PENDING | |
| 53470-53472 | data-generation-dense-omni | COMPLETED | 각 ~27m |

### Retry-Fail (데이터 생성 재시도)

| Job ID | Status | Notes |
|--------|--------|-------|
| 53318, 53319 | PENDING (Priority) | |
| 53320 | PENDING (Nodes DOWN/DRAINED) | |
| 53454-53456 | PENDING (Priority) | |

### Focused Browse (Wikipedia 기반 합성)

| Job ID | Status | Notes |
|--------|--------|-------|
| 53606, 53607 | PENDING (Priority) | |

---

## 환경 정보 요약

| 항목 | 값 |
|------|-----|
| Slurm Partition | `omni-preemptible` |
| 사용 가능 Nodelist | `Slurm-GPU-Node-[75-90]` |
| pt-eval 경로 | `/home/minstar/workspace/pt-eval/` |
| wbl-eval 경로 | `/home/minstar/workspace/solar-system/eval/wbl-eval/` |
| torchtitan_sft 경로 | `/home/minstar/workspace/torchtitan_sft/` |
| SFT 체크포인트 Base | `/mnt/weka/post_training/checkpoints/minstar_trajectory_sft_v1/checkpoint/` |
| Base 모델 | `/mnt/weka/post_training/checkpoints/Solar-Open-100B` |
| Chat Template | `/mnt/weka/post_training/checkpoints/Solar-Open-100B/chat_template.jinja` |
| vLLM (pt-eval conda) | `pt-eval.browsecomp_plus` → vLLM 0.11.1 |
| vLLM (browsecomp .venv) | `framework/browsecomp_plus/submodule/.venv` → vLLM 0.9.0.1 |
| SFT Training Config | `torchtitan/models/solar_wbl/train_configs/minstar_trajectory_sft_v1.toml` |

---

## TODO / 확인 필요사항 (03-11 기준)

- [ ] **pt-eval** browsecomp-plus jobs (54370-54373) 성공 여부 확인 → config 옵션이 정상 적용되었는지 로그 확인 필수
- [ ] **wbl-eval** MCP-Atlas jobs (54375-54378) 성공 여부 확인 → Docker pull + vLLM serve 정상 동작 확인
- [ ] 평가 결과 수집: SFT step별 (ep1/ep2/ep3) 성능 vs base 모델 비교 분석
- [ ] browsecomp-variants (54336) 결과 확인 → 최적 평가 방식 결정
- [ ] eval-closedbook 결과 수집
- [ ] Tau2, BFCL 평가 세팅 (wbl-eval에 due 3.16)
- [ ] MCP-Atlas API key 추가 세팅 (추가 MCP server 활성화 시 due 3.20)
- [ ] GAIA, FRAMES wbl-eval migration (due 3.20)
