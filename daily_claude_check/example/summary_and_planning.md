전체적인 search & conversational agent 개발에 관련된 progress summary와 planning을 체크하는 용도

[Data]
* Progress Summary
[Open source QA & Trajectory 수집]
 - trajectory (1M) & QA (800k) 수집 완료 [/home/minstar/data_generation/opensource]
 - open source (license issue X)
    (1) MCP: trajectory (920k) & QA (170k)
    (2) Search: trajectory (65k) & QA (530k)

[인하우스 데이터 생성 (with Data team)]
 - 총 수량 5K 예상중 (2.5k 중간 납품, 추후 납품 예정)

* Planning
[⏳] Open source QA & trajectory 수집 (due 3.15)
[ ] trajectory & QA pass@8 evaluation (due 3.18)
[ ] 인하우스 데이터 유효성 검증 (due 3.21)

[Data Generation]
* Progress Summary
[Search agent]
  • 실제 Web기반 QA & Trajectory 합성
     - web search & browse API setting후 진행예정 
  • Wikipedia 기반 QA & Trajectory 합성
     - 5.8k web trajectory data 확보 (추가 확보 예정)
     - 9.8k QA data 확보 (추가 확보 예정)
  • CommonCrawl 기반 QA & Trajectory 합성
  • 현재 수량 / 최종 타겟 수량
     - [Target] Mid-training &  SFT [40B & 100k], RL [todo]

[Conversational agent]
  • Tau2 기반 in-domain 합성 data 
     - data statistics 수량 및 domain 파악 [todo]
  • MCP 기반 합성 data
     - 현황 파악 필요 (data 합성 파이프라인) [todo]
     - MCP data trajectory 어떤 trajectory 구조인지 파악 


* Planning
[⏳] data generation pipeline vs. deep research pipeline기반으로 web search인 Browsecomp 기준으로 더 나은 파이프라인 선정 (due 3.13)
[⏳] GDPval 관련해서 commoncrawl pdf 데이터 활용한 데이터 합성 파이프라인 고안 (due 3.13)
[✅] Monorepo에 data generation pipeline 이관 및 자료 정리 후 공유 [[github branch](https://github.com/UpstageAI/solar-system/tree/data-prep/feat/search-data-synth-pipeline)] 
[ ] Data 합성 파이프라인의 다양한 search step에서의 효과를 보기 위해 인위적으로 step 별 분포 확인 후 데이터 구성 (due 3.15)
[ ] MCP 기준 데이터 합성 파이프라인 파악 (due 3.16)
[ ] Common crawl (6T) 데이터 retrieved corpus (e5-base-v2) (due 3.17)
[ ] Tau2 data pipeline 추가된 사항 합성가능하도록 인계받기 (due 3.20)


[SFT]

* Progress Summary
[Preliminary Experiments]
  • Qwen3-4b model 기준 1k SFT trajectory 학습 후 BC-plus (1.2%) 상승

[Solar open 102B]
  • total: 105k instances (MCP 67.7k, Search 38.1k)
  • full think (40%), last think (30%), no think (30%)


* Planning
[⏳] SFT 학습 완료된 모델로 BC-plus 및 MCP-ATLAS 평가 예정 (due 3.15)
[ ] SFT 학습 configuration 탐방 및 alpha 모델로 변경하여 추후 학습 진행 (due 3.19)
[⏳] pass@k evaluation에서 0.2~0.8 기준 데이터들로 SFT 학습 후 BC-plus, MCP-Atlas 평가 (due 3.20)

[RL]

* Progress Summary
[Preliminary Experiments]
  • self-assessment 기반 Qwen3-4b model 기준 BC-plus (6.89%) 상승

[Qwen 14B]
  • AWS Multi-node 재현성 실험 진행중

[Solar open 102B]
  • total: NQ (14.5k), HotpotQA (14.5k), Asearcher (14.5k)
  • 위의 데이터 기준으로 Multi-node 실험 진행 예정

[산학협력 KAIST 이재길 교수님]
  • self-rewarding 방식으로 small scale 모델 탐구 및 분석 진행중
  • 유효한 분석이 제공될 시 large scale 모델 적용해볼 예정 


* Planning
[⏳] self-rewarding 분석 관련 참조 자료 제시 및 미팅노트 작성 (due 3.11)
[ ] SARL solar open 102B experiment 후 BC-plus, GAIA, FRAMES 평가 (due 3.16)
[ ] SARL 관련 설명 필요 (due 3.19)
[ ] self-assessment initial (~50) step에 completeness 되어버리는 현상 analysis (due 3.21)



[Evaluation]

* Progress Summary
[Browsecomp]
  • deepresearch pipeline으로 GLM 5 FP8 모델 평가중
  • 
[Browsecomp-plus]
  • BM25 setting, get_document (full content browsing 방식) 등 variation 중 가장 좋은 평가방식 파악중

[GAIA]
  • evaluation 방식은 reference에서 찾아서 정립 예정 @Taewhoo(이태후) 

[FRAMES]
  • text-only data x개에 대하여 평가 제안 @Taewhoo(이태후) 
  • evaluation 방식은 reference에서 찾아서 정립 예정 @Taewhoo(이태후) 

[MCP-Atlas]
  • MCP server를 제공해주고 task 수행을 평가하는 지표
  • outcome 기반의 task rubric 수행 여부를 판별

[Tau2]
  • airline, retail, mock, telecom 총 4개의 specific domain에서의 tool use를 평가

[BFCL]
  • memory, multi-turn, single-turn등 general tool use ability를 평가

[Toolathlon]


* Planning
[✅] Toolathlon 파악
[⏳] deepresearch pipline으로 solar open, GLM-5, GLM-4.7 Browsecomp 평가 (due 3.13)
[ ] 공용 repo에 Tau2, BFCL 평가되도록 세팅 후 학습한 모델들로 평가 (due 3.16)
[⏳] pass@k evaluation data 기반 유효한 (0.2 ≤ pass rate ≤ 0.8) QA pair set statistics 전달 (due 3.19) 
[ ] MCP-Atlas API key setting되는대로 solar open 및 benchmark 평가 (due 3.20)
[⏳] GAIA, FRAMES benchmark 자료 정리 및 평가 방법 정립 @Taewhoo(이태후)  
[ ] GAIA, FRAMES wbl-eval migration (due 3.20)
[ ] GAIA, FRAMES benchmark 평가 (due 3.21)


[Misc.]

* Progress Summary
  • GLM 4.7 vs. GLM 5 FP8 inference quality 비교
  • GLM 4.7 max token에 따른 inference quality & speed
  • Paper list update & followup
  
* Planning