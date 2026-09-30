# 데이터셋과 생성 이력

경로 접두어 `PALDA = /path/to/workspace/PALDA`. 아래 숫자는 저장된 manifest/평가 기록 기준이며 인계 중 Test 원문을 다시 열어 집계하지 않았다.

## 1. 초기 Amazon 경로와 현재 한국어 경로

대화 초기에 Amazon Reviews는 `parent_asin`, Amazon-QA는 `asin`으로 상품별 그룹화하려 했다. Q&A에 asin이 없으면 상품 리포트용 주력 데이터가 아니라 단건 테마 분류 보조 데이터로 취급해야 한다. `collect-metadata --categories All_Beauty --max-products-per-category 100`의 100은 **상품 메타데이터 수**이지 리뷰 100건이나 학습 완료를 뜻하지 않는다.

초기 Amazon 샘플·tar.gz·네이버 xlsx 경로는 역사적 맥락이다. 현재 인계 대상 학습 manifest는 한국어 네이버 데이터와 내부 라벨링/분할 산출물을 가리킨다. 대화에 등장한 “5건”을 현재 전체 데이터 수라고 해석하면 안 된다. 초기 Amazon 파일의 현재 보존 위치·HF 원격 revision·라이선스는 이번 인계에서 확정하지 못했으며, 현재 학습 데이터의 출처로 임의 연결하지 않았다.

## 2. 한국어 원천 → 정규화 → 라벨 → 분할

원천 위치는 `qna_review_data/raw/crawling/2026-08-14-full-5090-w12-sqlite/`이다. 수집 상태는 progress/content-summary/content-manifest와 sealed content-chunks를 기준으로 확인한다. SQLite나 브라우저 프로필을 바로 학습 입력으로 쓰지 않는다.

정규화는 리뷰/문의 원문, 상품 식별자, 카테고리와 필요한 상품정보를 유지하고 작성자 프로필·주문정보·불필요한 이미지/수집 URL 등은 후속 모델 입력에서 제외하는 방향이다. 상품의 제목·브랜드·가격·옵션 등은 inventory 쪽 자료와 연결한다. 단순 키워드 파일과 원천 행에 개인정보가 없다는 보장은 별개다. 외부 전송 전 별도 검토가 필요하다.

라벨은 사람 검토 Gold와 LLM 생성/합의 Silver를 구분한다. LLM끼리 합의했다고 사람 정답이 되는 것은 아니다. 합성 자료의 생성 모델은 자기 결과의 심사에서 제외하고 다른 모델의 독립 판단을 사용한다. 불일치 표본은 hard pool로 분리한다.

분할은 상품 단위, 중복/동일 텍스트 그룹을 함께 고려한다. 같은 상품 또는 동일 문장이 Train과 평가로 섞이지 않게 한다. 이후 수정은 새 release로 만들고 기존 raw/normalized/분할은 보존한다.

## 3. A: `processed/encoder-v1`

원본: `qna_review_data/processed/encoder-v1/manifest.json`.

- datasetVersion: `palda-encoder-19127-gold-overlay-02`
- splitVersion: `product-dup-balanced-70-15-15-v2`
- source: `simulator/data/source/encoder-v1-overlay-02.jsonl`
- taxonomyVersion: `theme-v1.0.0`
- 생성: `scripts/export_encoder_from_simulation_split.py`가 simulator의 고정 분할을 사용하고 Train 정제·중복 제거, Validation/Test Gold-only를 적용.

| 분할 | 행 | Review | Q&A | 상품 | 정답 층위 |
|---|---:|---:|---:|---:|---|
| Train | 10,765 | 7,699 | 3,066 | 2,503 | Gold 5,363 + Silver 5,402 |
| Validation | 749 | 502 | 247 | 50 | Gold |
| Test | 823 | 559 | 264 | 55 | Gold; 최근 A 개선에서는 미사용 |

상품/중복 그룹 split overlap은 manifest 기준 0이다. Silver valid/high 조건, 어휘 근거가 약한 일부 테마, 모든 타깃이 제거된 행, 정규화 텍스트 중복을 정제했다. 세부 제외 수는 zip의 원본 manifest를 우선한다.

중요: **지금의 정제 코드와 이미 생성된 Train은 같은 시점의 정책이 아니다.** 최근 코드에서는 approved/corrected Gold를 보존하고 어휘 문제를 audit로만 남기도록 바뀌었다. PRICE 어휘에도 `싼`이 추가됐다. 현재 코드로 export를 재실행하면 기존 Train과 동일할 것으로 기대하면 안 된다. 기존 릴리스의 해시를 기준으로 재현한다.

기존 Train 해시: `c172f32bcf7eb23d427c332a9763585a2a5e366cc9a74939b6bc9f30023c6209`

Validation 해시: `48c37dac16e81156ece12230e0d9625deee8be9f8b6d915150ef513b4d87e04f`

Review target은 테마×감성의 다중 라벨, Q&A는 주 테마 1개·보조 테마 집합·usable이다. 상품 묶음 리포트 JSON과 다른 **단건 인코더 학습 레코드**다. Val Q&A usable은 247건 모두 True여서 usable accuracy 1.0은 무의미한 입력을 걸러내는 능력의 증거가 아니다.

### 파생 ablation과 내부 holdout

- Gold lexical 복원: 원래 Gold 기준 293행의 305라벨 복원안. 원본 Train에 덮어쓰지 않고 별도 실험했다.
- Review-only 복원: 187개 Review 행의 196라벨 복원안. Q&A 라벨 변경 효과와 분리하기 위한 실험이다.
- 자동 판정 abstention 21행 수정안도 별도 ablation으로만 사용. 정식 Gold에 반영하지 않았다.
- `processed/encoder-inner-holdout-20260929`: 기존 Train에서 Gold가 충분한 40개 상품을 분리. 내부 Validation 528건(Review 375/Q&A 153), 내부 Train 10,228건. 같은 상품의 Silver와 동일 텍스트도 Train에서 제외했다.
- 내부 holdout도 checkpoint 선택에 사용한 개발 자료다. 기존 749건과 분모가 다르고 독립 Test가 아니다.

## 4. B: 서비스 인코더 릴리스 계열

| 릴리스 | Train | Validation | Test | 설명 |
|---|---:|---:|---:|---|
| `palda-encoder-leaf-v1` | 3,046 Silver | 707 Gold | 2,808 Gold | 초기 격리 릴리스. simulation 3,507건 제외; 현재 champion 학습 릴리스 아님 |
| `gold-anchor-experiment-v1/gold_only` | 1,755 Gold | 707 | 2,810 | Gold anchor 효과 비교 |
| `gold-anchor-experiment-v1/gold_silver` | 11,656 | 707 | 2,810 | Gold 1,755 + Silver 9,901; ko-sroberta 후보의 학습 계열 |
| `gold-anchor-experiment-v1/gold_silver_selective` | 11,795 | 707 | 2,810 | 선택적 합성 추가 비교 |

B 기준 Gold+Silver manifest에서 Train 7,328상품, Validation 52상품, Test 204상품이며 product/exact_text leakage는 0이다. 기존 human simulation 3,509건 중 1,755건/132상품을 Gold anchor Train으로 이전하고 1,754건/130상품을 별도 보존했다. 초기 leaf 릴리스의 3,507/2,808과 이후 3,509/2,810은 서로 다른 릴리스의 기록이므로 숫자를 맞추려고 파일을 수정하지 않는다.

파이프라인은 Kanana/Llama/Qwen 등 독립 라벨링, 카테고리별 합의, 희귀 라벨 합성, hard pool 재검토를 지원한다. 하지만 기능이 있다는 사실과 특정 release가 그 모든 단계를 거쳤다는 사실은 다르다. 각 release manifest와 label_runs를 기준으로 추적해야 한다.

## 5. Decoder 학습 데이터

### A 원본 제품 묶음

`qna_review_data/processed/decoder-v1`: Gold 인코더 라벨을 사용한 oracle 집계 입력. Train 413상품 / Validation 50 / Test 55. 입력 Gold 레코드는 각각 5,363 / 749 / 823. 그룹당 근거 최대 3개. Qwen3-8B 생성 draft는 Train 412 valid+1 needs_review, Validation 49 valid+1 needs_review, Test 55 valid로 기록돼 있다.

이 원본 manifest는 `trainingEligibleNow=false`, `draft_pending_human_review`다. 형식 검증을 통과한 초안이 사람 승인 정답이라는 뜻은 아니다.

### B/C에서 import한 학습 릴리스

`palda-decoder-pipeline/data/releases/palda-decoder-v1-imported`:

| 분할 | 학습 task 행 | 상품 |
|---|---:|---:|
| Train | 3,522 | 412 |
| Validation | 462 | 49 |
| Test | 512 | 55 |

전체 4,496행은 **Silver**이며 review_summary 2,635 / qna_summary 1,347 / selling_point 514다. needs_review 상품은 Train/Validation 각각 1개 제외했다. split은 기존 product_group_id를 보존했고 manifest의 split overlap은 0이다.

즉 “원본 draft는 학습 적격 false인데 후속 파이프라인에서 Silver로 import해 학습했다”는 이력이 공존한다. 이를 사람 승인 완료로 소급 해석하지 않는다. 실서비스 승격에는 별도 Gold 평가가 필요하다.

Decoder는 실제 인코더 오류를 포함한 입력보다 oracle 라벨 기반 입력을 많이 보았을 수 있다. 원천 분류 오류 → 그룹 오류 → 요약 오류가 이어지는 분포 차이를 다음 E2E 평가에서 확인해야 한다.
