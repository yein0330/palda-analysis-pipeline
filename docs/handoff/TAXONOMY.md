# 테마·감성 분류 버전과 리프카테고리

## 1. 버전 비교

| 버전/파일 | 테마 단위 | 감성 | 현재 상태 |
|---|---|---|---|
| 최초 요구 | Review 9개 / Q&A 8개, Q&A 발생 원인 8개 별도 | positive/negative/mixed/neutral | 제품 요구. 현재 인코더 출력과 다름 |
| keyword v1 `config/category_themes.json` | global 4개 + Cat1 11개 그룹; Cat2/leaf 비어 있음 | 규칙 기반 초기 분석 계열 | 초기 baseline |
| keyword draft v2 `category_themes_cat2_draft.json` | Cat2 210개, leaf 비어 있음 | 초기 규칙 계열과 구분 | draft_pending_human_review |
| A `theme_taxonomy.json` | canonical 사전 334개, Cat2 210개, categoryTheme 3,439개 | positive/negative/neutral | 사전 전체가 학습 head는 아님 |
| A KcELECTRA 실험 | 고정 canonical head **51개**(OTHER/UNUSABLE 포함) | 테마×3 독립 sigmoid | 현재 재추론 후보 계열 |
| B service encoder | head **334개**, Cat2 210개 허용 테마 | 테마×3 sigmoid → 한 테마 한 감성 선택 | 런타임 샘플 고정 후보 |
| common-theme-v0.1 | T1~T10 공통 상위 테마 + 세부 주제/의미 단위 | 긍정/부정/혼합/중립/해당 없음/판단 불가 제안 | 40건 탐색 주석만 완료, 학습·배포 안 함 |

## 2. “리프 테마”가 현재 실제로 의미하는 것

`palda-encoder-leaf-v1/taxonomy.json`에도 실제 키는 `category_themes`이고 210개 **Cat1 > Cat2** 문자열을 사용한다. 서비스 코드 `palda_runtime/models.py`도 `_cat2_key(category_path)`로 앞 두 단계만 뽑는다. 전체 5,090개 리프별로 독립 테마를 학습/검증한 상태가 아니다.

`taxonomy/current-service-taxonomy.json`은 샘플 런타임이 고정한 ko-sroberta 후보의 실제 taxonomy 사본이다. `taxonomy/leaf-to-cat2-themes.json`은 `qna_review_data/reference/leaves.jsonl`의 리프 ID/경로에 현재 Cat2 테마를 연결한 **조회 편의용 파생물**이다.

- `category_themes`를 한 번만 저장하고 각 leaf가 `cat2_key`로 참조한다.
- `matched`는 문자열로 Cat2가 연결됐다는 뜻이지 그 리프의 의미 적합성 검증 통과가 아니다.
- `missing_cat2`는 누락을 그대로 노출한다. 임의 테마를 생성해 채우지 않는다.
- 원본 런타임은 Cat2 매칭이 없으면 head 전체 테마로 fallback한다. 파생 JSON은 이 동작을 `runtime_fallback`으로 명시한다. 조용한 fallback은 후속 품질 진단 대상이다.
- 실제 참조표 행 수와 매칭/미매칭 수는 생성된 `catalog.json`의 `leaf_mapping`에서 확인한다. “5,090”이라는 이름에 맞춰 행을 추가/삭제하지 않는다.

이번 참조표 대조 결과는 **5,090개 중 4,472개 matched / 618개 missing_cat2**다. 도서·면세점 다수와 일부 일반 상품군이 포함된다. 예를 들어 화장품/미용의 클렌징·향수·헤어스타일링·헤어케어 경로도 연결되지 않는다. 전체 미매칭 Cat2별 건수는 JSON에 보존했다. 카테고리 개편/별칭 문제인지 실제 미지원인지 확인하기 전 이름을 임의 치환하지 않는다. 이 618개는 모델 오분류 건수가 아니라 **카테고리 참조표 연결 누락 수**다.

## 3. A 분류 규칙

단건 입력은 `[TASK=...] [CAT1=...] [CAT2=...] 원문`이다. 공통 canonical theme로 head를 구성하고 카테고리 허용 테마를 마스크한다. `label_map.json`의 3,439개 categoryTheme 인덱스를 그대로 3,439-head로 읽으면 안 된다. **실제 선택 checkpoint의 `training_config.json/indexToCanonicalTheme` 51개**가 head의 정의다.

`expand_encoder_label_map.py` 정책은 `all taxonomy entries whose canonical theme exists in the fixed model head`다. `annotationReady=false`여도 고정 head에 존재하면 일부 카테고리 초안에서 예측 허용될 수 있다. annotationReady와 실제 mask는 같은 의미가 아니다.

Review는 한 행에 여러 테마를 붙이며 각 테마의 positive/negative/neutral 점수를 독립 계산한다. 같은 테마의 긍정·부정 동시 예측이 구조적으로 가능하다. 이것을 학습된 `mixed` 클래스라고 부르면 안 된다.

재추론 후보에서는 Review 신경망 확률과 char_wb TF-IDF+SGD 확률을 50:50 평균하고 0.55 threshold를 쓴다. Q&A는 신경망 그대로다. 이 확률은 보정된 정답 확률이 아니다. Q&A는 masked 주 테마 softmax, 보조 테마 sigmoid, usable을 사용하고 감성을 예측하지 않는다.

## 4. B 서비스 분류 규칙

ko-sroberta CLS → 공유 표현 → Review/theme-sentiment, Q&A primary, Q&A secondary, usable 4개 head다. 테마 목록은 checkpoint `label_map.json/themes`의 334개다.

Review의 `_strongest_eligible_sentiment`는 **각 감성의 threshold를 넘은 후보 중 최대 점수 하나**를 선택한다. 테마는 여러 개 가능하되 같은 테마 감성은 하나다. A 결과를 B에 연결할 때 이 후처리 차이를 무시하면 집계 수가 달라진다.

Q&A는 주 테마 1개, 보조 테마 여러 개를 추출하며 secondary에서 primary 및 특수 테마를 제외한다. Q&A 감성/문의 발생 원인은 별도 출력되지 않는다. runtime은 OTHER/UNUSABLE을 특수 테마로 취급한다. threshold 전체 값은 zip의 `snapshots/b-thresholds.json`이 기준이다.

## 5. 공통 테마 v0.1 제안

| ID | 이름 |
|---|---|
| T1 | 기능·효과·성능 |
| T2 | 소재·품질·내구성·안전 |
| T3 | 외관·디자인 |
| T4 | 규격·구성·적합성 |
| T5 | 사용·설치·관리 편의 |
| T6 | 감각·사용감 |
| T7 | 가격·경제적 가치 |
| T8 | 구매·배송·포장 |
| T9 | 판매자·브랜드·고객지원 |
| T10 | 기타·미분류 |

실제 40건/28상품/25 Cat2를 읽어 56개 의미 단위, 58개 테마 라벨을 검토했다. 모호 표시 14/56, 답변 주석 0건, 작성자 역할 확인 0건. 근거 56개 substring/offset 통과는 의미 정답률이 아니다. 혼합 감성 사례도 없어 mixed 정책 검증은 미완료다.

새 제안의 핵심은 상위 테마를 줄이는 것만이 아니다. 의미 단위별로 다음을 분리한다.

- evidence offset/text, 부정/조건 문맥, 원문 버전.
- 현재 제품/이전 제품/다른 상품, 질문/관찰/예상/미사용/전달.
- 확정 라벨과 candidate theme, review_required/reason.
- 적용 가능성(applicable/not_applicable/unknown)과 언급 여부(mentioned/not_observed).
- 기타 사유와 처리 실패 상태.

`common-theme-v0.1-proposal.json`은 이번 인계를 위해 작성한 검토안 요약이며 기존 334개 테마를 T1~T10으로 자동 변환하는 정답 crosswalk가 아니다. 기존 head나 학습 정답을 이 JSON으로 덮어쓰지 않는다. 상세 경계 규칙은 zip에 포함한 `common-theme-taxonomy-audit-2026-09-28.md`를 따른다.

## 6. 최초 요구한 문의 발생 원인

`missing_information`, `ambiguous_information`, `conflicting_information`, `hard_to_find`, `needs_personal_judgment`, `needs_seller_confirmation`, `post_purchase_issue`, `unknown`은 최초 요구에 포함됐다. 현재 단건 인코더의 주/보조 테마에서 이를 얻을 수 있다고 간주하면 안 된다. 특히 missing/conflicting 여부는 상품정보와 실제 문의/답변 문맥 비교가 필요하다. 별도 데이터 계약·라벨링·평가가 필요한 미구현 범위로 인계한다.
