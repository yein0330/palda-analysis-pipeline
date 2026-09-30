# 실험 결과와 해석

수치는 저장 결과를 정리한 것이며 인계 중 학습·평가를 다시 실행하지 않았다. 상세 원본은 zip의 `references/a-EXPERIMENTS.md` 및 B 실험 문서, `snapshots/`의 metrics다. 표의 소수 네 자리는 반올림이다.

## 1. 지표를 읽는 법

- Review micro-F1: 테마와 감성 조합을 맞히는 precision/recall의 조화평균. 일반적인 단일 라벨 accuracy가 아니다.
- Q&A primary accuracy: 주 테마 한 개를 맞힌 비율.
- Q&A secondary micro-F1: 보조 테마 집합의 precision/recall. 빈 집합이 많으므로 exact match만 보지 않는다.
- macro-F1은 희귀 테마까지 동일 가중으로 본다. 포함 라벨 집합/zero-support 정책에 따라 값이 달라진다. A의 원래 metric과 고정 147개(49개 일반 테마×3) macro 재계산을 섞지 않는다.
- B `service_macro_f1 = mean(Review theme-sentiment macro-F1, Q&A primary macro-F1, Q&A secondary macro-F1)`.
- A의 세 지표 합을 사용하는 선택 점수와 B의 service_macro_f1은 다른 지표다.

## 2. A: 동일 749건 Validation 비교

| 실험 | Review micro-F1 | Q&A primary accuracy | Q&A secondary micro-F1 | 판단 |
|---|---:|---:|---:|---|
| 3epoch baseline, seed 20260901 | .3633 | .4980 | .2422 | 기준 |
| PRICE 정제 변경 | .3762 | .4939 | .2396 | PRICE 문제가 해결됐다고 볼 근거 부족 |
| Silver 일괄 weight .3 | .3528 | .4777 | .2403 | 채택 안 함 |
| 6epoch, 선택 epoch 4 | .4355 | .5101 | .2442 | 가중치 보존 |
| positive weight 1 | .3083 | .4858 | .2180 | 채택 안 함 |
| train-balanced Review weight | .4180 | .4899 | .2426 | 종합 개선 아님 |
| Gold lexical 라벨 보존 | .4484 | .4980 | .2759 | 데이터 정책 연구; 원본 미변경 |
| mean pooling | .4077 | .5304 | .2564 | 태스크 trade-off |
| 문자 분류기 단독 | .4271 | .4211 | .2333 | Review 결합 재료 |
| **6epoch + 문자 Review 50:50** | **.5009** | **.5101** | **.2442** | 재추론 가능한 후보 |
| Review loss ×3 | .4649 | .5182 | .2486 | metrics-only |
| Review loss ×3 + 문자 | .5144 | .5182 | .2486 | 연구 최고; 신경망 가중치 없음 |
| 테마별 attention | .4361 | .5061 | .2238 | 추가 이득 없어 채택 안 함 |
| Q&A만 mean pooling | .4226 | .5385 | .2240 | 주 테마 개선/보조 악화 |
| 위 Q&A pooling + Review 문자 | .5033 | .5385 | .2240 | 우위 미확정 |
| Review Gold만 복원 + 문자 | .5025 | .5061 | .2191 | Q&A 악화 |
| char + word lexical 결합 | .4882 | .5101 | .2442 | 기존 문자 조합보다 낮음 |

현재 재추론 후보는 Review false positive를 많이 줄였지만 recall은 0.4591이다. 진단에서는 테마가 없는 곳에 붙인 FP 581→270, 맞는 테마의 감성 FP 93→54로 감소한 반면, 누락 테마 FN 364→426으로 증가했다. 숫자는 기존 Gold 기준 오류 분해이며 새 사람 심사 결과가 아니다.

최종 재추론은 `blend-weight-reproduction-v2`에 저장돼 있다. 저장 예측을 복사한 점수만이 아니라 실제 가중치에서 다시 추론해 Review 확률 최대 차이 0.0, Q&A 주 테마 변경 0을 확인한 기록이다. 인계 작업에서는 이를 재실행하지 않았다.

### 다른 seed

| 구성 | seed 1 Review F1 | seed 2 Review F1 | 해석 |
|---|---:|---:|---|
| 3epoch 기준 | .3633 | .3596 | 기준 변동 |
| 6epoch 신경망 | .4355 | .3867 | 신경망 seed 민감 |
| 6epoch + 같은 문자 분류기 | .5009 | .4787 | 두 번 모두 개선, 완전 독립 결합 seed 실험은 아님 |
| Review loss ×3 신경망 | .4649 | .4708 | Review 향상 |
| Review loss ×3 + 문자 | .5144 | .5070 | seed 2 Q&A .4858/.2259로 악화 |

두 seed가 모든 태스크의 안정적 개선을 증명하지 않는다. 특히 같은 문자 artifact를 공유했다. B의 최신 승격 규칙인 5개 이상의 paired seed를 충족하지 않는다.

### Train 내부 상품 holdout 528건

기존 749건에 반복 적응하는 문제를 점검하려고 기존 Train에서 40개 상품을 따로 떼었다. 절대 점수를 위 표와 비교하면 안 된다.

| 구성 | seed 1 Review F1 | seed 2 Review F1 |
|---|---:|---:|
| 내부 6epoch | .5136 | .4970 |
| 내부 6epoch + 문자 | .5337 | .5187 |
| 내부 Review loss ×3 | .5284 | .5350 |
| 내부 Review loss ×3 + 문자 | .5345 | .5376 |

seed 1 신경망→문자 결합의 상품 bootstrap 개선 구간은 [-.0166,+.0552]로 0을 포함했다. loss ×3의 결합 추가 이득도 seed 1은 +.0008에 불과했다. seed 2에서는 개선이 있었지만 내부 split에도 checkpoint 선택이 들어갔다. 독립 Test 확인 전 잠정 연구 결과다.

### Q&A 개선과 신뢰도 보정

Q&A 신경망과 문자 주 테마를 50:50 결합하면 accuracy .5101→.4939로 떨어졌다(기존 오류 5개 수정, 정답 9개 손상). Review 성공을 Q&A에 일반화하지 않았다.

신경망·문자 주 테마 동의 + 신경망 점수 ≥.8이면 247건 중 66건을 처리하고 60건을 맞혔다. 조건부 정확도 90.9% / coverage 26.7%다. 이것은 모든 문의의 90.9%를 처리한다는 뜻이 아니다. 실제 운영 gate로 적용하지 않았다.

상품별 5-fold OOF temperature 보정은 NLL 1.8036→1.6916, Brier .6680→.6346, ECE .1548→.1046이었다. 주 테마 accuracy는 그대로다. 보정된 신뢰도와 맞힌 문제 수를 구분한다. 운영 calibration artifact/threshold를 교체하지 않았다.

## 3. 자동 LaaJ 진단

LaaJ는 원문 근거를 확인하는 자동 판정이며 사람 Gold도 독립 외부 평가도 아니다. 같은 assistant가 이전 개발 상황을 알고 있어 완전 독립 blind judge라고 부르지 않는다.

- 초기 12건: 유지 1 / 보류 5 / 모델 오류 2 / 라벨 수정 후보 4. 자동 Gold 편입 안 함.
- 동결된 8건 회귀 묶음과 미사용·미래·근거 없음 등의 12개 합성 probe를 사용했다. 합성 probe는 Train이 아니다.
- blind-laaj-v2: 새 24상품/24건, Review/Q&A 각 12. 행별 정답과 예측을 가리고 먼저 판정. 평가 가능 14건, taxonomy 보류 7, 정책 보류 2, 문맥 부족 1.
- 그중 Review 9건: 승인 테마 범위 내 근거 없는 추가 라벨 **11→3**, 필수 라벨 검출 **5/13 그대로**, 모든 제약 통과 **0/9→2/9**.
- 단순 6epoch의 미사용 probe는 1/12 통과. loss ×3도 4/12 또는 3/12로 문제가 남음. 21행 수정 ablation은 핵심 문제를 해결하지 못했다.

오류 유형: 사용 전 기대를 경험으로 오인, 부정문 반전, 이전 상품 고장을 현재 상품 문제로 해석, 재고/옵션 문의 테마 공백, 특정 카테고리 안전 테마 누락, 호환과 지원 매체 경계. 형식 검사나 F1 하나만 높여서는 해결되지 않는다.

## 4. B: 별도 서비스 계열의 과거 실험

| 실험 | Validation service macro-F1 | 조건 |
|---|---:|---|
| KcELECTRA Gold only | .077438 | 3seed 평균 |
| KcELECTRA Gold+Silver | .149301 | 3seed 평균 |
| Gold+Silver+선택 합성 | .146858 | 3seed 평균 |
| mDeBERTa | .154527 | 단일 seed, 별도 backbone 비교 |
| multilingual-e5-base | .208948 | 단일 seed |
| ko-sroberta | .228846 | 3seed 평균 (.240086/.223465/.222987) |

ko-sroberta 선택 seed는 20260909다. aggregate 최고 seed 42는 Q&A secondary guard 문제가 있어 그대로 고르지 않았다. 5epoch 일부 결과는 점수가 더 높아도 secondary macro 악화로 유지하지 않았다. 길이 384/512, Silver weight .1/.5, hard-negative 추가도 일관된 이득을 보이지 못했다.

기존 보관 Test 결과(새로 열거나 재평가하지 않음):

| 지표 | KcELECTRA Gold+Silver | ko-sroberta 후보 |
|---|---:|---:|
| service macro-F1 | .098489 | .158921 |
| Review micro-F1 | .331898 | .465923 |
| Q&A primary accuracy | .436040 | .489433 |
| Q&A secondary micro-F1 | .073552 | .099567 |
| Q&A secondary exact match | .383760 | .684093 |
| encoder records/s | 910.59 | 932.66 |

처리량은 당시 환경/배치 조건의 인코더 단독 수치다. 리포트 API end-to-end 처리량과 다르다. 과거에는 Test 비교로 길이 후보를 판정한 기록도 있다. 이를 최신 정책의 독립 확증 결과처럼 포장하지 않는다.

후속 paired bootstrap에서는 backbone 개선 방향이 강했지만 3seed뿐이었다. relabel384(+.006207), secondary Silver negative discount(+.004176) 등의 CI는 0을 포함해 미확정이었다. registry에 과거 단일 seed 상태가 남아 있으며 전체 계보를 완전히 추적하지 못하므로 registry 한 파일만 보고 현재 최선을 결정하지 않는다.

## 5. 리포트·Decoder 실험

- 9/28 E2E: 쌀냉장고·벽지·손수건 3상품. 여러 입력 크기/반복 실행 수를 독립 상품 수로 세지 않는다.
- warm 단일상품 Review+Q&A 합계 1,000건에서 약 31.4초(encoder 3.2초, decoder 28.2초). cold start/네트워크/동시 요청 SLO는 아님.
- 100건 조건 baseline 25.6초 → LLM verify 40.7초. verify가 오분류 confidence를 1.0으로 만들고 부적절 근거 순위를 높이는 부작용이 있었다.
- grounded-v2.4.1은 intent 집계를 attribute usable과 분리하는 코드 수정까지 실험. 자동 운영 승격은 안 됨.
- 9/29 evidence-diversity 저장 요약: 26case, 그룹 수 141 유지, 공유 quote 수 91→39. 중복 감소이지 의미 정확도 향상 측정은 아님.
- 9/29 adapter-vs-base: 22case, 각 185group. 원문 grounding 185/185는 양쪽 동일. LoRA는 fallback summary 0 vs base 6, 비한국어 0 vs 5로 나았지만 generic summary 35 vs 30, 요약 복사 소구점 35 vs 13, 유형-내용 불일치 7 vs 5로 나빴다. “LoRA가 항상 더 좋다”는 결론 불가.

Decoder 데이터 감사: 별도 leaf-v1 소구점 100건은 judge_count 0인데 quality_score 1.0으로 선정됐던 경로가 있다. 현재 imported-v2 데이터와 직접 동일한 문제라고 단정하지는 않지만 새 수집 루프를 돌리기 전 반드시 고쳐야 한다. imported 학습 데이터에도 긍정 소구점 유형에 불만 내용이 붙은 사례가 Train 61/826, Val 5/103으로 기록됐다. 새 후보의 어휘 게이트는 추가됐으나 기존 release는 자동 수정하지 않았다.

소구점 pairwise 선호 데이터 생성 스크립트와 사람 비교 표본 도구는 존재한다. 합성 판정 smoke 결과를 실제 DPO 학습 완료로 해석하면 안 된다. GRPO는 보상 포화와 실제 품질 신호 부족으로 보류 권고다.

## 6. 성능 보고의 최종 문장

“최근 A 개발에서는 Review 결합 F1이 0.3633에서 0.5009로 개선됐고 로컬 가중치 재추론도 확인했다. Q&A 개선은 작고, 연구 최고치 0.5144는 가중치가 없다. B 서비스 후보는 다른 데이터/지표의 별도 모델이다. 두 계열 모두 의미 오류와 독립 검증의 한계가 남아 있어 전체 자동 리포트 품질이나 운영 정확도를 보장하지 않는다.”
