# 재현·운영·다음 작업

## 1. 중단 상태와 보존 대상

자동 개선 goal은 **paused**다. 인계 시 새 학습/다운로드/배포를 시작하지 않았다. 최근 A 개발에서 기존 `current`를 교체하지 않았고 Test 823건을 사용하지 않았다. B의 과거 Test 결과는 저장된 metrics/문서만 참조했다.

반드시 보존할 위치:

- `qna_review_data/raw`, `normalized`, `processed`, `reference`; simulator의 원천 overlay와 상품/중복 분할.
- A: `models/checkpoints/encoder/experiment-20260929-epochs6/best` 및 `experiment-20260929-lexical-repro/review-model.joblib`.
- A: `models/evaluations/validation-audit-20260929` 전체, 각 실험 metrics/provenance/config/predictions.
- B: `palda-encoder-pipeline/data/releases/gold-anchor-experiment-v1/gold_silver` 및 `artifacts/gold-anchor-backbone-pilot/ko-sroberta-champion-candidate-test-audit`.
- C: `palda-decoder-pipeline/data/releases/palda-decoder-v1-imported` 및 `artifacts/decoder-imported-v2-qwen2.5-7b/checkpoint-221`.
- 실제 사용한 tokenizer, backbone/adapter weights, thresholds, taxonomy, 환경 lock/requirements. 코드만으로 가중치가 복원되지는 않는다.

원문과 weights는 이 작은 zip에 들어 있지 않다. 서버 이관이 필요하면 접근권한·보관 위치·디스크 예산을 별도로 정한 뒤 복사한다. 자동으로 외부 저장소에 올리지 않는다.

## 2. 환경과 설치 실수 방지

A 최근 학습 환경은 `.venv-train`, Python 3.12.9, torch 2.5.1+cu121, RTX 4090 기록이다. 학습 config는 `cuda:0`을 사용했다. 다른 GPU 사용자 작업을 종료하지 않는다. B의 과거 metrics에는 cuda:3도 있으므로 현재 GPU 소유권을 과거 기록으로 판단하면 안 된다.

먼저 읽기 전용으로 확인:

```bash
cd /path/to/workspace/PALDA/qna_review_model
pwd
df -h .
nvidia-smi
.venv-train/bin/python --version
.venv-train/bin/python -m pip show torch transformers peft scikit-learn
```

인계 검사 시 여유 공간이 약 6MB였다. 새 env/모델을 만들기 전에 공간 문제부터 해결해야 한다. 이번에 기존 데이터를 지우지는 않았다.

`pip install -e .`는 pyproject.toml이 있는 프로젝트 루트에서 실행해야 한다. `/path/to/workspace`에서 실행하면 이전에 겪은 “setup.py/pyproject.toml 없음” 오류가 난다. conda activate 자체가 금지된 것은 아니다. 이 프로젝트의 재현은 **실제로 사용한 `.venv-train/bin/python`을 명시**하면 shell의 `(base)` 중첩 표시와 무관하게 interpreter가 고정된다. conda 환경을 새로 쓸 때는 실제 dependency를 대조해야 하며 여러 환경을 동일하다고 가정하지 않는다.

`requirements-train.txt`는 참고 pin 파일이다. 현재 설치 버전/드라이버/모델 revision까지 자동으로 보장하지 않는다. 패키지 업그레이드는 인계 작업에 포함하지 않았다.

## 3. A 후보 재추론 명령 — 후임자가 명시적으로 실행할 때만

다음은 최근 검증에 사용한 방식이다. **이 문서 작성 과정에서 실행하지 않았다.** 출력 이름은 존재하지 않는 새 이름으로 바꾼다.

```bash
cd /path/to/workspace/PALDA/qna_review_model
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-train/bin/python \
  -m scripts.evaluate_encoder_review_candidate \
  --data-dir /path/to/workspace/PALDA/qna_review_data/processed/encoder-v1 \
  --checkpoint models/checkpoints/encoder/experiment-20260929-epochs6/best \
  --lexical-artifact models/checkpoints/encoder/experiment-20260929-lexical-repro/review-model.joblib \
  --output-dir models/evaluations/handoff-validation-NEW \
  --review-threshold 0.55 --device cuda:0
```

예상 저장 결과: Review micro-F1 .5009 / primary .5101 / secondary .2442. 환경에 따른 부동소수 차이는 확인하되 기존 원본을 덮어쓰지 않는다. joblib source hash는 `c0541efb9e91127a099bd80fd5c5559f4a3f55c65e1b304c1ba07afe5bc8922a`.

주의: 범용 `evaluate_encoder_checkpoint.py`는 기본 split이 Test인 경로가 있으므로 개발 진단에서는 반드시 `--split validation`을 명시한다. 학습 개발은 `--skip-test`. `--metrics-only`는 weights를 남기지 않으며 `--skip-test`와 함께 사용한다. 새 연구에서 metrics-only 최고 점수를 얻어도 저장 모델이라고 소개하지 않는다.

## 4. 테스트와 검증 기록

최근 A 개발의 기존 전체 테스트는 **72 passed**로 기록돼 있다. 이 인계 작업은 새 학습 코드 수정 없이 문서/패키징만 작성했으며 테스트를 새로 돌렸다고 주장하지 않는다. 인계 패키지는 JSON 파싱, SVG XML 파싱, zip integrity, 사본 해시, 로컬 문서 링크를 별도로 검사한다.

다음 담당자가 코드 변경 후 필요 시:

```bash
cd /path/to/workspace/PALDA/qna_review_model
PYTHONDONTWRITEBYTECODE=1 .venv-train/bin/python -m pytest -q -p no:cacheprovider
```

이 테스트 통과는 모델 의미 품질, 원문 라이선스, 운영 배포 승인을 대신하지 않는다.

## 5. 바로 다음에 하려던 A 실험 — 미실행

**Q&A secondary loss coefficient 0.5 → 1.5**만 바꾸는 단일 요인 실험이 다음 후보였다. 원래 Train, Review loss 1, CLS pooling, 6epochs, seed 20260901, positive weight 3을 유지하려 했다. 아직 실행하지 않았다.

배경: Review보다 Q&A 보조 테마 개선이 작다. Train Q&A Gold 1,606건의 평균 secondary 수 .3082, Silver 1,460건은 .1870, Val 247건은 .3603으로 기록돼 있다. Silver secondary가 적다는 이유만으로 누락 오답으로 간주해서는 안 된다. 실제 원문·라벨 정의 점검이 필요하다.

예정 판단 기준: 같은 Validation 세 지표, false-positive/누락, seed 반복, 기존 회귀/probe. 먼저 디스크/리소스/목표 구조를 확정해야 하며 사용자 중단 요청에 따라 자동 실행하지 않는다.

## 6. 다음 담당자의 우선순위

| 우선순위 | 작업 | 완료 기준 |
|---|---|---|
| P0 | 데이터/weights 보존, 디스크 여유, 담당자·실행 환경 확보 | 필요한 artifact를 실제 열 수 있고 해시 확인; 임의 삭제 없음 |
| P0 | A 연구/B 서비스/최초 2-LoRA 요구 중 기준 구조 결정 | 데이터/출력/배포 계약을 문서로 확정 |
| P1 | 테마·경험/질문·대상·감성 경계 정의 | stock/option/safety/negation/미사용 regression과 명시적 abstain 규칙 |
| P1 | 미판정 생성물을 quality=1로 채택하는 경로 차단 | judge 0건을 승인으로 처리하지 않는 테스트 |
| P1 | 독립 평가와 5 paired seed 비교 | 동일 split/동일 코드 baseline; bootstrap 재계산이 metric과 일치 |
| P2 | Q&A secondary 단일 요인 실험 또는 의미 단위 파일럿 | 태스크별 guard 통과 여부와 비용 동시 보고 |
| P2 | 실제 encoder 오류를 포함한 decoder/E2E 검증 | quote 존재+의미 지지+수치 일치+한국어+유형 일치 검사 |
| P3 | 소구점 선호 데이터 품질 확보 후 DPO/GRPO 검토 | pairwise 일관성/독립 품질 개선 확인; 보상 포화 해결 |

사람 개입을 줄이는 방향은 “모든 불확실성을 정답으로 자동 승인”이 아니다. 독립 LLM 합의, quote offset/수치 검증, 의미 회귀, abstain, unresolved queue를 자동화하고 정책 경계·고위험 오류만 제한적으로 검토하는 것이 권고다. 현재 LaaJ 결과를 자동 Gold로 승격하지 않는다.

## 7. B 최신 승격 규칙

`palda-encoder-pipeline/docs/experiment-policy.md`와 `adoption-rule-2026-09-23.md`의 9/28 갱신을 우선한다.

- 사전 선언한 paired seed 최소 5개. 코드 변경 후 baseline 재실행.
- seed+상품 paired bootstrap 2,000회, `--resample-seeds`; 재계산 metric 일치 확인.
- service macro-F1 평균 개선 ≥.002, 개선 CI 하한 >0.
- 각 구성 macro-F1 악화 ≤.01, Gold-only Val, leakage 0, throughput 하락 ≤20%.
- CI가 0을 포함하면 미확정. 연속 3회 미확정이면 중단. 과거 2seed/3seed 결과는 이 규칙을 소급 충족하지 않는다.
- Test는 선택용이 아닌 감사용, 남은 simulation은 후보 동결 후 사용. Test를 반복 튜닝에 재사용하지 않는다.
- 로컬 모델 선정과 서비스/HF 배포는 별도 승인. 자동 push/교체 없음.

## 8. 남은 불확실성

현재 운영 서버의 실제 env/원격 HF revision/트래픽 품질은 이번 로컬 문서화에서 검증하지 않았다. 초기 Amazon 산출물의 완전한 보존 이력도 확정하지 못했다. 서비스용 B에 A의 문자 결합을 적용한 성능은 측정되지 않았다. 최초 Review/Q&A 분리 LoRA 완성과 Q&A 원인 분석 구현은 확인되지 않았다. 이 항목들은 “완료”가 아니라 후속 확인 대상으로 남긴다.
