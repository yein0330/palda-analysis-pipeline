# PALDA Analysis Pipeline

한국어 상품 Review/Q&A의 테마·감성 분류, 근거 기반 집계, 셀러용 분석 리포트를 위한 **실험 코드 공개 스냅샷**입니다. 개별 구매자 질문에 답하는 챗봇이 아닙니다.

2026-09-30 인수인계 시점의 코드입니다. 개선 작업은 일시 중지했으며 모델 승격·배포를 의미하지 않습니다. 기존 비공개 저장소의 Git 이력은 포함하지 않았습니다.

## 프로젝트 구성

| 디렉터리 | 역할 |
|---|---|
| `qna_review_model/` | KcELECTRA 51테마 개발, Review 문자 분류기 결합, 전처리·검증·LaaJ |
| `palda-encoder-pipeline/` | 독립 LLM 라벨링·합의, Gold/Silver 릴리스, 서비스용 334테마 encoder |
| `palda-decoder-pipeline/` | 근거 기반 summary/selling-point 학습 데이터, Qwen LoRA 학습·평가 |
| `palda-service-runtime/` | encoder → 코드 집계/근거 선택 → decoder → API |
| `experiments/encoder-20260929/` | 40개 개발 실험의 집계 metrics; 개별 원문/예측은 제외 |
| `docs/handoff/` | 데이터·분류 버전·구조·성능·재현·미완료 작업 |

## 인수인계 문서

- [데이터셋 생성 이력](docs/handoff/DATASETS.md)
- [테마/감성/리프카테고리 버전](docs/handoff/TAXONOMY.md)
- [모델 아키텍처](docs/handoff/ARCHITECTURE.md) · [SVG](docs/handoff/architecture.svg) · [Mermaid](docs/handoff/architecture.mmd)
- [실험 결과와 한계](docs/handoff/EXPERIMENTS.md)
- [재현 및 다음 작업](docs/handoff/RUNBOOK.md)
- [현재 서비스 후보 taxonomy](docs/handoff/taxonomy/current-service-taxonomy.json)
- [리프 → Cat2 테마 연결](docs/handoff/taxonomy/leaf-to-cat2-themes.json)
- [T1~T10 제안: 미학습/미배포](docs/handoff/taxonomy/common-theme-v0.1-proposal.json)

문서에 나오는 내부 ZIP, 원본 release, 서버 경로는 역사적 인계 기록이며 이 저장소에 모두 들어 있지 않습니다. `/path/to/workspace`는 예시 자리표시자입니다. 일부 실험 스크립트의 서버 전용 기본 경로는 실행 전에 본인 환경에 맞춰 확인해야 합니다.

## 성능을 해석할 때

KcELECTRA 계열의 같은 개발 Validation 749건에서 Review micro-F1은 **0.3633 → 0.5009**입니다. Q&A 주 테마 accuracy는 0.5101, 보조 테마 micro-F1은 0.2442입니다. 반복 사용한 개발 자료이므로 독립 Test/운영 성능이 아닙니다. 연구 최고치 0.5144는 metrics-only로 신경망 가중치가 없습니다.

별도 서비스용 ko-sroberta는 head 334개/Val 707건으로 데이터와 지표가 다릅니다. 두 모델을 같은 계열의 연속 점수로 비교하지 마세요. 최초 요구한 Review/Q&A별 두 전용 LoRA 구조가 그대로 완료된 것도 아닙니다.

리프 참조표 5,090개 중 기존 Cat2 테마에 연결되는 것은 4,472개, 미연결은 618개입니다. 리프별 독립 학습 모델이 아닙니다.

## 설치와 테스트

각 하위 프로젝트는 독립 Python package입니다. **아래 명령은 이 저장소 최상위에서 실행**합니다. 학습 dependency는 각 pyproject의 train extras와 qna_review_model의 requirements-train.txt를 확인하세요. 원래 환경 pin과 현재 설치 환경이 동일하다고 가정하지 마세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './qna_review_model[dev]'
python -m pip install -e './palda-encoder-pipeline[dev]'
python -m pip install -e './palda-decoder-pipeline[dev]'
python -m pip install -e './palda-service-runtime[dev]'
```

테스트는 프로젝트별로 실행합니다(동일한 test 파일명 충돌 방지). tensor/model 의존 테스트는 torch 등 추가 dependency가 필요할 수 있습니다. [공개본 검증 결과](PUBLIC_RELEASE.md)를 참고하세요.

```bash
cd qna_review_model
PYTHONPATH=src:. python -m pytest -q
```

학습을 자동 시작하는 CI나 업로드 자동화는 포함하지 않았습니다. 개발 평가는 Validation을 명시하고 Test/simulation은 후보 동결 전 사용하지 마세요. `evaluate_encoder_checkpoint.py`의 기본 Test 선택에 특히 주의하세요.

## 모델·데이터·공개 범위

가중치는 이 GitHub 저장소에 포함하지 않습니다. Hugging Face 재공유 URL/revision은 확정·검증 후 별도로 연결할 예정입니다. 기존 `.env.example`의 HF repo는 구성 예시/기존 경로이며 이번에 재업로드가 완료됐다는 뜻이 아닙니다.

원천 리뷰·문의, 학습/평가 JSONL, 개별 예측, 실제 주석 감사 JSON, credentials, `.env`, 로컬 endpoint 설정, 내부 인계 ZIP, 모델 binary와 joblib은 공개 대상에서 제외했습니다. 테스트에는 합성 예문과 비식별화 기능 검사에 쓰는 가짜 연락처가 있습니다.

코드 공개가 원천 데이터나 base model의 재배포 권한을 부여하지 않습니다. 저장소의 별도 오픈소스 라이선스는 소유자가 아직 지정하지 않았으며, 임의로 MIT/Apache 라이선스를 부여하지 않았습니다. 모델 재배포 시 upstream 조건과 학습 데이터 권한을 별도로 확인해야 합니다.
