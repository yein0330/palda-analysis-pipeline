# 공개 스냅샷 범위와 검증

대상: `yein0330/palda-analysis-pipeline`, public, 2026-09-30 스냅샷.

## 공개본 변경 사항

- 원본 네 작업 디렉터리를 수정하지 않고 별도 staging에서 선별했다. 원본의 uncommitted 실험 코드도 현재 파일 내용으로 포함하며 과거 Git 이력은 옮기지 않았다.
- 원문 데이터, weights, 개별 예측, 실제 라벨 주석 JSON, 내부 ZIP, secrets는 제외했다.
- `audit_common_themes.py`의 실제 원문 ID 14개 목록은 공개본에서 빈 set으로 바꿨다. 개인정보/출처 연결을 피하기 위한 것으로 원래 감사 결과가 그대로 재현된다고 주장하지 않는다. 본인에게 권한이 있는 주석 설정으로 재구성해야 한다.
- `SOURCE_EXPORT.json`은 최초 복사 시점의 원본 코드 해시다. 위 감사 스크립트 수정과 공개본 추가 문서는 원본과 다르다. 최종 공개 파일은 `PUBLIC_FILES_SHA256.json` 및 Git 커밋으로 식별한다.
- 비공개 artifact가 필요한 encoder integration test 5개는 명시적으로 skip한다. 해당 자료에 접근 권한이 있고 경로를 복구한 경우에만 `PALDA_RUN_PRIVATE_ARTIFACT_TESTS=1`로 실행한다. 테스트 통과를 위해 원문을 공개하지 않았다.
- 문서의 서버 home 경로는 `/path/to/workspace`로 대체했다. 코드의 기존 실험 경로·옵션은 유지했으므로 실행 전에 확인해야 한다.

## 검증

외부 모델 다운로드를 막고(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) staging 코드에 대해 프로젝트별로 테스트했다. 새 학습·Test 데이터 평가·모델 승격은 하지 않았다.

| 프로젝트 | 결과 |
|---|---|
| qna_review_model | 72 passed |
| palda-encoder-pipeline | 63 passed, 5 private-artifact tests skipped |
| palda-decoder-pipeline | 10 passed |
| palda-service-runtime | 48 passed |

Python 구문과 JSON 파싱, 비밀키/토큰/개인키/인증 URL/사설 IP/연락처 패턴을 점검했다. 연락처 탐지 2건은 비식별화 테스트용 가짜 전화번호와 `person@example.com`이었다. 패턴 검사가 모든 개인정보 또는 권리 문제를 보장하지 않으므로 새 파일 공개 시 다시 검토해야 한다.

학습 데이터/weights가 없으므로 Git clone만으로 전체 학습·리포트 생성까지 재현되지는 않는다. 검증 환경은 기존 로컬 venv를 사용했으며 새 환경에서 모든 dependency 설치까지 검증했다는 뜻은 아니다. HF 재공유는 이 코드 공개와 별도다.
