# Contributing Guide

이 프로젝트에 기여해주셔서 감사합니다! 아래 규칙을 따라 협업을 진행해주세요.

**가장 중요한 규칙: `main`에는 직접 push하지 않고, 최소 1명 이상이 코드를 확인하고 승인(Approve)한 PR만 머지합니다.**

## 목차

- [브랜치 전략](#브랜치-전략)
- [커밋 메시지 컨벤션](#커밋-메시지-컨벤션)
- [Pull Request 규칙](#pull-request-규칙)
- [Issue 작성 규칙](#issue-작성-규칙)
- [코드 리뷰 규칙](#코드-리뷰-규칙)
- [코딩 컨벤션](#코딩-컨벤션)
- [프로젝트 전용 규칙](#프로젝트-전용-규칙)

## 브랜치 전략

### 브랜치 종류

작업 성격에 따라 아래 브랜치를 사용합니다.

| 브랜치 | 설명 |
|---|---|
| `main` | 항상 실행되는 안정 버전 (시연, 제출 기준) |
| `develop` | 다음 버전을 위한 개발 브랜치 |
| `feature/*` | 기능 개발 브랜치 |
| `fix/*` | 버그 수정 브랜치 |
| `hotfix/*` | 시연, 제출 직전 긴급 수정 브랜치 |
| `refactor/*` | 리팩토링 브랜치 |
| `docs/*` | 문서 작업 브랜치 |
| `chore/*` | 설정, 패키지 등 기타 작업 브랜치 |

### 브랜치 네이밍 규칙

```
<type>/<본인 이름>-<간단한-설명>
```

예시

- `feature/subin-nest-detection`
- `fix/subin-csv-write-conflict`
- `docs/subin-update-readme`

## 커밋 메시지 컨벤션

Conventional Commits 규칙을 따릅니다.

### 형식

```
<type>(<scope>): <subject>

<body>

<footer>
```

- subject: 50자 이내, 마침표 없이, 명령형으로 작성 (예: "추가하다" O, "추가함" X)
- body: 변경 이유와 이전과 달라진 점을 설명 (선택 사항, 72자 줄바꿈 권장)
- footer: 이슈 트래커 참조 (`Closes #123`, `Related to #45`)
- scope는 `dashboard`, `model`, `onboard`, `train`, `data` 중에서 고릅니다.

### Type 종류

커밋 종류는 아래 중 하나로 표시합니다.

| Type | 설명 |
|---|---|
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `style` | 코드 포맷팅 등 (로직 변경 없음) |
| `refactor` | 코드 리팩토링 (기능 변화 없음) |
| `test` | 테스트 코드 추가/수정 |
| `chore` | 패키지, 설정 등 |
| `perf` | 성능 개선 |
| `ci` | CI 설정 파일/스크립트 변경 |
| `revert` | 이전 커밋 되돌리기 |

### 커밋 예시

```
feat(model): 조류 둥지 클래스 추론 추가

균열/부식과 별도 모델로 분리해 로딩
detect() 반환 형식은 기존과 동일하게 유지

Closes #12
```

```
fix(dashboard): 결함 탐지 시 종합등급이 정상으로 뜨는 버그 수정
```

### 커밋 시 주의사항

- 하나의 커밋에는 하나의 논리적 변경사항만 담습니다.
- 커밋은 작고 자주 나눕니다 (기능 단위가 아닌 작업 단위).
- WIP(작업 중) 커밋은 PR 전에 `rebase -i`로 정리합니다.

## Pull Request 규칙

### PR 제목 형식

커밋 컨벤션과 동일하게 작성합니다.

```
<type>(<scope>): <설명>
```

예: `feat(model): 조류 둥지 클래스 추론 추가`

### PR 크기

- 한 PR은 가능한 500줄 이하로 작게 유지합니다.
- 리뷰가 어려울 정도로 크다면 여러 PR로 분리합니다.

### PR 절차

1. `develop`(또는 `main`)에서 작업 브랜치 생성
2. 작업 완료 후 원격 브랜치에 push
3. PR 템플릿(`.github/PULL_REQUEST_TEMPLATE.md`)에 맞춰 작성
4. **최소 1명 이상의 리뷰어 승인(Approve) 후 머지** (승인 없이 본인이 머지하지 않습니다)
5. 머지 방식은 Squash and Merge를 기본으로 함
6. 머지 후 작업 브랜치는 삭제

## Issue 작성 규칙

### Issue 제목 형식

```
[TYPE] 간단한 설명
```

예:

- `[BUG] 영상 파일 입력 시 FPS가 표시되지 않음`
- `[FEATURE] 탐지 기록 csv 내보내기`

### 라벨(Label) 규칙

이슈에는 아래 라벨을 붙입니다.

| 라벨 | 설명 |
|---|---|
| `bug` | 버그 리포트 |
| `feature` | 신규 기능 요청 |
| `enhancement` | 기존 기능 개선 |
| `documentation` | 문서 관련 |
| `question` | 질문 |
| `wontfix` | 처리하지 않을 이슈 |
| `duplicate` | 중복 이슈 |
| `priority: high/medium/low` | 우선순위 |

Issue 템플릿은 `.github/ISSUE_TEMPLATE/` 폴더의 `bug_report.md`, `feature_request.md`를 참고해주세요.

## 코드 리뷰 규칙

- 리뷰는 24시간 이내 응답을 원칙으로 합니다.
- 리뷰 코멘트는 구체적이고 건설적으로 작성합니다.
- Approve 전 반드시 로컬에서 동작을 확인합니다. 젯슨 관련 변경은 젯슨에서 확인합니다.
- 사소한 스타일 지적은 `nit:` 접두사를 붙입니다 (머지를 막지 않는 의견임을 표시).

```
nit: 변수명을 좀 더 명확하게 바꾸면 좋을 것 같아요.
```

## 코딩 컨벤션

- Python 코드는 PEP 8을 따르고, 포맷터는 `black`을 사용합니다.
- 커밋 전 반드시 로컬에서 실행해 동작을 확인합니다.
- 매직 넘버, 하드코딩된 값은 상수로 분리합니다 (예: 위험등급 임계값).
- 함수/변수명은 의미가 명확하게 드러나도록 작성합니다.

## 프로젝트 전용 규칙

### 인터페이스 변경

아래 두 가지는 모델 쪽과 대시보드 쪽이 서로 기대고 있는 약속이라 혼자 바꾸지 않습니다.

- `model_infer.py`의 `detect(img) -> dict` 반환 형식
- `detections.csv`의 11개 필드

바꿔야 하면 Issue를 열고 변경 전과 후의 실제 예시를 붙여서 합의합니다.

```json
{
  "변경 전": {"type": "균열", "area_pct": 0.42, "confidence": 0.87},
  "변경 후": {"type": "균열", "area_pct": 0.42, "confidence": 0.87, "mask_path": "caps/0012_mask.png"}
}
```

### 레포에 올리지 않는 것

- 가중치: `*.pt`, `*.onnx`, `*.engine`
- 데이터와 영상: `data/`, `caps/`, `*.mp4`, `*.mov`, `*.zip`
- 실행 중에 생기는 파일: `detections.csv`, `status.json`
- 비밀 값: `.env`, API 키, 토큰

GitHub은 100MB가 넘는 파일의 push를 거부합니다. 가중치와 데이터는 공유 드라이브에 올리고 README에 링크를 적습니다.

### 패키지

- 새 패키지를 쓰면 같은 PR에서 `requirements.txt`에 버전과 함께 추가합니다.
- 젯슨에서는 `sudo pip install`을 쓰지 않습니다. torch를 의존성으로 끌어오는 패키지는 `--no-deps`로 설치합니다.

## 질문이 있다면

Issue 또는 팀 채널을 통해 언제든 편하게 문의해주세요.
