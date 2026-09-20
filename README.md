# 안벽 크레인 균열/부식 진단 드론 플랫폼

항만 안벽 크레인의 균열과 부식을 드론과 엣지 AI로 점검하는 플랫폼임. 점검자가 크레인 위로 올라가지 않고도, 드론에 실린 Jetson Orin Nano가 영상을 현장에서 바로 추론하고 관제 대시보드에 점검 우선순위를 띄움.

- 사업: 2026 스마트해운물류 x ICT 멘토링 (해양수산부, 운영 한국정보산업연합회)
- 기간: 2026-06-01 ~ 2026-10-31
- 팀: 크랙캐쳐 (HP038)
- 팀원: 서은수, 정수빈, 정혜민
- 멘토: 박철훈

## 무엇을 하는가

1. 카메라 영상에서 균열과 부식을 픽셀 단위로 분할함 (세그멘테이션은 결함 영역을 픽셀 단위로 칠해 주는 방식임)
2. 결함이 화면에서 차지하는 면적 비율로 위험등급 4단계를 판정함
3. 판정 결과를 csv 하나에 기록하고, 대시보드가 그 csv를 읽어 실시간으로 표시함
4. 같은 코드가 노트북과 Jetson Orin Nano에서 모두 동작함

## 구조

입력 소스가 무엇이든 대시보드 코드는 바뀌지 않음. 모델 쪽과 화면 쪽은 `detections.csv` 하나로만 연결되며, 서버와 DB 없이 파일만으로 동작함.

```
입력(카메라 / 영상 파일 / 사진 / 화면 캡처)
        |
  model_infer.py   detect(img) -> dict, draw() -> 오버레이
        |
  detections.csv   탐지 기록 (11필드)
  status.json      조치 완료 여부
  caps/            탐지 시점 캡처 이미지
        |
  dashboard.py     Streamlit 관제 화면
```

## 폴더 구조

```
.
├── dashboard.py        관제 대시보드 (메인 실행 파일)
├── detect_run.py       사진 폴더 일괄 탐지 CLI
├── model_infer.py      모델 로딩과 추론
├── weights/            가중치 (git 제외, 아래 링크에서 받음)
├── caps/               캡처 이미지 (git 제외)
├── docs/               가이드와 기록 문서
├── requirements.txt
└── README.md
```

## 설치

Python 3.10 기준임.

```bash
git clone <레포 주소>
cd <레포 폴더>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

가중치는 레포에 없음. 아래에서 받아 `weights/`에 둠.

- 균열 모델 `best_E5_tversky.pt` (U-Net + EfficientNet-B0): (드라이브 링크 작성 필요)
- 부식 모델: (드라이브 링크 작성 필요)

## 실행

```bash
streamlit run dashboard.py
```

사이드바에서 입력 소스를 고르고 시작을 누름. 첫 모델 로딩에 20~30초가 걸림. 코드를 고친 뒤에는 브라우저 새로고침이 아니라 `Ctrl+C` 후 다시 실행해야 반영됨.

사진 폴더를 한 번에 돌릴 때는 아래 명령을 씀.

```bash
python detect_run.py <사진 폴더 경로>
```

### Jetson Orin Nano에서 실행

```bash
source ~/ICT_drone/bin/activate
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

같은 네트워크의 노트북에서 `http://<젯슨 IP>:8501`로 접속함. `Port 8501 is not available`이 뜨면 이전 프로세스가 남아 있는 것이므로 `pkill -f "streamlit run dashboard.py"` 후 다시 실행함.

Jetson에 패키지를 설치할 때는 아래 3가지를 지킴. JetPack에 맞춰 설치된 CUDA용 torch가 일반 torch로 덮어써지는 것을 막기 위함임.

- `sudo pip install` 금지
- torch나 torchvision을 의존성으로 끌어오는 패키지는 `--no-deps`로 설치
- 설치 후 매번 `python -c "import torch;print(torch.cuda.is_available())"`로 확인

## 위험등급 기준

등급은 결함 영역이 프레임에서 차지하는 면적 비율로 정함. 결함이 하나라도 잡히면 정상이 될 수 없고, 한 프레임에 결함이 여러 개면 가장 높은 등급이 종합등급이 됨.

| 등급 | 균열 | 부식 |
|---|---|---|
| 정상 | 결함 없음 | 결함 없음 |
| 경미 | 0.3% 미만 | 3% 미만 |
| 주의 | 0.3% 이상 1.5% 미만 | 3% 이상 12% 미만 |
| 위험 | 1.5% 이상 | 12% 이상 |

구현은 `dashboard.py`의 `grade_of()`와 `overall_of()`에 있음.

## detections.csv 형식

모델 쪽과 화면 쪽이 공유하는 유일한 계약임. 필드를 바꾸려면 CONTRIBUTING.md의 인터페이스 변경 절차를 따름.

| 필드 | 설명 |
|---|---|
| id | 탐지 번호 |
| timestamp | 탐지 시각 |
| source | 입력 소스 |
| type | 균열 / 부식 / 점검(결함 없음) |
| grade | 해당 결함의 등급 |
| area_pct | 면적 비율(%) |
| confidence | 신뢰도 |
| bbox | 결함 영역 박스 좌표 |
| image_path | 캡처 이미지 경로 |
| overall_grade | 해당 프레임의 종합등급 |
| status | 상태 |

조치 완료 여부는 csv가 아니라 `status.json`에 따로 저장함. 탐지 중에 csv를 동시에 쓰다가 충돌하는 것을 막기 위함임.

## 성능

Jetson Orin Nano 온보드 추론 기준임. (측정 조건과 표 작성 필요)

- 지연: 85.5ms
- 처리 속도: 11.7FPS

## 하드웨어

- 비행 제어기: SpeedyBee F405 (PX4)
- 온보드 컴퓨터: NVIDIA Jetson Orin Nano
- 카메라: (작성 필요)

## 문서

- `docs/` 폴더에 대시보드 가이드, 젯슨 온보드 구축 기록을 둠
- 협업 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md) 참고

## 라이선스

(작성 필요)
