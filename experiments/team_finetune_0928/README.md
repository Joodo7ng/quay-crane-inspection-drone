# 팀 통합 학습과 평가 코드

작성일: 2026-09-28

팀원이 준비한 사진과 Labelme JSON을 공통 형식으로 검사하고, 기존 균열 모델과 부식 모델을 각각 추가 학습하는 코드임. 기존 가중치와 추가 학습 가중치를 같은 조건에서 비교하고 결과표를 생성하는 구성임.

현재 상태는 코드와 실행 절차 준비 완료임. 검수 완료한 팀 데이터 전체로 통합 학습하거나 최종 Test를 실행한 상태는 아님. `reports/코드통합_회의공유0928.md`에서 최신 구현 범위 및 입력 준비 상태 확인 가능함.

처음 확인하는 경우 [통합 코드 설명](통합코드_설명.md)에서 목적, 처리 순서와 결과 해석 확인 가능함.

## 바로 사용할 파일

- `팀통합_균열부식_학습평가0928.ipynb`: Colab 실행 순서
- `import_metadata.py`: 노션에서 합의한 metadata.csv를 검사 명세로 변환
- `data.py`: 중복, 분할, 검수 상태 검사와 공통 마스크 생성
- `engine.py`, `run.py`: 두 모델의 학습과 분리된 최종 평가
- `metrics.py`: IoU, Dice, Precision, Recall, 정상 오탐 계산
- `templates/metadata.csv`: 팀 전달 형식의 빈 양식
- `templates/crack_config.json`, `templates/corrosion_config.json`: 학습 설정
- `은수_평가기준과논문기록.md`: 평가 계산과 논문 3절 및 4절 작성 기준

## 입력 자료와 검수 기록

학습 및 검증용 ZIP과 Test용 ZIP을 별도로 해제하는 방식임. ZIP 안에 `images/`, `labelme/`, `metadata.csv`, `README.md` 포함 필요함. 자동 검사가 제출자의 실제 검수를 대신하지 않으며 사진채택과 라벨 승인을 구분함.

노션에서 합의한 `image_id`, `image_file`, `label_file`, `classes`, `split`, `data_type`, `source_group`, `source_url`, `prior_use`, `reviewed`, `notes` 열 지원함. 합성 데이터의 `defect_source_ids`, `background_source_id`도 지원함.

- `split`: `train`, `val`, `test` 중 하나로 직접 확정 필요함. 약 8:2 권장이며 이 코드는 기존 분할을 임의 변경하지 않음.
- `reviewed`: 라벨 검수 완료 시 `true`, 미완료 시 `false` 입력함. 사진 적합성은 기본적으로 같은 검수 기록을 따르며 별도 `image_reviewed`로 구분 가능함.
- `prior_use`: `none`, `train`, `val`, `train+val`, `model_selection` 중 하나 사용함. `사용 없음`도 허용함. 그 외 자유 서술은 보존하되 사용 이력 미확정으로 처리함.
- `complete_targets`: 이미지 전체에서 빠짐없이 검수한 클래스의 목록임. `crack|corrosion`처럼 입력 가능함. 미기재 시 `classes`에 표시한 결함만 완전 라벨로 취급함. 균열만 검수한 사진을 부식 모델의 정상으로 자동 사용하지 않음.
- `independence_reviewed`, `independence_evidence`: Test에 추가로 필요한 열임. 원본, 촬영 사례, 합성 결함 및 배경의 이전 사용 이력을 확인한 후 `true`와 근거 입력 필요함.
- 같은 결함 원본 또는 배경 ID는 팀 전체에서 동일하게 사용 필요함. 파일명 앞에 SB, ES, HM을 붙여도 원본 그룹 ID는 별도로 유지함.

변환 시 제공한 검수 근거와 현재 파일 해시를 기록함. 이는 팀원의 검수 완료 선언을 기록하는 절차이며, 변환 프로그램이 라벨을 다시 검수하거나 승인하는 의미는 아님. 이후 파일이 바뀌면 검사에서 중단함.

균열 및 부식 라벨명은 `crack`, `corrosion`임. 혜민 자료의 `corrosion_fair`, `corrosion_poor`, `corrosion_severe`는 원본 JSON을 보존하고 학습 마스크 생성 시에만 `corrosion`으로 합침. 이 변환이 등급의 정확성이나 라벨 검수 완료를 보장하지 않음. 직사각형과 번호가 붙은 임의 클래스는 자동 폴리곤 변환 없이 오류로 표시함.

원본 크기의 기본 최소 변은 512픽셀임. 9/28 사용자 지시에 따라 작은 사진도 결함이 뚜렷하고 라벨 승인이 완료된 경우 명시적 예외 지원함. 예외 사용에는 사진별 `low_resolution_exception=true`, `low_resolution_reason`과 실행 옵션 `--allow-reviewed-low-resolution`이 모두 필요함. Colab에서는 `ALLOW_REVIEWED_LOW_RESOLUTION=True`로 선택함. 옵션만 켜서 미검수 사진 전체를 허용하지 않음. 확대하여 기준을 우회하는 자료와 미승인 라벨은 계속 차단함. 소프트웨어 검사의 64픽셀 난수 배열은 실제 데이터와 별개임.

## 실행 순서

Python 실행 위치는 이 README가 있는 폴더 기준임. 아래 경로는 실제 Drive 경로로 변경하는 예시임.

```bash
python -m pip install -r requirements.txt

python import_metadata.py --metadata /data/subin_trainval/metadata.csv --out /data/subin_trainval/manifest_v1.json --review-evidence "수빈 검수결과 링크와 버전"
python import_metadata.py --metadata /data/eunsu_trainval/metadata.csv --out /data/eunsu_trainval/manifest_v1.json --review-evidence "은수 검수결과 링크와 버전"
python import_metadata.py --metadata /data/hyemin_trainval/metadata.csv --out /data/hyemin_trainval/manifest_v1.json --review-evidence "혜민 검수결과 링크와 버전"

python data.py audit --manifests /data/subin_trainval/manifest_v1.json /data/eunsu_trainval/manifest_v1.json /data/hyemin_trainval/manifest_v1.json --out /data/audit_trainval.json
python data.py prepare --manifests /data/subin_trainval/manifest_v1.json /data/eunsu_trainval/manifest_v1.json /data/hyemin_trainval/manifest_v1.json --out /data/prepared_v1

python run.py train --config templates/crack_config.json --out /results/crack_v1
python run.py train --config templates/corrosion_config.json --out /results/corrosion_v1
```

설정 파일의 `data`는 `/data/prepared_v1/trainval`처럼 준비된 trainval 폴더로 지정함. `weights`는 기존 학습 가중치의 실제 경로임. 가중치 누락이나 모델 구조 불일치 시 중단하며 새 모델로 자동 대체하지 않음.

균열 기준 가중치는 기존 `best_E5_tversky.pt`임. 부식 기준 가중치는 9/26 CorrosionCS v2로 재학습한 `best_corrosion_efficientnet-b0.pt`임. 35장 추가 학습 후 가중치와 혼동하지 않도록 해시 기록 필요함. 현재 로컬에서 부식 기준 가중치 파일을 확인하지 못했으므로 Colab 또는 Drive의 기존 경로 확인 필요함.

Test 준비는 같은 import 및 prepare 과정을 Test 자료에 별도로 적용함. 결과는 `test_sealed/`로 내보내며 학습 실행에는 전달하지 않음. 학습 자료와 Test를 함께 검사할 수 있는 경우 `data.py audit`에 모든 manifest를 전달하여 팀 전체 원본 중복 확인 가능함. 최종 평가 직전에도 실행에 사용한 trainval 원본 기록과 Test를 다시 대조함.

## 모델과 기준 확정 후 Test 실행

Validation 결과에서 학습 전후 IoU, Precision, Recall 및 정상 오탐을 검토한 뒤 평가 대상 실행을 확정하는 순서임. 최고 Validation IoU 가중치를 저장하지만 자동 배포 또는 기존 모델 교체는 수행하지 않음.

```bash
python run.py lock --run /results/crack_v1 --note "Validation 검토 후 평가 대상으로 확정한 근거"
python run.py lock --run /results/corrosion_v1 --note "Validation 검토 후 평가 대상으로 확정한 근거"

python run.py test --run /results/crack_v1 --data /data/prepared_test_v1/test_sealed --out /results/crack_test_v1
python run.py test --run /results/corrosion_v1 --data /data/prepared_test_v1/test_sealed --out /results/corrosion_test_v1
```

Test 실행은 학습 때의 입력 크기, 임계값과 정상 오탐 기준을 그대로 사용함. 별도 임계값 지정 기능 없음. 같은 실행 폴더의 반복 Test 실행은 차단함. 파일 복사나 기록 삭제까지 막는 보안 장치는 아니므로 팀 차원의 Test 비공개 원칙 유지 필요함. 기술적 실패 시 실패 기록과 이미 본 결과 범위를 먼저 확인하고 복구 필요함.

## 출력 자료

- `config.json`: 실제 설정, 라이브러리 버전, 코드 및 기준 가중치 해시
- `trainval_provenance.json`: 사용한 원본 그룹 및 이전 사용 이력
- `history.json`: Validation 기준 epoch별 변화
- `best.pt`: 추가 학습 가중치, 기존 가중치와 별도 저장
- `baseline_val.json`, `finetuned_val.json`: 학습 전후 Validation 세부 점수
- `summary.csv`, `per_image.csv`, `결과표.md`: 논문 표와 사진별 계산 근거
- `baseline/`, `finetuned/`: 원본, 정답, 예측 비교 그림. 초록은 겹친 정답, 빨강은 놓친 영역, 노랑은 잘못 검출한 영역임.

Test 실행 폴더에는 별도 Test 결과와 동일 형식의 표를 생성함. 실제 사진과 합성 사진은 분리 집계하며 평가 사진이 없는 집단의 점수는 `해당 없음`으로 표시함.

## 이번 버전의 범위

기본 U-Net EfficientNet-B0 이진 모델과 공통 추가 학습 흐름의 통합임. 혜민 개별 실험의 replay, 조각 확대, LOPO 및 새 구조 인식은 이번 기본 설정에 자동 추가하지 않음. 후속 조건 비교가 필요한 경우 Validation에서만 비교한 뒤 최종 Test 이전에 결정 필요함. 픽셀 단위 평가를 사물 탐지 mAP로 대체하지 않음.

분할 검사는 기록된 원본 그룹과 합성 원본, 파일 해시 및 복호화된 픽셀의 중복을 검사함. 기록되지 않은 유사 촬영 장면이나 크롭 관계를 자동 증명하는 기능은 없으므로 출처 검수 필요함.

## 코드 검증

```bash
python -m unittest discover -s tests -v
```

분할 누출, 미승인 자료, 라벨 변경, 잘못된 정상 라벨, 직사각형 라벨, 지표 계산 및 학습부터 최종 평가까지의 실행 검사 포함함. 임시 난수 이미지와 임시 모델로 수행하는 검사이며 논문 성능 수치에 포함하지 않음. 기존 균열 가중치도 별도로 불러와 입력 및 출력 형식 확인함.

## 2026-09-28 밤 보완 및 검증

승인된 작은 사진의 명시적 예외와 metadata의 불리언 변환 보완함. 소프트웨어 검사 11개 통과함. 수빈 선별본 139장의 CSV 직접 입력 및 Colab용 변환 경로 검사 통과함. 실제 팀 데이터 학습 및 최종 Test 미실행 상태 유지함.
