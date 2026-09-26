# 신규 크레인 35장 파인튜닝 코드임

- 대상은 균열15장·부식15장·정상5장으로 사진 채택된 신규 크레인 자료임. 기존 9월23일 예비실험 자료와 구분함.
- 균열과 부식을 각각 U-Net/EfficientNet-B0 이진 분할 모델로 학습함. 기존 균열 E5와 9월26일 범용 부식 기본 가중치를 초기값으로 사용함. 가중치 누락 시 중단하며 ImageNet 신규 학습으로 바꾸지 않음.
- 기본 설정은 512px, 최대20epoch, batch2, lr3e-5, patience5, 임계값0.5임. 인코더와 인코더 BatchNorm을 고정함.
- 균열은 BCE+Tversky(0.3/0.7), 부식은 BCE+Dice 계열 손실을 사용함. 밝기·좌우반전·상하반전을 학습 자료에만 적용함.

## 입력과 검수 반영

`pending_manifest.json`은 현재35장 경로를 연결한 대기 명세임. 학습 승인이 아니며 원본 라벨을 수정하지 않음.

- 실제 검수 결과를 근거로 `annotation_reviewed`, `complete_targets`, `normal_confirmed`, `training_allowed`, `review_evidence`를 반영해야 함.
- `complete_targets`는 해당 사진에서 빠짐없이 검수한 결함명 목록임. 균열만 검수한 사진을 부식 모델의 정상 정답으로 사용하지 않음.
- `reviewed_image_sha256`, `reviewed_labelme_sha256`는 검수 당시 파일의 SHA256임. 검수 뒤 라벨이 바뀌면 재검수 연결이 필요함. 현재 해시를 복사하여 승인을 만들어내는 용도가 아님.
- 원본 JSON의 오래된 draft 플래그보다 해시로 묶은 별도 최종 검수 명세를 사용함. 사진 채택과 라벨 승인을 구분함.
- 허용 라벨은 `crack`, `corrosion`, `ignore`이며 polygon만 지원함. 애매한 영역은 ignore로 제외함. 정상 사진은 폴리곤0개 및 normal_confirmed=true로 명시함.
- 기존 자료에 남아 있는 권리·도메인·부분 라벨 보류 사유는 최종 학습 명세를 만들 때 실제 해결 상태를 반영할 사항임. 코드가 이를 임의로 승인하지 않음.

```bash
python prepare_data.py --manifest pending_manifest.json --audit-drafts
python prepare_data.py --manifest reviewed_manifest.json --out dataset
python train_finetune.py --task crack --data dataset --weights best_E5_tversky.pt --out runs/crack
python train_finetune.py --task corrosion --data dataset --weights best_corrosion_efficientnet-b0.pt --out runs/corrosion
```

## 분할과 결과

- 같은 source_group, 같은 original_sha256, 명시된 group_constraints를 묶어 분할함. 기본 검증 비율은20% 목표이나 그룹 크기에 따라 달라짐. 장수를 맞추려고 같은 현장 그룹을 나누지 않음.
- 각 분할에 균열·부식 양성 및 정상 자료가 있도록 정답 분포만으로 선택함. 성능을 보고 재분할하지 않음. 조건 충족이 불가능하면 중단함.
- 정상5장은 독립 촬영그룹이 적어 정상 검증의 대표성이 제한됨. 완전 독립 시험셋이 없는 예비실험임.
- 학습 전후 동일 검증 자료와 임계값에서 IoU·Dice·Recall·Precision, 정상 오탐 이미지 수와 오탐 면적비를 저장함. best는 양성 검증 macro IoU로 선택함.
- `results.json`, `history.json`, `config.json`, `best_*_finetuned.pt`, `baseline/`, `finetuned/`를 생성함. 비교 그림은 원본·정답·예측오류 순서이며 녹색=TP, 빨강=FN, 노랑=FP, 회색=ignore임.
- `review_candidate`는 IoU 개선·Recall 비감소·정상 오탐 면적 비증가를 충족한 검토 후보 표시임. 자동 배포나 안전 판정이 아님. 기존 배포 모델을 교체하지 않음.
- 전체 사진을512px로 줄이므로 가는 균열이 사라질 수 있음. 양성 소실은 오류로 중단함. 입력 해상도와 크롭 전략 변경 시 별도 실험으로 기록함.

## 검증

- `python -m unittest test_pipeline -v`로 마스크 중첩·ignore·지표·검수 차단·파일 변경·촬영그룹 분리를 검사함.
- 합성 소형 자료의 실제1epoch 실행은 코드 동작 확인이며 신규35장 성능 결과가 아님.

## API 근거

- [SMP U-Net 공식 문서](https://smp.readthedocs.io/en/docs/models.html)를 기준으로 모델 구성을 유지함.
- [PyTorch 저장·로딩 공식 문서](https://docs.pytorch.org/tutorials/beginner/saving_loading_models)를 기준으로 state_dict를 저장하고 weights_only=True로 불러옴.
