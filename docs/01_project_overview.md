# 프로젝트 개요

## 1. 목표

한국인 얼굴 이미지에서 다음 7개 감정을 분류하는 이미지 분류 모델을 만드는 것이 목표다.

`기쁨 · 당황 · 분노 · 불안 · 상처 · 슬픔 · 중립`

프로젝트 범위는 원본 데이터 점검, 이미지와 라벨 전처리, GPU 학습 환경 구축, 세 모델 비교, 결과 시각화, 세 모델 체크포인트 저장과 팀원 전달까지다.

## 2. Training 데이터 범위 결정

감정별 Training 원천 데이터는 4개 파트로 나뉘어 있었다. 제한된 프로젝트 시간 안에 전처리와 세 모델 학습을 완료하기 위해 데이터 양을 약 절반으로 줄이기로 했다.

검토한 방법은 다음 두 가지였다.

1. 4개 파트를 모두 받은 뒤 각 파트의 70~80%를 무작위로 삭제
2. 4개 파트 중 2개 파트만 내려받아 사용

첫 번째 방법은 전체 데이터를 내려받고 다시 선별·삭제·검증해야 하므로 작업량과 처리 시간이 늘어난다. 무작위 삭제 결과를 재현하려면 삭제 목록이나 seed도 별도로 관리해야 한다. 따라서 파트 단위로 범위가 명확하고 추가 선별 작업이 적은 두 번째 방법을 선택했다.

라벨 JSON은 전체 파트 범위가 남아 있으므로 선택하지 않은 파트에 해당하는 Training 라벨 193,589건은 대응 원천 JPG가 없다. 이 항목은 전처리 오류나 다운로드 실패가 아니라 계획한 데이터 축소의 결과로 기록한다.

## 3. 전체 작업 흐름

```text
원본 JPG·JSON 수집
  → EDA와 품질 점검
  → 방향 보정·흑백화·224×224 PNG 변환
  → 라벨 filename·얼굴 박스 좌표 변환
  → PyTorch/CUDA 환경 검증
  → ResNet18·MobileNetV2·EfficientNet-B0 학습
  → Accuracy·Macro-F1·클래스별 F1 비교
  → 세 모델의 best.pt 보관
  → 선택 실행·비교 실행이 가능한 웹캠 프로그램에 전달
```

## 4. 폴더 구조

```text
emotion_project/
├─ .venv/                              이 PC의 Python 가상환경, Git 제외
├─ .vscode/                            팀 공용 VS Code 설정
├─ dataset/
│  ├─ raw/{Training,Validation}/       원본 JPG, Git 제외
│  ├─ labels/{Training,Validation}/    원본 JSON, Git 제외
│  └─ processed/
│     ├─ images/{Training,Validation}/ 224×224 흑백 PNG, Git 제외
│     ├─ labels/{Training,Validation}/ 변환된 JSON, Git 제외
│     └─ preprocessing_summary.json    전처리 결과와 오류 기록
├─ docs/                               프로젝트 단계별 문서
├─ models/                             세 모델의 best.pt 보관
├─ results/
│  ├─ preprocessing_eda_report.md      상세 EDA 보고서
│  ├─ experiments/                    모델별 수치와 history
│  └─ comparisons/all_three_models/   통합 비교표와 발표용 그림
├─ scripts/setup_cuda126.ps1           환경 자동 구성
├─ src/                                전처리·학습·시각화·이전 코드
├─ transfer/processed_dataset/         데이터 전달용 분할 ZIP, Git 제외
└─ requirements*.txt                  고정 패키지 버전
```

## 5. 코드별 책임

| 파일 | 책임 |
|---|---|
| `src/preprocess_data.py` | 원본 이미지를 전처리하고 JSON 라벨을 함께 변환 |
| `src/train_models.py` | 데이터를 불러와 모델을 학습하고 `best.pt` 저장 |
| `src/visualize_results.py` | 학습 기록으로 곡선·혼동행렬·비교표 생성 |
| `src/check_environment.py` | Python·패키지·CUDA와 실제 GPU 연산 검사 |
| `src/package_processed_dataset.py` | 대용량 데이터를 분할 ZIP과 해시로 포장 |
| `src/unpack_processed_dataset.py` | 분할 ZIP을 검증하고 다른 PC에 복원 |
| `scripts/setup_cuda126.ps1` | 가상환경과 CUDA 12.6용 PyTorch 설치 |

## 6. 주요 산출물

| 산출물 | 위치 | 용도 |
|---|---|---|
| 세 모델 체크포인트 | `models/GPU_*/<model>/best.pt` | 웹캠의 모델 선택·비교 기능 |
| 모델 비교표 | `results/comparisons/all_three_models/comparison_table.csv` | 정량 결과 확인 |
| 발표용 그래프 | `results/comparisons/all_three_models/*.png` | 모델 성능 발표 |
| 상세 EDA | `results/preprocessing_eda_report.md` | 데이터 분석 근거 |
| 전처리 요약 | `dataset/processed/preprocessing_summary.json` | 성공·실패·시간의 원본 기록 |

## 7. 현재 프로그램 방향

- EfficientNet-B0가 현재 Validation의 Accuracy와 Macro-F1에서 가장 높았다는 사실은 비교 결과로만 제시한다.
- ResNet18, MobileNetV2, EfficientNet-B0의 `best.pt`를 모두 프로그램에 포함한다.
- 일반 모드에서는 사용자가 선택한 모델 하나를 실행하고, 비교 모드에서는 세 모델의 감정 예측과 신뢰도·FPS·지연 시간을 함께 표시한다.
- 모델별 장단점을 실제 웹캠 환경에서 확인하되 하나의 모델만 최종 모델로 고정하지 않는다.

## 8. 평가 해석 원칙

- 한 사람이 여러 장의 유사한 이미지를 제공하는 데이터 구조를 고려한다.
- Training과 Validation에 공통으로 나타난 파일명 선두 식별자 528개가 같은 인물이나 촬영 세션을 뜻하는지는 데이터 명세로 확인해야 한다.
- 같은 인물의 이미지가 두 분할에 포함됐다면 모델이 감정 특징뿐 아니라 인물 특징을 학습해 현재 Validation 성능이 높게 측정됐을 수 있다.
- 현재 제공 분할 결과와 별도로 인물·촬영 세션 기준 `group split`을 수행해 새로운 사람에 대한 일반화 성능을 평가한다.
