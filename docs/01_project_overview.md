# 프로젝트 개요

## 1. 목표

한국인 얼굴 이미지에서 다음 7개 감정을 분류하는 이미지 분류 모델을 만드는 것이 목표다.

`기쁨 · 당황 · 분노 · 불안 · 상처 · 슬픔 · 중립`

프로젝트 범위는 원본 데이터 점검, 이미지와 라벨 전처리, GPU 학습 환경 구축, 세 모델 비교, 결과 시각화, 최종 모델 저장과 팀원 전달까지다.

## 2. 전체 작업 흐름

```text
원본 JPG·JSON 수집
  → EDA와 품질 점검
  → 방향 보정·흑백화·224×224 PNG 변환
  → 라벨 filename·얼굴 박스 좌표 변환
  → PyTorch/CUDA 환경 검증
  → ResNet18·MobileNetV2·EfficientNet-B0 학습
  → Accuracy·Macro-F1·클래스별 F1 비교
  → EfficientNet-B0 best.pt 선정
  → 결과 시각화 및 팀원에게 모델 전달
```

## 3. 폴더 구조

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
├─ models/                             최종 best.pt 보관
├─ results/
│  ├─ preprocessing_eda_report.md      상세 EDA 보고서
│  ├─ experiments/                    모델별 수치와 history
│  └─ comparisons/all_three_models/   통합 비교표와 발표용 그림
├─ scripts/setup_cuda126.ps1           환경 자동 구성
├─ src/                                전처리·학습·시각화·이전 코드
├─ transfer/processed_dataset/         데이터 전달용 분할 ZIP, Git 제외
└─ requirements*.txt                  고정 패키지 버전
```

## 4. 코드별 책임

| 파일 | 책임 |
|---|---|
| `src/preprocess_data.py` | 원본 이미지를 전처리하고 JSON 라벨을 함께 변환 |
| `src/train_models.py` | 데이터를 불러와 모델을 학습하고 `best.pt` 저장 |
| `src/visualize_results.py` | 학습 기록으로 곡선·혼동행렬·비교표 생성 |
| `src/check_environment.py` | Python·패키지·CUDA와 실제 GPU 연산 검사 |
| `src/package_processed_dataset.py` | 대용량 데이터를 분할 ZIP과 해시로 포장 |
| `src/unpack_processed_dataset.py` | 분할 ZIP을 검증하고 다른 PC에 복원 |
| `scripts/setup_cuda126.ps1` | 가상환경과 CUDA 12.6용 PyTorch 설치 |

## 5. 최종 산출물

| 산출물 | 위치 | 용도 |
|---|---|---|
| 최종 모델 | `models/GPU_efficientnet_b0_e50_p3_b32/efficientnet_b0/best.pt` | 웹캠·추론 프로그램 개발 |
| 모델 비교표 | `results/comparisons/all_three_models/comparison_table.csv` | 정량 결과 확인 |
| 발표용 그래프 | `results/comparisons/all_three_models/*.png` | 모델 성능 발표 |
| 상세 EDA | `results/preprocessing_eda_report.md` | 데이터 분석 근거 |
| 전처리 요약 | `dataset/processed/preprocessing_summary.json` | 성공·실패·시간의 원본 기록 |

## 6. 현재 결론

- EfficientNet-B0가 Accuracy 0.7400, Macro-F1 0.7357로 가장 높다.
- MobileNetV2는 1위보다 Macro-F1이 약 0.0062 낮지만 모델 크기와 학습 시간이 작아 실시간 프로그램 후보로 가치가 있다.
- 최종 웹캠 프로그램은 EfficientNet-B0부터 적용하고, 실제 FPS가 부족하면 MobileNetV2와 지연 시간을 비교하는 방향이 적절하다.
- 각 모델들을 전부 사용하여 웹캠에서 모델 선택을 가능하게 하는 프로그램으로 구성할 예정이다.

