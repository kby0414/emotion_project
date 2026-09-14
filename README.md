# Emotion Project

한국인 얼굴 이미지로 `기쁨·당황·분노·불안·상처·슬픔·중립` 7개 감정을 분류하는 프로젝트입니다. 원본 JPG와 라벨 JSON을 보존하고, 학습에는 EXIF 방향을 보정한 `224×224` 흑백 PNG를 사용합니다.

Training 원천 데이터는 시간과 연산 자원을 고려해 감정별 4개 파트 중 2개 파트만 사용했습니다. 전체 파트를 받은 뒤 각 파트의 70~80%를 무작위로 삭제하는 방법도 검토했지만, 다운로드·선별·검증 작업이 늘고 표본 재현도 어려워질 수 있어 파트 단위 선택을 사용했습니다. 라벨 JSON은 전체 범위가 남아 있어 선택하지 않은 파트에 해당하는 193,589건은 대응 JPG가 없습니다. 이는 전처리 실패가 아니라 프로젝트 범위를 의도적으로 줄인 결과입니다.

또한 한 사람이 여러 장의 유사한 표정 이미지를 제공하는 데이터 구조이므로, 같은 인물이나 촬영 세션이 Training과 Validation에 함께 들어가면 검증 성능이 높게 측정될 수 있습니다. 현재 결과는 제공 Validation 기준이며, 새로운 사람에 대한 일반화 성능은 인물·촬영 세션 기준 `group split`으로 추가 검증해야 합니다.

## 실험 결과와 프로그램 구성

| 순위 | 모델 | Accuracy | Macro-F1 | 파라미터 |
|---:|---|---:|---:|---:|
| 1 | EfficientNet-B0 | 0.7400 | 0.7357 | 4.02M |
| 2 | MobileNetV2 | 0.7309 | 0.7296 | 2.23M |
| 3 | ResNet18 | 0.7275 | 0.7253 | 11.18M |

- 현재 Validation에서는 `EfficientNet-B0`가 두 지표 모두 가장 높지만 단일 최종 모델로 고정하지 않습니다.
- 웹캠 프로그램은 `ResNet18`, `MobileNetV2`, `EfficientNet-B0`를 모두 탑재하고 선택 실행과 비교 실행을 지원합니다.
- 세 모델은 동일한 Training 223,570장과 Validation 52,118장으로 평가했습니다.
- 통합 그래프와 비교표는 `results/comparisons/all_three_models/`에 있습니다.

## 폴더 구조

```text
emotion_project/
├─ .venv/                              Python 3.12 로컬 가상환경(Git 제외)
├─ .vscode/                            팀 공용 VS Code 설정
├─ dataset/
│  ├─ raw/{Training,Validation}/       원본 JPG
│  ├─ labels/{Training,Validation}/    원본 JSON 라벨
│  └─ processed/
│     ├─ images/{Training,Validation}/ 224×224 흑백 PNG
│     ├─ labels/{Training,Validation}/ 정리된 라벨
│     └─ preprocessing_summary.json    전처리 실행 요약
├─ models/
│  ├─ GPU_efficientnet_b0_e50_p3_b32/efficientnet_b0/best.pt
│  ├─ GPU_mobilenet_v2_e50_p3_b32/mobilenet_v2/best.pt
│  └─ GPU_resnet18_e50_p3_b32/resnet18/best.pt
├─ results/
│  ├─ preprocessing_eda_report.md      전처리·EDA 발표 근거
│  ├─ experiments/                    모델별 history와 run_summary
│  └─ comparisons/all_three_models/   3모델 통합 표·그래프
├─ scripts/
│  └─ setup_cuda126.ps1               Python/CUDA 가상환경 구성
├─ src/
│  ├─ preprocess_data.py              이미지·라벨 전처리
│  ├─ train_models.py                 세 모델 학습과 체크포인트 저장
│  ├─ visualize_results.py            학습 결과 비교 시각화
│  ├─ check_environment.py            Python/PyTorch/CUDA 검사
│  ├─ package_processed_dataset.py    데이터 전송 ZIP 생성
│  └─ unpack_processed_dataset.py     전송 ZIP 검증·복원
├─ transfer/processed_dataset/        데이터 전달용 분할 ZIP(Git 제외)
├─ requirements.txt                  공통 패키지 진입점
├─ requirements-common.lock.txt      공통 패키지 고정 버전
└─ requirements-cu126.lock.txt       CUDA 12.6 PyTorch 고정 버전
```

## 핵심 파일

### 단계별 문서

환경 구성, 전처리, 모델 학습, 결과 평가, 데이터 이전, GitHub 협업과 발표 Q&A는 `docs/README.md`에서 순서대로 확인할 수 있습니다.

### 프로그램 개발용 모델

```text
models/GPU_efficientnet_b0_e50_p3_b32/efficientnet_b0/best.pt
models/GPU_mobilenet_v2_e50_p3_b32/mobilenet_v2/best.pt
models/GPU_resnet18_e50_p3_b32/resnet18/best.pt
```

각 `best.pt`에는 모델 이름, 가중치, 클래스 순서, 입력 크기, 정규화 값과 최고 검증 지표가 들어 있습니다. 프로그램은 선택한 모델만 실행하는 일반 모드와 세 모델 결과를 함께 보여 주는 비교 모드를 지원하도록 구성합니다. `last.pt`는 학습 재개용이므로 정리 과정에서 제거했습니다.

프로그램 담당자에게는 세 모델의 `best.pt`가 모두 필요합니다. GitHub로 공유할 때는 세 파일을 Git LFS 대상으로 추가해야 하며, 현재 `.gitignore`가 모델 폴더를 제외하므로 명시적으로 추가하거나 예외 규칙을 설정해야 합니다.

GitHub에서 모델까지 내려받으려면 Git LFS가 필요합니다.

```powershell
git lfs install
git pull origin main
git lfs pull
```

### 발표용 결과

- `results/comparisons/all_three_models/model_comparison.png`
- `results/comparisons/all_three_models/per_class_f1.png`
- `results/comparisons/all_three_models/comparison_table.csv`
- `results/comparisons/all_three_models/presentation_summary.md`
- 모델별 학습곡선과 혼동행렬 PNG

## 주요 실행 명령

### 환경 구성

```powershell
nvidia-smi
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_cuda126.ps1
```

### 전처리

```powershell
.\.venv\Scripts\python.exe .\src\preprocess_data.py
```

이미 처리된 정상 파일은 건너뛰므로 중단 후 같은 명령으로 이어서 실행할 수 있습니다.

### 모델 학습 예시

```powershell
.\.venv\Scripts\python.exe .\src\train_models.py `
  --models efficientnet_b0 `
  --epochs 50 `
  --patience 3 `
  --batch-size 32 `
  --run-name GPU_efficientnet_b0_e50_p3_b32
```

기본 비교 모델은 `resnet18`, `efficientnet_b0`, `mobilenet_v2`입니다. CUDA를 사용할 수 있으면 자동으로 GPU를 선택합니다.

### 통합 비교 그래프 재생성

```powershell
.\.venv\Scripts\python.exe .\src\visualize_results.py `
  .\results\experiments\GPU_resnet18_e50_p3_b32 `
  .\results\experiments\GPU_mobilenet_v2_e50_p3_b32 `
  .\results\experiments\GPU_efficientnet_b0_e50_p3_b32 `
  --output-dir .\results\comparisons\all_three_models
```

## 보관 원칙

- `dataset/raw`, `dataset/labels`, `dataset/processed`는 개인정보와 라이선스 때문에 GitHub에 올리지 않습니다.
- `transfer/processed_dataset`은 다른 PC로 데이터를 전달할 때만 사용하며 GitHub에서 제외됩니다.
- 모델별 `best.pt`를 모두 보관하고, 학습 재개용 `last.pt`는 정리 후 제거합니다.
- 모델별 수치 기록은 `results/experiments`, 발표용 통합 그림은 `results/comparisons/all_three_models`에서 관리합니다.
