# Emotion Project

한국인 얼굴 이미지로 `기쁨·당황·분노·불안·상처·슬픔·중립` 7개 감정을 분류하는 프로젝트다. 원본은 보존하고, 학습에는 EXIF 방향을 보정한 224×224 흑백 PNG를 사용한다.

## 현재 데이터

- 전처리 성공: Training 223,570장 + Validation 52,118장 = 275,688장
- 손상 제외: 16장
- 전처리 시간: 1시간 35분 35초
- 상세 EDA와 발표 근거: `results/preprocessing_eda_report.md`
- 실행 원본 기록: `dataset/processed/preprocessing_summary.json`

## 폴더 구조

```text
dataset/
  raw/{Training,Validation}/          원본 JPG
  labels/{Training,Validation}/       원본 JSON
  processed/
    images/{Training,Validation}/     224×224 흑백 PNG
    labels/{Training,Validation}/     변환된 JSON
    preprocessing_summary.json
src/
  preprocess_data.py
  train_models.py
  check_environment.py
  package_processed_dataset.py
  unpack_processed_dataset.py
scripts/setup_cuda126.ps1
results/preprocessing_eda_report.md
```

## 전처리 재실행

```powershell
.\.venv\Scripts\python.exe .\src\preprocess_data.py
```

기존 정상 PNG는 건너뛰므로 중단 후 같은 명령으로 이어서 실행할 수 있다. 원본 파일은 수정하지 않는다.

## 모델 비교

```powershell
.\.venv\Scripts\python.exe .\src\train_models.py
```

기본 비교 모델은 ResNet-18, EfficientNet-B0, MobileNetV3-Small이며 CUDA가 인식되면 자동으로 GPU를 사용한다.

### 장시간 학습 권장 명령

1 epoch 시험이 끝났다면 최대 epoch를 50으로 두고 macro-F1이 7회 연속 개선되지 않을 때 조기 종료하는 방식을 권장한다.

```powershell
.\.venv\Scripts\python.exe .\src\train_models.py `
  --models resnet18 `
  --epochs 50 `
  --patience 7 `
  --batch-size 32 `
  --run-name pc_main_resnet18_e50
```

현재 PC의 1 epoch 실측 시간이 약 40.8분이므로 조기 종료가 없다면 30 epoch는 약 20시간, 50 epoch는 약 34시간이 걸릴 수 있다. 검증 macro-F1이 가장 높은 모델은 `best.pt`, 매 epoch의 재개용 상태는 `last.pt`에 저장된다.

중단 후에는 최초 명령과 같은 옵션에 `--resume`만 추가한다.

```powershell
.\.venv\Scripts\python.exe .\src\train_models.py `
  --models resnet18 `
  --epochs 50 `
  --patience 7 `
  --batch-size 32 `
  --run-name pc_main_resnet18_e50 `
  --resume
```

저장 구조는 다음과 같다.

```text
models/<run-name>/<model>/
  best.pt             최고 macro-F1 추론용 모델
  last.pt             중단 재개용 모델·optimizer 상태
  history.json        epoch별 지표
  run_summary.json    최고 성능·GPU·환경 정보

results/experiments/<run-name>/
  model_comparison.csv
  <model>/history.json
  <model>/run_summary.json
  figures/            학습 곡선·클래스별 F1·혼동행렬·비교표
```

체크포인트는 크기 때문에 GitHub에서 제외되고, `results/experiments`의 작은 JSON·CSV·PNG는 팀 비교를 위해 GitHub에 올릴 수 있다.

### 여러 PC 결과 비교

각 PC가 겹치지 않는 `--run-name`을 사용한다. 예: `pc_a_resnet18_e50`, `pc_b_efficientnet_e50`, `pc_c_mobilenet_e50`.

각 PC에서 학습 후 결과를 공유한다.

```powershell
git add results
git commit -m "Add training result pc_b_efficientnet_e50"
git pull --rebase
git push
```

모든 결과를 한 PC에 `git pull`한 뒤 최종 발표 그래프를 다시 만든다.

```powershell
.\.venv\Scripts\python.exe .\src\visualize_results.py `
  .\results\experiments `
  --output-dir .\results\final_comparison
```

모델은 Accuracy만으로 고르지 않고 1순위 macro-F1, 2순위 Accuracy, 클래스별 F1과 혼동행렬 순으로 확인한다.

## GitHub로 팀 코드 공유

이 저장소에는 코드·설정·보고서만 올린다. 얼굴 원본, 라벨, 전처리 이미지, 모델 가중치와 ZIP은 `.gitignore`로 제외된다. 데이터 라이선스와 개인정보 문제 때문에 GitHub 저장소는 `Private`으로 만드는 것을 권장한다.

### 처음 올리는 PC

1. GitHub 웹사이트에서 `emotion_project`라는 새 **Private** 저장소를 만든다.
2. GitHub의 README, `.gitignore`, License 자동 생성 옵션은 선택하지 않는다.
3. PowerShell에서 아래 명령을 실행한다. `<사용자명>`은 실제 GitHub 사용자명으로 바꾼다.

```powershell
cd D:\emotion_project
git init
git branch -M main
git add .
git status
git commit -m "Initial emotion project"
git remote add origin https://github.com/<사용자명>/emotion_project.git
git push -u origin main
```

`git status`에서 `dataset`, `.venv`, `transfer`, `models`가 올라갈 목록에 없는지 반드시 확인한 후 commit한다. 첫 commit에서 사용자 정보 오류가 나면 다음을 한 번 실행한다.

```powershell
git config --global user.name "GitHub 표시 이름"
git config --global user.email "GitHub 이메일"
```

### 다른 PC에서 내려받기

먼저 Git과 VS Code User Setup을 설치한 후 PowerShell을 새로 연다.

```powershell
cd D:\
git clone https://github.com/<사용자명>/emotion_project.git
cd emotion_project
code .
```

VS Code가 열리면 추천 확장 설치 알림에서 Python, Pylance, Jupyter를 설치한다. 직접 설치하려면 왼쪽 Extensions에서 각각 검색하면 된다.

## 다른 Windows PC에서 같은 Python 환경 구성

`.venv`는 절대 경로를 포함하므로 복사하지 않고 새 PC에서 다시 만든다. NVIDIA 드라이버와 Python 3.12.10 64-bit 설치 후 프로젝트 루트에서 실행한다.

```powershell
nvidia-smi
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_cuda126.ps1
```

고정 환경은 `torch 2.13.0+cu126`, `torchvision 0.28.0+cu126`, `Pillow 12.3.0`이다. 설치 후 실제 CUDA 행렬 연산까지 자동 검사한다. VS Code 인터프리터는 `.venv\Scripts\python.exe`를 선택한다.

VS Code에서 인터프리터를 수동 선택할 때는 `Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv\Scripts\python.exe` 순서로 선택한다.

## 13.13 GiB 데이터 옮기는 권장 방법

GitHub에는 데이터셋을 올리지 않는다. 현재 전처리 결과는 13.13 GiB이며 PNG는 이미 압축된 형식이라 ZIP으로 묶어도 크기가 크게 줄지 않는다.

### 방법 A: 외장 SSD — 가장 빠르고 권장

원래 PC에서 다음 명령으로 3.5 GiB 이하 ZIP 여러 개와 SHA-256 manifest를 만든다.

```powershell
cd D:\emotion_project
.\.venv\Scripts\python.exe .\src\package_processed_dataset.py
```

1. `D:\emotion_project\transfer\processed_dataset` 폴더 전체를 외장 SSD에 복사한다.
2. 다른 PC에서 GitHub로 받은 프로젝트의 `transfer\processed_dataset` 위치로 폴더 전체를 복사한다.
3. 다음 명령으로 손상 여부를 검사하면서 복원한다.

```powershell
.\.venv\Scripts\python.exe .\src\unpack_processed_dataset.py
```

외장 저장장치가 FAT32여도 각 ZIP이 4 GiB보다 작아 복사할 수 있다. ZIP 일부나 `transfer_manifest.json` 하나라도 빠지면 복원이 중단된다.

### 방법 B: Google Drive·OneDrive 등 클라우드 드라이브

1. 방법 A와 같은 패키징 명령을 실행한다.
2. 브라우저보다 재시작·이어받기가 쉬운 데스크톱 동기화 앱을 설치한다.
3. `transfer\processed_dataset` 폴더 전체를 비공개 공유 폴더에 복사하고 동기화 완료 표시를 기다린다.
4. 다른 PC에서 폴더 전체를 내려받아 프로젝트의 `transfer\processed_dataset`에 둔다.
5. `unpack_processed_dataset.py`를 실행한다. SHA-256 불일치가 나오면 해당 ZIP만 다시 내려받는다.

얼굴 이미지 데이터이므로 공개 링크보다는 팀원만 접근 가능한 비공개 공유를 사용하고, 원 배포처의 재배포 허용 범위를 먼저 확인한다.

### 방법 C: 두 PC가 같은 네트워크에 있을 때

Windows에서 대상 PC의 공유 폴더를 만든 뒤 원래 PC에서 재시작 가능한 복사를 사용할 수 있다.

```powershell
robocopy D:\emotion_project\transfer\processed_dataset \\대상PC이름\공유폴더\processed_dataset /E /Z /J /R:3 /W:5
```

복사 후에도 반드시 다른 PC에서 `unpack_processed_dataset.py`를 실행해 해시를 확인한다.

## 다른 PC에서 모델 시험

먼저 환경과 데이터가 정상인지 확인한다.

```powershell
.\.venv\Scripts\python.exe .\src\check_environment.py
```

가장 가벼운 모델로 1 epoch 시험한다.

```powershell
.\.venv\Scripts\python.exe .\src\train_models.py --models mobilenet_v3_small --epochs 1 --batch-size 32
```

정상 완료되면 컴퓨터별로 모델을 나누어 실행할 수 있다.

```powershell
# 컴퓨터 A
.\.venv\Scripts\python.exe .\src\train_models.py --models resnet18

# 컴퓨터 B
.\.venv\Scripts\python.exe .\src\train_models.py --models efficientnet_b0

# 컴퓨터 C 또는 추가 실험
.\.venv\Scripts\python.exe .\src\train_models.py --models mobilenet_v3_small
```

CUDA 메모리 부족 오류가 나면 `--batch-size 32`를 `16` 또는 `8`로 낮춘다. 각 컴퓨터의 `models` 결과 폴더는 GitHub에 자동으로 올라가지 않으므로, 최종 지표 JSON/CSV처럼 작은 결과만 별도로 `results`에 복사해 commit한다.
