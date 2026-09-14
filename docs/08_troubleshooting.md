# 오류 대응 기록

## 1. 손상 이미지 처리 실패

### 증상

```text
이미지를 열거나 저장하지 못했습니다
OSError: broken data stream when reading image file
```

### 원인과 대응

한글 경로나 파일명 문제가 아니라 JPEG 내부 픽셀 스트림 손상이다. 최종 전수 처리에서 Training 8장, Validation 8장, 총 16장이 제외됐다. 기본 설정은 손상 파일을 기록하고 나머지를 계속 처리한다. 오류 경로와 원인은 `preprocessing_summary.json`의 `failed_images`에서 확인한다.

손상 데이터 비율이 약 0.0058%로 매우 작으므로 강제 복원보다 제외가 합리적이다. `--strict-images`를 사용하면 하나라도 실패할 때 전체 작업을 오류로 종료하므로 최종 대량 처리에는 기본 비엄격 모드를 사용했다.

## 2. 전처리 중 재부팅 또는 강제 종료

같은 명령을 다시 실행한다.

```powershell
.\.venv\Scripts\python.exe .\src\preprocess_data.py
```

완성된 PNG는 건너뛰고 남은 파일을 처리한다. PNG와 JSON은 `.tmp` 임시 파일에 완성한 뒤 최종 파일명으로 바꾸므로 중간 파일이 정상 결과로 오인될 가능성을 낮췄다.

`dataset/processed/preprocessing_summary.json`은 모든 요청 split이 끝난 뒤 생성된다. 이 파일이 없으면 전체 완료로 판단하지 않는다.

## 3. 추가 하위 폴더 때문에 이미지를 못 찾는 문제

`raw/Training/EMOIMG_기쁨_TRAIN_01/...`처럼 폴더가 한 단계 더 있어도 현재 코드는 재귀 탐색한다. 원본 폴더 구조를 억지로 평탄화할 필요가 없다. 단, 같은 파일명이 여러 폴더에 중복되면 라벨 매칭이 모호해지므로 코드가 충돌 오류로 중단한다.

## 4. 이미지가 거꾸로 또는 옆으로 보이는 문제

원본 파일은 픽셀 자체가 회전되어 있고 EXIF 태그를 해석하는 뷰어만 정상 방향으로 보여 줄 수 있다. `ImageOps.exif_transpose()`로 EXIF 방향을 실제 픽셀에 반영한 뒤 PNG로 저장한다. 최종 데이터에서 83,308장이 보정됐다.

## 5. CUDA가 아니라 CPU로 잡히는 문제

순서대로 확인한다.

```powershell
nvidia-smi
.\.venv\Scripts\python.exe .\src\check_environment.py
```

주요 원인은 다음과 같다.

- VS Code가 `.venv`가 아닌 다른 Python을 선택함
- CPU 전용 PyTorch가 설치됨
- CUDA 12.6용 wheel이 아닌 다른 빌드가 설치됨
- NVIDIA 드라이버가 GPU를 인식하지 못함
- 설치 후 VS Code 터미널을 다시 열지 않음

다음 값을 함께 확인해야 한다.

```python
torch.__version__
torch.version.cuda
torch.cuda.is_available()
```

`nvidia-smi`의 CUDA 표시는 드라이버 지원 버전이고 `torch.version.cuda`는 현재 PyTorch wheel의 런타임 버전이다.

## 6. matplotlib 모듈 오류

### 증상

```text
ModuleNotFoundError: No module named 'matplotlib'
```

### 대응

현재 선택된 가상환경에 공통 패키지를 설치한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

학습 자체가 이미 끝났다면 다시 학습할 필요 없이 시각화 코드만 실행한다.

```powershell
.\.venv\Scripts\python.exe .\src\visualize_results.py `
  .\results\experiments `
  --output-dir .\results\comparisons\all_three_models
```

## 7. Git 명령을 찾지 못하는 문제

```powershell
git --version
where.exe git
Test-Path "C:\Program Files\Git\cmd\git.exe"
```

설치 파일이 존재하면 현재 PowerShell PATH에 추가할 수 있다.

```powershell
$gitPath = "C:\Program Files\Git\cmd"
$env:Path = "$gitPath;$env:Path"
git --version
```

이후 VS Code를 재실행해 영구 PATH를 다시 읽게 한다.

## 8. 학습 중단 후 재개

학습 폴더에 `last.pt`가 남아 있고 최초 실행과 같은 `--run-name`, `--epochs`를 사용할 때만 재개한다.

```powershell
.\.venv\Scripts\python.exe .\src\train_models.py `
  --models efficientnet_b0 `
  --epochs 50 `
  --run-name 기존_실험_이름 `
  --resume
```

현재 정리된 세 모델 폴더에서는 `last.pt`를 제거했으므로 기존 실험을 재개할 수 없다. 추가 학습은 새 실험 이름으로 시작하고 결과를 별도로 비교한다.

## 9. 조기 종료가 50 epoch 이전에 발생

오류가 아니다. `patience=3`이면 검증 Macro-F1이 3회 연속 개선되지 않을 때 학습을 멈춘다. 가장 좋았던 epoch의 가중치는 이미 `best.pt`에 저장된다.

## 10. 다른 PC에서 ZIP 검증 실패

SHA-256 불일치는 전송 중 파일이 손상됐거나 다른 버전의 ZIP이 섞였다는 뜻이다. 학습을 진행하지 말고 manifest와 ZIP 4개를 같은 묶음에서 다시 복사한다. 파일명을 바꾸거나 일부 ZIP만 교체하지 않는다.

## 11. VS Code를 열지 않아 진행 로그가 보이지 않는 경우

터미널에서 직접 실행한 프로세스의 로그는 해당 터미널에서만 확인할 수 있다. 완료 여부는 다음 두 가지로 판단한다.

- 터미널에 `[TOTAL] preprocessing complete` 출력
- `dataset/processed/preprocessing_summary.json` 생성

단순히 결과 폴더 일부가 생긴 것만으로 전체 완료라고 판단하지 않는다.
