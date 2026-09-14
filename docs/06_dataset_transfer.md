# 전처리 데이터셋을 다른 PC로 이전하기

## 1. GitHub에 직접 올리지 않는 이유

전처리 데이터는 약 12.85 GiB의 이미지와 라벨로 구성되고 얼굴 이미지의 개인정보·라이선스 문제도 있다. GitHub는 대용량 학습 데이터 저장소가 아니므로 코드·작은 결과·필요한 모델만 공유하고 데이터셋은 별도 저장장치나 허가된 클라우드로 전달한다.

현재 `.gitignore`는 다음을 Git에서 제외한다.

```text
dataset/raw/
dataset/labels/
dataset/processed/
transfer/
*.zip
```

## 2. 전달 묶음 만들기

전처리가 완전히 끝난 PC에서 실행한다.

```powershell
.\.venv\Scripts\python.exe .\src\package_processed_dataset.py
```

기본 출력 위치는 다음과 같다.

```text
transfer/processed_dataset/
├─ processed_dataset_part_001.zip
├─ processed_dataset_part_002.zip
├─ processed_dataset_part_003.zip
├─ processed_dataset_part_004.zip
└─ transfer_manifest.json
```

현재 묶음은 4개 ZIP, 총 275,703개 파일이다. 이는 이미지 275,688개, JSON 라벨 14개, 전처리 summary 1개를 합한 수다. 원본 파일 크기 합은 약 13.13 GiB다.

ZIP은 PNG를 다시 압축하지 않는 `ZIP_STORED`를 사용한다. PNG는 이미 압축되어 있어 재압축 이득이 작고 시간만 늘어날 수 있기 때문이다.

기존 묶음을 교체해야 할 때만 다음을 사용한다.

```powershell
.\.venv\Scripts\python.exe .\src\package_processed_dataset.py --overwrite
```

## 3. 어떤 파일을 복사해야 하는가

다른 PC에는 `processed_dataset_part_*.zip` 전부와 `transfer_manifest.json`을 같은 폴더 구조로 복사한다. ZIP 하나라도 빠지면 복원이 실패한다.

전송 수단 예시:

- 외장 SSD
- 같은 네트워크의 공유 폴더
- Google Drive, OneDrive, Dropbox 등 팀에서 허가한 클라우드
- 학교·기관의 NAS 또는 대용량 스토리지

데이터 이용 약관이 외부 클라우드 업로드를 허용하는지 먼저 확인한다.

## 4. 다른 PC 준비

먼저 GitHub에서 코드를 받은 뒤 환경을 구성한다.

```powershell
git lfs install
git clone https://github.com/kby0414/emotion_project.git
cd emotion_project
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_cuda126.ps1
```

그다음 전달받은 파일을 아래 위치에 둔다.

```text
emotion_project/transfer/processed_dataset/
```

## 5. 검증하고 복원하기

```powershell
.\.venv\Scripts\python.exe .\src\unpack_processed_dataset.py
```

복원 스크립트는 다음을 검사한다.

1. 모든 ZIP 존재 여부
2. manifest에 기록된 파일 크기
3. ZIP별 SHA-256 해시
4. ZIP 내부 CRC 오류
5. 중복 또는 안전하지 않은 내부 경로
6. 전체 복원 파일 수
7. 이미지·라벨 수와 summary 해시

완료 위치는 `dataset/processed/`다. 기존 폴더가 비어 있지 않으면 실수로 덮어쓰지 않도록 중단한다. 기존 파일을 유지하면서 묶음 파일만 교체하려는 경우에만 `--allow-existing`을 사용한다.

```powershell
.\.venv\Scripts\python.exe .\src\unpack_processed_dataset.py --allow-existing
```

## 6. 복원 후 학습 전 확인

```powershell
.\.venv\Scripts\python.exe .\src\check_environment.py
```

그리고 다음 폴더가 존재하는지 확인한다.

```text
dataset/processed/images/Training/
dataset/processed/images/Validation/
dataset/processed/labels/Training/
dataset/processed/labels/Validation/
dataset/processed/preprocessing_summary.json
```

## 7. 보안과 관리 원칙

- 원본과 전처리 얼굴 이미지는 공개 GitHub에 올리지 않는다.
- 팀원도 데이터 이용 범위와 삭제 시점을 공유한다.
- ZIP을 전송한 뒤 SHA-256 검증에 실패하면 학습하지 말고 다시 복사한다.
- `transfer_manifest.json`만으로는 데이터가 복원되지 않으며 모든 ZIP이 필요하다.

