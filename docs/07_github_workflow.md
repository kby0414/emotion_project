# GitHub 코드 공유와 협업

## 1. 저장소

- GitHub 사용자: `kby0414`
- 저장소: `emotion_project`
- 원격 주소: `https://github.com/kby0414/emotion_project.git`
- 기본 브랜치: `main`

## 2. 공개 범위와 협업자

- 공개 저장소는 GitHub 사용자 누구나 볼 수 있다.
- 비공개 저장소는 소유자와 초대받아 승인한 collaborator만 볼 수 있다.
- GitHub에서 서로 팔로우하거나 친구처럼 연결되는 것만으로 비공개 저장소 전체가 공개되지는 않는다.
- collaborator는 초대받은 해당 저장소에만 접근하며 다른 비공개 저장소까지 자동으로 볼 수 없다.

## 3. Git이 인식되지 않을 때

```powershell
git --version
where.exe git
```

명령을 찾지 못하면 Git for Windows를 설치하고 VS Code와 PowerShell을 완전히 닫았다가 다시 연다. 설치됐지만 현재 터미널 PATH에 없을 때는 다음처럼 확인할 수 있다.

```powershell
Test-Path "C:\Program Files\Git\cmd\git.exe"
$gitPath = "C:\Program Files\Git\cmd"
$env:Path = "$gitPath;$env:Path"
git --version
```

현재 터미널에서 정상화한 뒤 Git 설치 경로를 Windows 사용자 PATH에 추가하면 다음 실행부터 자동 인식된다.

## 4. 다른 PC에서 처음 받기

세 모델의 `best.pt`는 Git LFS로 관리해야 하므로 Git LFS도 필요하다.

```powershell
git lfs install
git clone https://github.com/kby0414/emotion_project.git
cd emotion_project
git lfs pull
```

복제 후 `.venv`와 데이터셋은 자동으로 생기지 않는다. 환경은 `scripts/setup_cuda126.ps1`로 만들고 데이터는 별도 전달 묶음으로 복원한다.

## 5. 이미 받은 프로젝트 최신화

작업 파일이 없는 상태에서 실행한다.

```powershell
cd D:\emotion_project
git status
git pull origin main
git lfs pull
```

`git status`에 본인 수정 사항이 있다면 먼저 커밋하거나 팀원과 확인한 뒤 pull한다. 무조건 덮어쓰거나 삭제하지 않는다.

## 6. 코드 올리기

### VS Code 화면

1. 소스 제어에서 수정 파일을 클릭해 변경 내용을 확인한다.
2. `변경 사항` 옆 `+`를 눌러 올릴 파일을 스테이징한다.
3. 커밋 메시지를 작성한다.
4. `커밋`을 누른다.
5. `변경 내용 동기화` 또는 `Push`를 누른다.

커밋은 내 컴퓨터의 Git 기록만 만든다. GitHub에 반영하려면 push까지 해야 한다.

### PowerShell

```powershell
cd D:\emotion_project
git status
git add docs README.md results/preprocessing_eda_report.md
git commit -m "Document project workflow and results"
git push origin main
```

올리기 전 `git status`로 데이터셋이나 불필요한 대용량 파일이 포함되지 않았는지 확인한다.

## 7. 커밋 메시지 예시

| 작업 | 메시지 예시 |
|---|---|
| 코드 주석 | `Add Korean comments to project code` |
| 문서화 | `Document project workflow and results` |
| 오류 수정 | `Fix preprocessing resume handling` |
| 결과 추가 | `Add EfficientNet B0 experiment results` |

## 8. 모델 파일과 Git LFS

일반 GitHub 파일은 크기 제한 때문에 학습 모델에 적합하지 않을 수 있다. 현재 `.gitattributes`는 `*.pt`를 Git LFS 대상으로 지정한다.

프로그램 개발을 위한 공유 대상은 다음 세 파일이다.

```text
models/GPU_efficientnet_b0_e50_p3_b32/efficientnet_b0/best.pt
models/GPU_mobilenet_v2_e50_p3_b32/mobilenet_v2/best.pt
models/GPU_resnet18_e50_p3_b32/resnet18/best.pt
```

현재 `.gitignore`는 모델 폴더와 `*.pt`를 제외한다. 세 파일을 GitHub로 공유할 때는 Git LFS 설정을 확인한 뒤 해당 경로만 명시적으로 추가하거나 `.gitignore`에 세 경로의 예외를 설정한다.

## 9. 절대 올리지 않을 항목

- `.venv/`: PC마다 다시 만드는 가상환경
- `dataset/raw/`: 원본 얼굴 이미지
- `dataset/labels/`: 원본 라벨
- `dataset/processed/`: 전처리 얼굴 이미지와 라벨
- `transfer/`: 대용량 전달 ZIP
- `last.pt`: 용량이 크고 최종 추론에 필요 없는 재개용 체크포인트
- 비밀번호, 토큰, 개인 인증 파일

## 10. 업로드 완료 확인

```powershell
git status
git log -3 --oneline
```

정상적인 최종 상태는 다음과 비슷하다.

```text
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```
