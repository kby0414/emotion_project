# Python·PyTorch·GPU 환경 구성

## 1. 선택한 프레임워크

이 프로젝트의 모델 학습은 TensorFlow가 아니라 PyTorch와 torchvision으로 구현했다. 세 비교 모델이 torchvision에서 같은 방식으로 제공되고, 학습 반복문·지표·체크포인트 저장을 한 코드에서 통제하기 쉽기 때문이다.

TensorFlow의 성능이 부족해서 제외한 것이 아니다. 프로젝트 전체를 한 프레임워크로 통일해 코드 충돌을 줄이고 재현하기 위한 선택이다.

## 2. 기준 환경

| 항목 | 기준 |
|---|---|
| 운영체제 | Windows |
| Python | 3.12.10 64-bit |
| PyTorch 기준 lock | 2.13.0+cu126 |
| torchvision 기준 lock | 0.28.0+cu126 |
| PyTorch CUDA runtime | 12.6 |
| Pillow | 12.3.0 |
| matplotlib | 3.11.1 |

여러 Python 버전은 함께 설치되어 있어도 된다. 중요한 것은 프로젝트의 `.venv`를 Python 3.12.10으로 만들고 VS Code가 그 인터프리터를 선택하는 것이다. 다른 Python을 무조건 삭제할 필요는 없다.

## 3. 설치 순서

프로젝트 루트 PowerShell에서 실행한다.

```powershell
nvidia-smi
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_cuda126.ps1
```

설정 스크립트는 다음 순서로 동작한다.

1. `nvidia-smi`로 NVIDIA GPU와 드라이버 확인
2. Python 3.12.10 확인
3. `.venv` 생성 또는 기존 환경 검사
4. pip·setuptools·wheel 버전 고정
5. 공통 패키지와 CUDA 12.6용 PyTorch wheel 설치
6. 패키지 버전과 실제 GPU 행렬 연산 검사

`nvidia-smi`에 표시되는 CUDA 버전은 드라이버가 지원하는 최대 버전이다. PyTorch가 실제로 사용하는 런타임은 `torch.version.cuda`로 별도 확인한다. PyTorch wheel에는 해당 CUDA 런타임이 포함되므로, 시스템 CUDA Toolkit 문자열만 보고 성공 여부를 판단하지 않는다.

## 4. 수동 검증

```powershell
.\.venv\Scripts\python.exe .\src\check_environment.py
```

핵심 확인 항목은 다음과 같다.

```python
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

정상 기준은 CUDA runtime이 `12.6`, `torch.cuda.is_available()`이 `True`, GPU 이름이 출력되고 행렬 연산 테스트가 `PASS`인 것이다.

## 5. 학습 코드의 GPU 선택

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images = images.to(device)
targets = targets.to(device)
```

CUDA가 인식되면 모델과 배치가 GPU로 이동한다. CUDA 환경에서는 AMP 혼합 정밀도도 활성화해 속도와 메모리 사용량을 개선한다. GPU가 인식되지 않으면 CPU로 실행되지만 전체 데이터 학습에는 매우 오래 걸릴 수 있다.

## 6. VS Code 설정

1. VS Code에서 프로젝트 루트 `D:\emotion_project`를 연다.
2. `Ctrl+Shift+P`를 누른다.
3. `Python: Select Interpreter`를 선택한다.
4. `D:\emotion_project\.venv\Scripts\python.exe`를 고른다.
5. 새 터미널을 열어 프롬프트 앞에 `(.venv)`가 표시되는지 확인한다.

프로젝트의 `.vscode/settings.json`도 위 가상환경을 기본 인터프리터로 지정한다.

## 7. 실제 실험 환경에서 확인된 차이

- EfficientNet-B0와 ResNet18 결과 기록: PyTorch `2.13.0+cu126`
- MobileNetV2 결과 기록: PyTorch `2.14.0+cu126`
- 세 실험 모두 Python `3.12.10`, CUDA runtime `12.6`, NVIDIA GeForce RTX 4060

따라서 현재 성능 비교는 데이터와 주요 학습 조건은 같지만 PyTorch 패치 환경까지 완전히 동일한 실험은 아니다. 후속 재현 실험에서는 `requirements-cu126.lock.txt`의 단일 버전으로 세 모델을 다시 실행하는 것이 가장 엄밀하다.

