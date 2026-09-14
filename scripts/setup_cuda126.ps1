param(
    # 저장소를 다른 위치에 복제해도 스크립트 위치를 기준으로 프로젝트를 찾는다.
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

# 한 단계라도 실패하면 다음 설치 단계로 넘어가지 않는다.
$ErrorActionPreference = "Stop"
$ExpectedPython = "3.12.10"
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$CommonRequirements = Join-Path $ProjectRoot "requirements-common.lock.txt"
$CudaRequirements = Join-Path $ProjectRoot "requirements-cu126.lock.txt"
$CheckScript = Join-Path $ProjectRoot "src\check_environment.py"

Write-Host "[1/6] NVIDIA 드라이버와 GPU 확인"
# nvidia-smi의 CUDA 표시는 드라이버가 지원하는 최대 버전이다.
# PyTorch가 사용하는 런타임은 마지막 환경 검사에서 따로 확인한다.
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi를 찾지 못했습니다. NVIDIA 드라이버를 먼저 설치하세요."
}
nvidia-smi

Write-Host "[2/6] Python $ExpectedPython 확인"
# 여러 Python이 설치되어 있어도 py -3.12로 3.12 계열을 명시적으로 선택한다.
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Windows Python Launcher(py.exe)가 없습니다. Python $ExpectedPython 64-bit를 설치하세요."
}
$DetectedPython = (& py -3.12 -c "import platform; print(platform.python_version())").Trim()
if ($LASTEXITCODE -ne 0 -or $DetectedPython -ne $ExpectedPython) {
    throw "Python $ExpectedPython이 필요하지만 감지된 버전은 '$DetectedPython'입니다."
}

Write-Host "[3/6] 가상환경 준비"
# 기존 .venv는 자동 삭제하지 않고 버전이 맞을 때만 재사용해 사용자 파일을 보호한다.
if (Test-Path $VenvPath) {
    if (-not (Test-Path $VenvPython)) {
        throw "기존 .venv가 불완전합니다. 안전을 위해 자동 삭제하지 않습니다. 이름을 바꾸거나 삭제한 뒤 다시 실행하세요."
    }
    try {
        $venvVersion = & $venvPython -c "import platform; print(platform.python_version())"
        $venvVersion = $venvVersion.Trim()
    }
    catch {
        throw "기존 .venv가 실행되지 않습니다. 진행 중인 작업이 없다면 .venv 이름을 바꾸고 다시 실행하세요."
    }
    if ($VenvVersion -ne $ExpectedPython) {
        throw "기존 .venv Python은 $VenvVersion입니다. Python $ExpectedPython 가상환경이 필요합니다."
    }
    Write-Host "기존 가상환경을 재사용합니다: $VenvPath"
}
else {
    & py -3.12 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "가상환경 생성 실패" }
}

Write-Host "[4/6] pip 도구 버전 고정"
# 가상환경의 python.exe로 pip를 실행해야 시스템 Python에 잘못 설치되지 않는다.
& $VenvPython -m pip install --upgrade "pip==26.2.1" "setuptools==84.0.0" "wheel==0.48.0"
if ($LASTEXITCODE -ne 0) { throw "pip 도구 설치 실패" }

Write-Host "[5/6] 공통 패키지와 PyTorch CUDA 12.6 wheel 설치"
# 일반 라이브러리와 CUDA 12.6용 PyTorch wheel을 분리해 설치 출처를 명확히 한다.
& $VenvPython -m pip install --requirement $CommonRequirements
if ($LASTEXITCODE -ne 0) { throw "공통 패키지 설치 실패" }
& $VenvPython -m pip install --no-deps --requirement $CudaRequirements
if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA 12.6 패키지 설치 실패" }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "패키지 의존성 검사 실패" }

Write-Host "[6/6] Python/PyTorch/CUDA 실제 연산 검증"
# 버전 문자열뿐 아니라 GPU 행렬 연산까지 통과해야 설치 성공으로 판단한다.
& $VenvPython $CheckScript
if ($LASTEXITCODE -ne 0) { throw "환경 검증 실패" }

Write-Host "[DONE] VS Code에서 D:\emotion_project가 아닌 이 ProjectRoot 폴더를 열고 .venv 인터프리터를 선택하세요."
