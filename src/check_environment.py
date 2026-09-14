"""프로젝트 Python/PyTorch/CUDA 환경을 검증한다."""

from __future__ import annotations

import argparse
import platform
import sys


# 팀원 PC에서도 같은 결과를 재현할 수 있도록 프로젝트의 기준 버전을 고정한다.
EXPECTED_PYTHON = (3, 12, 10)
EXPECTED_TORCH = "2.13.0+cu126"
EXPECTED_TORCHVISION = "0.28.0+cu126"
EXPECTED_PILLOW = "12.3.0"
EXPECTED_MATPLOTLIB = "3.11.1"
EXPECTED_CUDA = "12.6"


def parse_args() -> argparse.Namespace:
    """환경 검사 방식에 관한 명령행 옵션을 읽는다."""
    parser = argparse.ArgumentParser(description="Python/PyTorch/CUDA 환경 확인")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="CUDA GPU가 없어도 실패로 처리하지 않음",
    )
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help="고정 버전이 달라도 장치 동작만 검사",
    )
    return parser.parse_args()


def main() -> None:
    """패키지 버전, CUDA 인식 여부, 실제 GPU 연산을 차례로 검증한다."""
    args = parse_args()

    # import 자체가 실패하면 설치가 덜 되었거나 다른 가상환경을 선택한 경우다.
    try:
        import matplotlib
        import PIL
        import torch
        import torchvision
    except ImportError as exc:
        raise SystemExit(f"[FAIL] 필수 패키지를 불러오지 못했습니다: {exc}") from exc

    actual = {
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda or "NONE",
        "cuda_available": str(torch.cuda.is_available()),
    }

    print("[ENV] executable:", sys.executable)
    for name, value in actual.items():
        print(f"[ENV] {name}: {value}")

    # --skip-version-check를 쓰지 않은 기본 실행에서는 lock 파일 기준과 정확히 비교한다.
    mismatches: list[str] = []
    if not args.skip_version_check:
        expected = {
            "python": ".".join(map(str, EXPECTED_PYTHON)),
            "pillow": EXPECTED_PILLOW,
            "matplotlib": EXPECTED_MATPLOTLIB,
            "torch": EXPECTED_TORCH,
            "torchvision": EXPECTED_TORCHVISION,
            "torch_cuda_runtime": EXPECTED_CUDA,
        }
        for name, expected_value in expected.items():
            if actual[name] != expected_value:
                mismatches.append(
                    f"{name}: expected={expected_value}, actual={actual[name]}"
                )

    if torch.cuda.is_available():
        # torch.cuda.is_available()만 확인하면 드라이버 문제를 놓칠 수 있으므로
        # 행렬 곱셈을 실행하고 synchronize()로 GPU 작업 완료까지 기다린다.
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        print("[GPU] name:", properties.name)
        print("[GPU] compute capability:", f"{properties.major}.{properties.minor}")
        print("[GPU] memory GiB:", f"{properties.total_memory / 1024**3:.2f}")

        # 단순 인식만이 아니라 실제 CUDA 커널 실행까지 확인한다.
        left = torch.randn((1024, 1024), device=device)
        right = torch.randn((1024, 1024), device=device)
        result = left @ right
        torch.cuda.synchronize()
        print("[GPU] matrix test:", f"PASS ({result.mean().item():.6f})")
    elif not args.allow_cpu:
        mismatches.append("CUDA GPU를 PyTorch에서 사용할 수 없습니다")
    else:
        print("[GPU] CUDA 미사용(--allow-cpu)")

    if mismatches:
        print("[FAIL] 환경이 고정 사양과 일치하지 않습니다:")
        for mismatch in mismatches:
            print("  -", mismatch)
        raise SystemExit(1)

    print("[OK] 프로젝트 환경 및 CUDA 연산 검증 완료")


if __name__ == "__main__":
    main()
