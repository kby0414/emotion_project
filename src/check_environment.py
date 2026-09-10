"""프로젝트 Python/PyTorch/CUDA 환경을 검증한다."""

from __future__ import annotations

import argparse
import platform
import sys


EXPECTED_PYTHON = (3, 12, 10)
EXPECTED_TORCH = "2.13.0+cu126"
EXPECTED_TORCHVISION = "0.28.0+cu126"
EXPECTED_PILLOW = "12.3.0"
EXPECTED_CUDA = "12.6"


def parse_args() -> argparse.Namespace:
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
    args = parse_args()

    try:
        import PIL
        import torch
        import torchvision
    except ImportError as exc:
        raise SystemExit(f"[FAIL] 필수 패키지를 불러오지 못했습니다: {exc}") from exc

    actual = {
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda or "NONE",
        "cuda_available": str(torch.cuda.is_available()),
    }

    print("[ENV] executable:", sys.executable)
    for name, value in actual.items():
        print(f"[ENV] {name}: {value}")

    mismatches: list[str] = []
    if not args.skip_version_check:
        expected = {
            "python": ".".join(map(str, EXPECTED_PYTHON)),
            "pillow": EXPECTED_PILLOW,
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
