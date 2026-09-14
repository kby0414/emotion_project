"""이동용 전처리 데이터 묶음을 SHA-256 검증 후 안전하게 복원한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


# 대용량 파일을 일정 크기로 나누어 읽어 메모리 사용량을 제한한다.
COPY_BUFFER_SIZE = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    """manifest 위치와 복원할 대상 폴더 옵션을 읽는다."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="전처리 데이터 ZIP 검증 및 복원")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "transfer" / "processed_dataset" / "transfer_manifest.json",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=project_root / "dataset" / "processed",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="대상 폴더에 기존 파일이 있어도 묶음 파일로 교체(묶음 밖 파일은 삭제하지 않음)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """복사 전후 파일이 같은지 비교하기 위한 SHA-256 값을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(COPY_BUFFER_SIZE):
            digest.update(block)
    return digest.hexdigest()


def safe_destination(target_dir: Path, member_name: str) -> Path:
    """ZIP 내부 경로가 대상 폴더 밖을 가리키지 못하도록 검사한다."""
    member = PurePosixPath(member_name)
    if member.is_absolute() or not member.parts or ".." in member.parts:
        raise RuntimeError(f"안전하지 않은 ZIP 경로: {member_name}")
    if ":" in member.parts[0]:
        raise RuntimeError(f"드라이브가 포함된 ZIP 경로: {member_name}")
    destination = target_dir.joinpath(*member.parts).resolve()
    try:
        destination.relative_to(target_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"대상 폴더 밖 ZIP 경로: {member_name}") from exc
    return destination


def verify_archives(bundle_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    """모든 분할 ZIP의 존재 여부, 크기, SHA-256 해시를 검증한다."""
    archives: list[Path] = []
    for record in manifest.get("archives", []):
        archive_path = bundle_dir / record["name"]
        if not archive_path.is_file():
            raise RuntimeError(f"ZIP 파일이 없습니다: {archive_path}")
        actual_size = archive_path.stat().st_size
        if actual_size != int(record["size_bytes"]):
            raise RuntimeError(
                f"ZIP 크기 불일치: {archive_path.name} expected={record['size_bytes']} actual={actual_size}"
            )
        actual_hash = sha256_file(archive_path)
        if actual_hash.casefold() != str(record["sha256"]).casefold():
            raise RuntimeError(f"SHA-256 불일치(복사 손상): {archive_path.name}")
        print(f"[VERIFY] PASS {archive_path.name}")
        archives.append(archive_path)
    if not archives:
        raise RuntimeError("manifest에 ZIP 목록이 없습니다.")
    return archives


def main() -> None:
    """묶음을 검증하고 임시 파일을 거쳐 안전하게 전처리 폴더를 복원한다."""
    args = parse_args()
    started = time.perf_counter()
    manifest_path = args.manifest.resolve()
    target_dir = args.target_dir.resolve()

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"지원하지 않는 manifest 버전: {manifest.get('schema_version')}")

    if target_dir.exists() and any(target_dir.iterdir()) and not args.allow_existing:
        raise RuntimeError(
            f"대상 폴더가 비어 있지 않습니다: {target_dir}. 새 폴더를 사용하거나 --allow-existing를 지정하세요."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    archives = verify_archives(manifest_path.parent, manifest)
    # 동일 파일이 여러 ZIP에 중복되면 덮어쓰기 순서에 따라 결과가 달라지므로 막는다.
    seen_members: set[str] = set()
    extracted = 0

    for archive_path in archives:
        with zipfile.ZipFile(archive_path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"ZIP CRC 오류: {archive_path.name} / {bad_member}")
            for info in archive.infolist():
                if info.is_dir():
                    continue
                normalized = PurePosixPath(info.filename).as_posix().casefold()
                if normalized in seen_members:
                    raise RuntimeError(f"여러 ZIP에 중복된 파일: {info.filename}")
                seen_members.add(normalized)

                destination = safe_destination(target_dir, info.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                # 복원 도중 중단되어도 불완전한 파일이 정상 파일명으로 남지 않게 한다.
                temporary_path = destination.with_name(destination.name + ".transfer.tmp")
                with archive.open(info, "r") as source, temporary_path.open("wb") as target:
                    shutil.copyfileobj(source, target, COPY_BUFFER_SIZE)
                os.replace(temporary_path, destination)
                extracted += 1
                if extracted % 10_000 == 0:
                    print(f"[UNPACK] progress={extracted:,}/{manifest['total_files']:,}", flush=True)

    if extracted != int(manifest["total_files"]):
        raise RuntimeError(
            f"복원 파일 수 불일치: expected={manifest['total_files']} actual={extracted}"
        )
    summary_path = target_dir / "preprocessing_summary.json"
    if sha256_file(summary_path) != manifest["source_summary_sha256"]:
        raise RuntimeError("복원된 preprocessing_summary.json의 SHA-256이 다릅니다.")

    # 마지막으로 이미지와 라벨 개수까지 원본 manifest와 같은지 확인한다.
    for key, expected in manifest.get("counts", {}).items():
        category, split = key.split("/", 1)
        suffix = "*.png" if category == "images" else "*.json"
        actual = sum(1 for _ in (target_dir / category / split).rglob(suffix))
        if actual != int(expected):
            raise RuntimeError(f"{key} 수 불일치: expected={expected}, actual={actual}")

    elapsed = time.perf_counter() - started
    print(f"[DONE] restored={target_dir}")
    print(f"[DONE] files={extracted:,}, elapsed={elapsed:.2f} sec")


if __name__ == "__main__":
    main()
