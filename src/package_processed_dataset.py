"""완료된 전처리 데이터셋을 이동용 분할 ZIP과 SHA-256 manifest로 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


# 묶음 파일명과 manifest 이름은 복원 스크립트에서도 같은 규칙을 사용한다.
ARCHIVE_PREFIX = "processed_dataset_part_"
MANIFEST_NAME = "transfer_manifest.json"
COPY_BUFFER_SIZE = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    """전처리 폴더, 저장 위치, ZIP 한 개의 최대 크기를 입력받는다."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="전처리가 끝난 데이터셋을 검증 가능한 분할 ZIP으로 생성"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=project_root / "dataset" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "transfer" / "processed_dataset",
    )
    parser.add_argument(
        "--chunk-size-gb",
        type=float,
        default=3.5,
        help="ZIP 한 개의 최대 원본 파일 크기 합계. 기본 3.5 GiB(FAT32 이동 가능)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이 스크립트가 전에 만든 ZIP/manifest만 교체",
    )
    args = parser.parse_args()
    if args.chunk_size_gb <= 0:
        parser.error("--chunk-size-gb는 0보다 커야 합니다.")
    return args


def sha256_file(path: Path) -> str:
    """큰 파일도 메모리에 전부 올리지 않고 SHA-256 해시를 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(COPY_BUFFER_SIZE):
            digest.update(block)
    return digest.hexdigest()


def collect_and_validate(processed_dir: Path) -> tuple[list[Path], dict[str, Any], dict[str, int]]:
    """전처리 완료 여부와 summary의 이미지·라벨 개수를 실제 파일과 대조한다."""
    summary_path = processed_dir / "preprocessing_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            "preprocessing_summary.json이 없습니다. 전처리가 완전히 끝난 뒤 실행하세요."
        )

    # .tmp가 남아 있다면 강제 종료 등으로 아직 완성되지 않은 파일일 수 있다.
    temporary_files = [path for path in processed_dir.rglob("*") if path.is_file() and path.name.endswith(".tmp")]
    if temporary_files:
        raise RuntimeError(f"임시 파일이 {len(temporary_files)}개 남아 있습니다. 전처리 중일 수 있습니다.")

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    counts: dict[str, int] = {}
    for split, split_summary in summary.get("splits", {}).items():
        image_count = sum(1 for _ in (processed_dir / "images" / split).rglob("*.png"))
        label_count = sum(1 for _ in (processed_dir / "labels" / split).rglob("*.json"))
        expected_images = int(split_summary.get("images", -1))
        expected_labels = int(split_summary.get("labels", {}).get("json_files", -1))
        if image_count != expected_images:
            raise RuntimeError(
                f"{split} PNG 수 불일치: summary={expected_images}, actual={image_count}"
            )
        if label_count != expected_labels:
            raise RuntimeError(
                f"{split} 라벨 JSON 수 불일치: summary={expected_labels}, actual={label_count}"
            )
        counts[f"images/{split}"] = image_count
        counts[f"labels/{split}"] = label_count

    files = sorted(path for path in processed_dir.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("패키징할 파일이 없습니다.")
    return files, summary, counts


def split_files(files: list[Path], max_bytes: int) -> list[list[Path]]:
    """원본 파일 크기 합이 제한을 넘지 않도록 파일 목록을 여러 묶음으로 나눈다."""
    chunks: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in files:
        size = path.stat().st_size
        if size > max_bytes:
            raise RuntimeError(f"한 파일이 chunk 제한보다 큽니다: {path} ({size} bytes)")
        if current and current_bytes + size > max_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    """데이터 검증 → 분할 ZIP 생성 → 해시 manifest 저장 순서로 실행한다."""
    args = parse_args()
    started = time.perf_counter()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    max_bytes = int(args.chunk_size_gb * 1024**3)

    files, summary, counts = collect_and_validate(processed_dir)
    chunks = split_files(files, max_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_source_bytes = sum(path.stat().st_size for path in files)
    # ZIP_STORED 데이터와 ZIP 디렉터리/파일명 메타데이터를 함께 둘 여유 공간.
    estimated_required_bytes = total_source_bytes + len(files) * 512 + 16 * 1024**2
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < estimated_required_bytes:
        raise RuntimeError(
            "묶음을 만들 디스크 공간이 부족합니다: "
            f"required~={estimated_required_bytes:,}, free={free_bytes:,} bytes"
        )

    existing = list(output_dir.glob(f"{ARCHIVE_PREFIX}*.zip"))
    manifest_path = output_dir / MANIFEST_NAME
    temporary = list(output_dir.glob("*.tmp"))
    if (existing or manifest_path.exists() or temporary) and not args.overwrite:
        raise RuntimeError(
            f"기존 묶음이 있습니다: {output_dir}. 교체하려면 --overwrite를 사용하세요."
        )
    if args.overwrite:
        for path in [*existing, *temporary, manifest_path]:
            if path.exists() and path.is_file():
                path.unlink()

    print(f"[PACK] files={len(files):,}, chunks={len(chunks)}, limit={args.chunk_size_gb:.2f} GiB")
    archive_records: list[dict[str, Any]] = []
    completed_files = 0

    # 각 ZIP을 임시 이름으로 완성한 뒤 최종 이름으로 바꿔 중단 시 손상을 구별한다.
    for index, chunk in enumerate(chunks, start=1):
        archive_name = f"{ARCHIVE_PREFIX}{index:03d}.zip"
        archive_path = output_dir / archive_name
        temporary_path = archive_path.with_suffix(".zip.tmp")
        chunk_bytes = sum(path.stat().st_size for path in chunk)
        print(f"[PACK] {archive_name}: files={len(chunk):,}, bytes={chunk_bytes:,}")

        # PNG는 이미 압축되어 있으므로 재압축하지 않아 속도와 CPU 사용량을 아낀다.
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in chunk:
                archive.write(path, path.relative_to(processed_dir).as_posix())
                completed_files += 1
                if completed_files % 10_000 == 0:
                    print(f"[PACK] progress={completed_files:,}/{len(files):,}", flush=True)
        temporary_path.replace(archive_path)

        archive_records.append(
            {
                "name": archive_name,
                "sha256": sha256_file(archive_path),
                "size_bytes": archive_path.stat().st_size,
                "file_count": len(chunk),
            }
        )

    # 다른 PC에서는 아래 해시와 개수를 사용해 복사 중 손상 여부를 확인한다.
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "format": "ZIP_STORED",
        "source_summary_sha256": sha256_file(processed_dir / "preprocessing_summary.json"),
        "preprocessing_total_elapsed": summary.get("total_elapsed_hhmmss"),
        "chunk_size_bytes": max_bytes,
        "total_files": len(files),
        "total_source_bytes": total_source_bytes,
        "counts": counts,
        "archives": archive_records,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_manifest.replace(manifest_path)

    elapsed = time.perf_counter() - started
    print(f"[DONE] bundle={output_dir}")
    print(f"[DONE] manifest={manifest_path}")
    print(f"[DONE] elapsed={elapsed:.2f} sec")


if __name__ == "__main__":
    main()
