"""감정 이미지 데이터셋 전처리.

기본 동작
  * dataset/raw/{Training,Validation} 아래의 JPG/JPEG를 재귀적으로 탐색
  * 비율을 유지한 채 224 x 224로 축소하고 남는 영역을 패딩
  * 1채널 흑백 PNG로 저장
  * 대응하는 JSON의 filename(.jpg -> .png)과 얼굴 박스 좌표를 함께 변환

원본 이미지와 원본 라벨은 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError


# 로그 이름과 반복해서 사용하는 확장자·좌표 키를 한곳에서 관리한다.
LOGGER = logging.getLogger("emotion-preprocess")
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
BOX_KEYS = ("minX", "minY", "maxX", "maxY")


@dataclass(frozen=True)
class ImageTransform:
    """한 이미지의 회전·축소·패딩 결과와 라벨 좌표 변환 정보를 보관한다."""

    source_relative_path: str
    output_relative_path: str
    original_width: int
    original_height: int
    exif_orientation: int
    oriented_width: int
    oriented_height: int
    label_coordinate_space: str
    output_width: int
    output_height: int
    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int


def parse_args() -> argparse.Namespace:
    """전처리 입력·출력 경로와 이미지 변환 옵션을 읽고 값의 범위를 검사한다."""
    script_path = Path(__file__).resolve()
    default_project_root = script_path.parent.parent

    parser = argparse.ArgumentParser(
        description="JPG 감정 이미지를 흑백 PNG로 만들고 라벨도 함께 변환합니다."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_project_root / "dataset" / "raw",
        help="Training/Validation 이미지 폴더가 있는 경로",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=default_project_root / "dataset" / "labels",
        help="Training/Validation JSON 라벨 폴더가 있는 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_project_root / "dataset" / "processed",
        help="전처리 결과를 저장할 경로",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="출력 이미지의 가로/세로 크기 (기본값: 224)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Training", "Validation"],
        help="처리할 데이터 분할 이름",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 3),
        default=1,
        help="1: 실제 1채널 흑백, 3: 동일한 흑백값을 RGB 3채널로 저장",
    )
    parser.add_argument(
        "--pad-value",
        type=int,
        default=0,
        help="패딩 밝기 0~255 (기본값: 검정 0)",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        choices=range(10),
        default=1,
        metavar="0-9",
        help="PNG 압축 단계. 낮을수록 빠름 (기본값: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="동시에 처리할 이미지 수",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 결과 파일을 다시 생성",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="라벨에 대응하는 이미지를 찾지 못하면 오류로 종료",
    )
    parser.add_argument(
        "--strict-images",
        action="store_true",
        help="손상된 이미지가 하나라도 있으면 전체 작업을 오류로 종료",
    )
    parser.add_argument(
        "--keep-unmatched-labels",
        action="store_true",
        help="변환 이미지가 없는 라벨 레코드도 결과 JSON에 유지",
    )
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="잘린 JPEG를 가능한 범위까지 복구해 사용 (기본값: 안전하게 제외)",
    )
    parser.add_argument(
        "--max-megapixels",
        type=float,
        default=250.0,
        help="허용할 이미지 최대 화소 수(MP). 신뢰하는 데이터에만 높게 설정",
    )
    parser.add_argument(
        "--label-coordinate-space",
        choices=("displayed", "raw"),
        default="displayed",
        help=(
            "displayed: 라벨 좌표가 EXIF 적용 후 화면 기준(현재 데이터셋), "
            "raw: JPEG 저장 픽셀 기준"
        ),
    )
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size는 1 이상이어야 합니다.")
    if not 0 <= args.pad_value <= 255:
        parser.error("--pad-value는 0~255여야 합니다.")
    if args.workers <= 0:
        parser.error("--workers는 1 이상이어야 합니다.")
    if args.max_megapixels <= 0:
        parser.error("--max-megapixels는 0보다 커야 합니다.")
    return args


def find_images(folder: Path) -> list[Path]:
    """하위 폴더 구조와 관계없이 모든 JPG/JPEG 파일을 재귀적으로 찾는다."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )


def png_relative_path(image_path: Path, split_dir: Path) -> Path:
    """원본의 하위 폴더 구조는 유지하고 확장자만 .png로 바꾼다."""
    return image_path.relative_to(split_dir).with_suffix(".png")


def check_output_collisions(images: Iterable[Path], split_dir: Path) -> None:
    """여러 원본이 같은 출력 PNG 또는 같은 라벨 filename으로 겹치는지 확인한다."""
    seen: dict[str, Path] = {}
    seen_basenames: dict[str, Path] = {}
    for image_path in images:
        output_key = png_relative_path(image_path, split_dir).as_posix().casefold()
        previous = seen.get(output_key)
        if previous is not None:
            raise ValueError(
                "같은 PNG 출력 경로가 생기는 이미지가 있습니다: "
                f"{previous} / {image_path}"
            )
        seen[output_key] = image_path

        basename_key = image_path.name.casefold()
        previous_basename = seen_basenames.get(basename_key)
        if previous_basename is not None:
            raise ValueError(
                "라벨 filename으로 구별할 수 없는 중복 이미지가 있습니다: "
                f"{previous_basename} / {image_path}"
            )
        seen_basenames[basename_key] = image_path


def oriented_size(width: int, height: int, orientation: int) -> tuple[int, int]:
    """EXIF 방향 보정을 적용한 뒤의 가로·세로 크기를 계산한다."""
    if orientation in (5, 6, 7, 8):
        return height, width
    return width, height


def orient_point(
    x: float,
    y: float,
    width: int,
    height: int,
    orientation: int,
) -> tuple[float, float]:
    """원본 픽셀 좌표 한 점에 EXIF 회전·반전을 적용한다."""
    transforms = {
        1: lambda px, py: (px, py),
        2: lambda px, py: (width - px, py),
        3: lambda px, py: (width - px, height - py),
        4: lambda px, py: (px, height - py),
        5: lambda px, py: (py, px),
        6: lambda px, py: (height - py, px),
        7: lambda px, py: (height - py, width - px),
        8: lambda px, py: (py, width - px),
    }
    return transforms.get(orientation, transforms[1])(x, y)


def format_duration(seconds: float) -> str:
    """초 단위 시간을 터미널에서 읽기 쉬운 HH:MM:SS 문자열로 바꾼다."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def configure_pillow_worker(
    max_megapixels: float, allow_truncated: bool
) -> None:
    """Windows에서 생성된 각 작업 프로세스에 Pillow 안전 옵션을 적용한다."""
    Image.MAX_IMAGE_PIXELS = int(max_megapixels * 1_000_000)
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated


def preprocess_one_image(
    image_path: Path,
    split_dir: Path,
    output_split_dir: Path,
    size: int,
    channels: int,
    pad_value: int,
    overwrite: bool,
    label_coordinate_space: str,
    png_compress_level: int,
) -> ImageTransform:
    """이미지 한 장을 방향 보정한 224×224 흑백 PNG로 변환한다.

    원본 비율을 유지해 축소한 뒤 중앙에 패딩하며, 같은 변환을 JSON 얼굴
    박스에도 적용할 수 있도록 :class:`ImageTransform`을 반환한다.
    """
    relative = image_path.relative_to(split_dir)
    output_relative = relative.with_suffix(".png")
    output_path = output_split_dir / output_relative

    try:
        with Image.open(image_path) as source:
            original_width, original_height = source.size
            if original_width <= 0 or original_height <= 0:
                raise ValueError("이미지 크기가 0입니다.")

            exif_orientation = int(source.getexif().get(274, 1) or 1)
            if exif_orientation not in range(1, 9):
                LOGGER.warning(
                    "알 수 없는 EXIF 방향값 %s를 기본 방향으로 처리합니다: %s",
                    exif_orientation,
                    image_path,
                )
                exif_orientation = 1
            oriented_width, oriented_height = oriented_size(
                original_width, original_height, exif_orientation
            )

            # 긴 변을 size에 맞추는 letterbox 방식이라 얼굴 비율이 찌그러지지 않는다.
            scale = min(size / oriented_width, size / oriented_height)
            resized_width = max(1, round(oriented_width * scale))
            resized_height = max(1, round(oriented_height * scale))
            offset_x = (size - resized_width) // 2
            offset_y = (size - resized_height) // 2

            if overwrite or not output_path.exists():
                # 큰 JPEG는 디코더 단계에서 먼저 축소해 수백 MB의 메모리 사용을 피한다.
                # 기존 PNG가 있으면 헤더만 읽고 픽셀 디코딩은 생략해 재시작을 빠르게 한다.
                if source.format == "JPEG":
                    source.draft("L", (size, size))
                source.load()

                # EXIF 방향을 픽셀에 실제 적용하고 태그를 제거한다. 이 데이터셋의 라벨
                # 좌표는 이미 화면 방향 기준이므로 회전 후 크기의 배율만 적용한다.
                oriented = ImageOps.exif_transpose(source)
                grayscale = oriented.convert("L")
                resized = grayscale.resize(
                    (resized_width, resized_height), Image.Resampling.LANCZOS
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                canvas = Image.new("L", (size, size), color=pad_value)
                canvas.paste(resized, (offset_x, offset_y))
                if channels == 3:
                    canvas = canvas.convert("RGB")

                # 임시 파일을 완성 후 교체하여 강제 종료 시 깨진 PNG가 남는 것을 막는다.
                temporary_path = output_path.with_suffix(".png.tmp")
                canvas.save(
                    temporary_path,
                    format="PNG",
                    compress_level=png_compress_level,
                )
                temporary_path.replace(output_path)

    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise RuntimeError(
            f"이미지를 열거나 저장하지 못했습니다: {image_path} "
            f"[{type(exc).__name__}: {exc}]"
        ) from exc

    return ImageTransform(
        source_relative_path=relative.as_posix(),
        output_relative_path=output_relative.as_posix(),
        original_width=original_width,
        original_height=original_height,
        exif_orientation=exif_orientation,
        oriented_width=oriented_width,
        oriented_height=oriented_height,
        label_coordinate_space=label_coordinate_space,
        output_width=size,
        output_height=size,
        scale_x=resized_width / original_width,
        scale_y=resized_height / original_height,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def normalized_label_key(value: str) -> str:
    """Windows/Unix 경로 구분자와 대소문자 차이를 제거해 비교용 키를 만든다."""
    return value.replace("\\", "/").lstrip("./").casefold()


def make_transform_lookup(
    transforms: list[ImageTransform],
) -> tuple[dict[str, ImageTransform], dict[str, ImageTransform]]:
    """라벨 filename을 이미지 변환 정보에 연결할 두 가지 검색표를 만든다."""
    by_relative_path: dict[str, ImageTransform] = {}
    basename_candidates: dict[str, list[ImageTransform]] = {}

    for transform in transforms:
        relative_key = normalized_label_key(transform.source_relative_path)
        by_relative_path[relative_key] = transform
        basename_key = Path(transform.source_relative_path).name.casefold()
        basename_candidates.setdefault(basename_key, []).append(transform)

    # 같은 파일명이 여러 하위 폴더에 있으면 basename만으로는 안전하게 매칭하지 않는다.
    by_unique_basename = {
        name: candidates[0]
        for name, candidates in basename_candidates.items()
        if len(candidates) == 1
    }
    return by_relative_path, by_unique_basename


def lookup_transform(
    filename: str,
    by_relative_path: dict[str, ImageTransform],
    by_unique_basename: dict[str, ImageTransform],
) -> ImageTransform | None:
    """상대 경로를 먼저 찾고, 실패하면 중복 없는 파일명으로 한 번 더 찾는다."""
    normalized = normalized_label_key(filename)
    transform = by_relative_path.get(normalized)
    if transform is not None:
        return transform
    return by_unique_basename.get(Path(normalized).name.casefold())


def replace_jpg_with_png(filename: str) -> str:
    """라벨에 기록된 .jpg/.jpeg 확장자를 대소문자와 무관하게 .png로 바꾼다."""
    return re.sub(r"(?i)\.jpe?g$", ".png", filename)


def update_boxes(value: Any, transform: ImageTransform) -> int:
    """레코드 안의 모든 boxes 객체를 찾아 좌표를 변환한다."""
    updated = 0
    if isinstance(value, dict):
        boxes = value.get("boxes")
        if isinstance(boxes, dict) and all(key in boxes for key in BOX_KEYS):
            try:
                min_x = float(boxes["minX"])
                min_y = float(boxes["minY"])
                max_x = float(boxes["maxX"])
                max_y = float(boxes["maxY"])
                corners = (
                    (min_x, min_y),
                    (max_x, min_y),
                    (min_x, max_y),
                    (max_x, max_y),
                )
                if transform.label_coordinate_space == "raw":
                    oriented_corners = [
                        orient_point(
                            x,
                            y,
                            transform.original_width,
                            transform.original_height,
                            transform.exif_orientation,
                        )
                        for x, y in corners
                    ]
                else:
                    # 이 데이터셋의 라벨은 이미지 뷰어가 EXIF를 적용한 화면 좌표다.
                    # 따라서 좌표를 다시 회전하지 않고 회전 후 크기로만 축소한다.
                    oriented_corners = list(corners)
                oriented_x = [point[0] for point in oriented_corners]
                oriented_y = [point[1] for point in oriented_corners]

                # 축소 배율과 중앙 패딩 오프셋을 적용하고 224×224 범위로 제한한다.
                boxes["minX"] = max(
                    0.0,
                    min(
                        float(transform.output_width),
                        min(oriented_x) * transform.scale_x + transform.offset_x,
                    ),
                )
                boxes["maxX"] = max(
                    0.0,
                    min(
                        float(transform.output_width),
                        max(oriented_x) * transform.scale_x + transform.offset_x,
                    ),
                )
                boxes["minY"] = max(
                    0.0,
                    min(
                        float(transform.output_height),
                        min(oriented_y) * transform.scale_y + transform.offset_y,
                    ),
                )
                boxes["maxY"] = max(
                    0.0,
                    min(
                        float(transform.output_height),
                        max(oriented_y) * transform.scale_y + transform.offset_y,
                    ),
                )
                updated += 1
            except (TypeError, ValueError):
                LOGGER.warning("숫자가 아닌 박스 좌표를 건너뜁니다: %s", boxes)

        for child in value.values():
            updated += update_boxes(child, transform)
    elif isinstance(value, list):
        for child in value:
            updated += update_boxes(child, transform)
    return updated


def update_label_tree(
    value: Any,
    by_relative_path: dict[str, ImageTransform],
    by_unique_basename: dict[str, ImageTransform],
    drop_unmatched: bool,
) -> tuple[int, int, list[str]]:
    """중첩 JSON을 재귀 순회하며 filename과 그 레코드의 박스를 수정한다.

    반환값은 매칭 레코드 수, 변환 박스 수, 매칭 실패 filename 목록이다.
    """
    matched_records = 0
    updated_boxes = 0
    unmatched_filenames: list[str] = []

    if isinstance(value, list):
        children = list(value)
    elif isinstance(value, dict) and "filename" not in value:
        children = list(value.values())
    else:
        children = []

    if isinstance(value, dict) and isinstance(value.get("filename"), str):
        filename = value["filename"]
        transform = lookup_transform(
            filename, by_relative_path, by_unique_basename
        )
        if transform is None:
            unmatched_filenames.append(filename)
        else:
            value["filename"] = replace_jpg_with_png(filename)
            updated_boxes += update_boxes(value, transform)
            matched_records += 1

    kept_children: list[Any] = []
    for child in children:
        child_matched, child_boxes, child_unmatched = update_label_tree(
            child, by_relative_path, by_unique_basename, drop_unmatched
        )
        matched_records += child_matched
        updated_boxes += child_boxes
        unmatched_filenames.extend(child_unmatched)
        direct_filename_record = isinstance(child, dict) and isinstance(
            child.get("filename"), str
        )
        if not (
            drop_unmatched
            and direct_filename_record
            and child_matched == 0
            and child_unmatched
        ):
            kept_children.append(child)

    if isinstance(value, list) and drop_unmatched:
        value[:] = kept_children

    return matched_records, updated_boxes, unmatched_filenames


def process_labels(
    labels_split_dir: Path,
    output_labels_split_dir: Path,
    transforms: list[ImageTransform],
    strict: bool,
    keep_unmatched_labels: bool,
) -> dict[str, Any]:
    """한 split의 모든 JSON을 이미지 변환 결과와 맞춰 새 라벨 폴더에 저장한다."""
    if not labels_split_dir.exists():
        LOGGER.warning("라벨 폴더가 없어 건너뜁니다: %s", labels_split_dir)
        return {
            "json_files": 0,
            "matched_records": 0,
            "updated_boxes": 0,
            "unmatched_records": 0,
        }

    by_relative_path, by_unique_basename = make_transform_lookup(transforms)
    json_paths = sorted(labels_split_dir.rglob("*.json"))
    matched_total = 0
    boxes_total = 0
    unmatched_total: list[str] = []

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

        matched, boxes, unmatched = update_label_tree(
            data,
            by_relative_path,
            by_unique_basename,
            drop_unmatched=not keep_unmatched_labels,
        )
        matched_total += matched
        boxes_total += boxes
        unmatched_total.extend(unmatched)

        output_path = output_labels_split_dir / json_path.relative_to(
            labels_split_dir
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 이미지와 마찬가지로 JSON도 원자적으로 교체해 중단된 파일을 구별한다.
        temporary_path = output_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(output_path)

    if unmatched_total:
        examples = ", ".join(unmatched_total[:5])
        message = (
            f"{labels_split_dir.name}: 이미지와 매칭되지 않은 라벨 "
            f"{len(unmatched_total)}개 (예: {examples})"
        )
        if strict:
            raise RuntimeError(message)
        LOGGER.warning(message)

    return {
        "json_files": len(json_paths),
        "matched_records": matched_total,
        "updated_boxes": boxes_total,
        "unmatched_records": len(unmatched_total),
    }


def process_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    """Training 또는 Validation 하나의 이미지와 라벨을 처리하고 통계를 반환한다."""
    split_started = time.perf_counter()
    raw_split_dir = args.raw_dir / split
    labels_split_dir = args.labels_dir / split
    output_images_split_dir = args.output_dir / "images" / split
    output_labels_split_dir = args.output_dir / "labels" / split

    if not raw_split_dir.exists():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {raw_split_dir}")

    # EMOIMG_기쁨_TRAIN_01 같은 추가 하위 폴더가 있어도 재귀 탐색으로 모두 포함한다.
    images = find_images(raw_split_dir)
    check_output_collisions(images, raw_split_dir)
    relative_paths = [path.relative_to(raw_split_dir) for path in images]
    direct_images = sum(len(path.parts) == 1 for path in relative_paths)
    nested_images = len(images) - direct_images
    source_subdirectories = sorted(
        {path.parts[0] for path in relative_paths if len(path.parts) > 1}
    )
    LOGGER.info(
        "%s 이미지 %d개 처리 시작 (직접 위치 %d개, 하위 폴더 %d개)",
        split,
        len(images),
        direct_images,
        nested_images,
    )
    if source_subdirectories:
        LOGGER.info(
            "%s에서 발견한 원천 하위 폴더: %s",
            split,
            ", ".join(source_subdirectories),
        )

    transforms: list[ImageTransform] = []
    errors: list[dict[str, str]] = []
    # 데이터가 아주 많아도 Future 객체를 전부 한꺼번에 만들지 않도록 묶어서 처리한다.
    batch_size = max(args.workers * 8, 64)
    completed_count = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=configure_pillow_worker,
        initargs=(args.max_megapixels, args.allow_truncated),
    ) as executor:
        for batch_start in range(0, len(images), batch_size):
            batch = images[batch_start : batch_start + batch_size]
            futures = {
                executor.submit(
                    preprocess_one_image,
                    image_path,
                    raw_split_dir,
                    output_images_split_dir,
                    args.size,
                    args.channels,
                    args.pad_value,
                    args.overwrite,
                    args.label_coordinate_space,
                    args.png_compress_level,
                ): image_path
                for image_path in batch
            }
            for future in as_completed(futures):
                image_path = futures[future]
                completed_count += 1
                try:
                    transforms.append(future.result())
                except Exception as exc:  # 한 파일의 오류 때문에 전체 진행 상황을 잃지 않는다.
                    error = {
                        "path": str(image_path),
                        "error_type": type(exc.__cause__ or exc).__name__,
                        "reason": str(exc.__cause__ or exc),
                    }
                    errors.append(error)
                    LOGGER.error(
                        "%s: %s: %s",
                        error["path"],
                        error["error_type"],
                        error["reason"],
                    )
                if completed_count % 500 == 0 or completed_count == len(images):
                    LOGGER.info(
                        "%s 이미지 진행률: %d/%d",
                        split,
                        completed_count,
                        len(images),
                    )

    if errors and args.strict_images:
        raise RuntimeError(
            f"{split} 이미지 {len(errors)}개 처리 실패. "
            f"첫 오류: {errors[0]['path']} ({errors[0]['reason']})"
        )
    if errors:
        LOGGER.warning(
            "%s 손상 이미지 %d개를 제외하고 계속합니다.", split, len(errors)
        )

    label_stats = process_labels(
        labels_split_dir,
        output_labels_split_dir,
        transforms,
        args.strict,
        args.keep_unmatched_labels,
    )
    LOGGER.info(
        "%s 완료: 이미지 %d개, 라벨 레코드 %d개, 박스 %d개",
        split,
        len(transforms),
        label_stats["matched_records"],
        label_stats["updated_boxes"],
    )

    elapsed_seconds = time.perf_counter() - split_started
    rotated_images = sum(
        transform.exif_orientation != 1 for transform in transforms
    )
    print(
        f"[{split}] images={len(transforms)}, "
        f"exif_corrected={rotated_images}, "
        f"failed={len(errors)}, "
        f"elapsed={format_duration(elapsed_seconds)} "
        f"({elapsed_seconds:.2f} sec)",
        flush=True,
    )

    return {
        "images": len(transforms),
        "source_layout": {
            "direct_images": direct_images,
            "nested_images": nested_images,
            "subdirectories": source_subdirectories,
        },
        "exif_corrected_images": rotated_images,
        "failed_images": errors,
        "labels": label_stats,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hhmmss": format_duration(elapsed_seconds),
    }


def main() -> None:
    """요청한 모든 split을 순서대로 처리하고 전체 요약 JSON과 시간을 기록한다."""
    total_started = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    args.raw_dir = args.raw_dir.resolve()
    args.labels_dir = args.labels_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pillow 기본 상한보다 큰 정상 고해상도 데이터가 있어 명시적인 상한을 사용한다.
    Image.MAX_IMAGE_PIXELS = int(args.max_megapixels * 1_000_000)
    ImageFile.LOAD_TRUNCATED_IMAGES = args.allow_truncated

    summary: dict[str, Any] = {
        "settings": {
            "raw_dir": str(args.raw_dir),
            "labels_dir": str(args.labels_dir),
            "output_dir": str(args.output_dir),
            "size": args.size,
            "channels": args.channels,
            "pad_value": args.pad_value,
            "png_compress_level": args.png_compress_level,
            "allow_truncated": args.allow_truncated,
            "keep_unmatched_labels": args.keep_unmatched_labels,
            "max_megapixels": args.max_megapixels,
            "label_coordinate_space": args.label_coordinate_space,
            "splits": args.splits,
        },
        "splits": {},
    }

    for split in args.splits:
        summary["splits"][split] = process_split(args, split)

    total_elapsed_seconds = time.perf_counter() - total_started
    summary["total_elapsed_seconds"] = total_elapsed_seconds
    summary["total_elapsed_hhmmss"] = format_duration(total_elapsed_seconds)

    # 이 파일은 전처리 완료 증빙이자 이후 패키징 단계의 개수 검증 기준이다.
    summary_path = args.output_dir / "preprocessing_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(summary_path)
    LOGGER.info("전체 전처리 완료: %s", args.output_dir)
    print(
        f"[TOTAL] preprocessing complete: "
        f"elapsed={format_duration(total_elapsed_seconds)} "
        f"({total_elapsed_seconds:.2f} sec)",
        flush=True,
    )


if __name__ == "__main__":
    main()
