"""전처리된 흑백 감정 이미지로 3개 전이학습 모델을 비교한다.

기본 비교 모델: ResNet-18, EfficientNet-B0, MobileNetV3-Small

Validation에 Training의 클래스가 모두 없으면(현재 기쁨 누락), 공정한 7개 클래스
비교를 위해 Training에서 클래스별 10%를 고정 seed로 분리한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LOGGER = logging.getLogger("emotion-training")
CLASS_NAMES = ("기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
MODEL_NAMES = ("resnet18", "efficientnet_b0", "mobilenet_v3_small")
IMAGE_SUFFIXES = {".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    target: int


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float | int]]


class EmotionDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            # 저장 파일은 1채널 흑백이지만 ImageNet 사전학습 모델 입력은 3채널이다.
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.target


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="ResNet18/EfficientNet-B0/MobileNetV3-Small 감정 분류 비교"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "dataset" / "processed" / "images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--validation-source",
        choices=("auto", "provided", "stratified"),
        default="auto",
        help="auto는 제공 Validation의 클래스가 불완전하면 Training에서 층화 분리",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="ImageNet 사전학습 가중치를 내려받지 않고 처음부터 학습",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs/batch-size는 양수이고 workers는 0 이상이어야 합니다.")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio는 0과 1 사이여야 합니다.")
    if args.patience < 1:
        parser.error("--patience는 1 이상이어야 합니다.")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def label_from_filename(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    label = parts[3]
    return label if label in CLASS_TO_INDEX else None


def scan_samples(folder: Path) -> list[Sample]:
    if not folder.exists():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {folder}")

    samples: list[Sample] = []
    unknown: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        label = label_from_filename(path)
        if label is None:
            unknown.append(path)
        else:
            samples.append(Sample(path=path, target=CLASS_TO_INDEX[label]))

    if unknown:
        LOGGER.warning(
            "파일명에서 감정 라벨을 읽지 못한 PNG %d개를 제외합니다. 예: %s",
            len(unknown),
            unknown[0],
        )
    if not samples:
        raise RuntimeError(f"학습 가능한 PNG가 없습니다: {folder}")
    return samples


def present_class_names(samples: Sequence[Sample]) -> set[str]:
    return {CLASS_NAMES[sample.target] for sample in samples}


def stratified_split(
    samples: Sequence[Sample], val_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.target].append(sample)

    generator = random.Random(seed)
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    for target in range(len(CLASS_NAMES)):
        group = grouped[target]
        if len(group) < 2:
            raise RuntimeError(
                f"층화 분리에 필요한 {CLASS_NAMES[target]} 데이터가 부족합니다: {len(group)}개"
            )
        generator.shuffle(group)
        val_count = max(1, round(len(group) * val_ratio))
        val_samples.extend(group[:val_count])
        train_samples.extend(group[val_count:])

    generator.shuffle(train_samples)
    generator.shuffle(val_samples)
    return train_samples, val_samples


def choose_train_and_validation(
    data_dir: Path,
    validation_source: str,
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], str]:
    all_training = scan_samples(data_dir / "Training")
    provided_folder = data_dir / "Validation"
    provided = scan_samples(provided_folder) if provided_folder.exists() else []
    training_classes = present_class_names(all_training)
    provided_classes = present_class_names(provided) if provided else set()

    if validation_source == "provided":
        if not provided:
            raise RuntimeError("제공된 Validation 이미지가 없습니다.")
        missing = training_classes - provided_classes
        if missing:
            LOGGER.warning(
                "제공 Validation에 다음 클래스가 없습니다: %s. 해당 클래스 성능은 평가되지 않습니다.",
                ", ".join(sorted(missing)),
            )
        return all_training, provided, "provided"

    use_stratified = validation_source == "stratified" or (
        validation_source == "auto" and training_classes != provided_classes
    )
    if use_stratified:
        missing = training_classes - provided_classes
        if validation_source == "auto":
            LOGGER.warning(
                "제공 Validation 클래스가 Training과 다릅니다(누락: %s). "
                "Training에서 클래스별 %.1f%%를 검증용으로 분리합니다.",
                ", ".join(sorted(missing)) or "없음",
                val_ratio * 100,
            )
        train, val = stratified_split(all_training, val_ratio, seed)
        return train, val, "stratified_from_training"

    return all_training, provided, "provided"


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.04, 0.04),
                scale=(0.96, 1.04),
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
        return model

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
        return model

    raise ValueError(f"지원하지 않는 모델: {name}")


def confusion_metrics(confusion: torch.Tensor) -> tuple[float, float, dict]:
    confusion = confusion.to(torch.float64)
    total = confusion.sum().item()
    accuracy = confusion.diag().sum().item() / total if total else 0.0
    class_report: dict[str, dict[str, float | int]] = {}
    supported_f1: list[float] = []

    for index, name in enumerate(CLASS_NAMES):
        true_positive = confusion[index, index].item()
        support = int(confusion[index, :].sum().item())
        predicted = confusion[:, index].sum().item()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            supported_f1.append(f1)
        class_report[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0
    return accuracy, macro_f1, class_report


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
) -> Metrics:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64
    )
    loss_sum = 0.0
    sample_count = 0

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        encoded = targets.detach().cpu() * len(CLASS_NAMES) + predictions.detach().cpu()
        confusion += torch.bincount(
            encoded, minlength=len(CLASS_NAMES) ** 2
        ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))

        if training and batch_index % 200 == 0:
            LOGGER.info("학습 배치 %d/%d", batch_index, len(loader))

    accuracy, macro_f1, class_report = confusion_metrics(confusion)
    return Metrics(
        loss=loss_sum / sample_count,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=class_report,
    )


def train_one_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    validation_source: str,
) -> dict:
    LOGGER.info("%s 학습 시작", model_name)
    model = build_model(
        model_name, len(CLASS_NAMES), pretrained=not args.no_pretrained
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    model_output = args.output_dir / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    best_path = model_output / "best.pt"
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, optimizer=None, scaler=None
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": asdict(train_metrics),
            "validation": asdict(val_metrics),
        }
        history.append(row)
        LOGGER.info(
            "%s epoch %d/%d | train loss %.4f acc %.4f | "
            "val loss %.4f acc %.4f macro-F1 %.4f",
            model_name,
            epoch,
            args.epochs,
            train_metrics.loss,
            train_metrics.accuracy,
            val_metrics.loss,
            val_metrics.accuracy,
            val_metrics.macro_f1,
        )

        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": 224,
                    "normalization": {
                        "mean": IMAGENET_MEAN,
                        "std": IMAGENET_STD,
                    },
                    "validation_source": validation_source,
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_f1,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                LOGGER.info("%s 조기 종료: %d epoch", model_name, epoch)
                break

    history_path = model_output / "history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_minutes": (time.time() - started) / 60,
        "checkpoint": str(best_path),
        "history": str(history_path),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    train_samples, val_samples, validation_source = choose_train_and_validation(
        args.data_dir,
        args.validation_source,
        args.val_ratio,
        args.seed,
    )
    LOGGER.info(
        "학습 %d개 / 검증 %d개 / 검증 방식: %s",
        len(train_samples),
        len(val_samples),
        validation_source,
    )

    train_transform, val_transform = make_transforms()
    persistent_workers = args.workers > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("사용 장치: %s", device)
    if device.type != "cuda":
        LOGGER.warning("GPU를 찾지 못했습니다. 전체 모델 비교에는 시간이 오래 걸립니다.")

    train_loader = DataLoader(
        EmotionDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        EmotionDataset(val_samples, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
    )

    comparison = [
        train_one_model(
            model_name,
            train_loader,
            val_loader,
            args,
            device,
            validation_source,
        )
        for model_name in args.models
    ]
    comparison.sort(key=lambda row: row["best_macro_f1"], reverse=True)

    comparison_path = args.output_dir / "model_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    LOGGER.info(
        "비교 완료. 최고 모델: %s (macro-F1 %.4f)",
        comparison[0]["model"],
        comparison[0]["best_macro_f1"],
    )
    LOGGER.info("결과: %s", comparison_path)


if __name__ == "__main__":
    main()
