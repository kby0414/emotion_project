"""전처리된 흑백 감정 이미지로 3개 전이학습 모델을 비교한다.

기본 비교 모델: ResNet-18, EfficientNet-B0, MobileNetV2

Validation에 Training의 클래스가 모두 없으면 공정한 7개 클래스 비교를 위해
Training에서 클래스별 10%를 고정 seed로 분리한다.

장시간 학습을 위해 epoch마다 last.pt/history.json을 원자적으로 저장하고,
검증 macro-F1이 가장 높은 모델은 best.pt로 별도 보존한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import random
import socket
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# 클래스 순서는 학습 target, 혼동행렬, 저장 모델에서 모두 동일해야 한다.
LOGGER = logging.getLogger("emotion-training")
CLASS_NAMES = ("기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
DEFAULT_MODEL_NAMES = ("resnet18", "efficientnet_b0", "mobilenet_v2")
MODEL_NAMES = (*DEFAULT_MODEL_NAMES, "mobilenet_v3_small")
IMAGE_SUFFIXES = {".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    """이미지 경로와 신경망이 사용하는 정수 클래스 번호 한 쌍."""

    path: Path
    target: int


@dataclass
class Metrics:
    """한 epoch에서 계산한 전체 및 클래스별 평가 지표."""

    loss: float
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]


class EmotionDataset(Dataset[tuple[torch.Tensor, int]]):
    """전처리 PNG를 필요할 때 열어 텐서와 정답 번호로 반환한다."""

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        """샘플 목록과 학습/검증용 이미지 변환을 저장한다."""
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """DataLoader가 순회할 전체 이미지 수를 반환한다."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """한 이미지를 읽고 변환한 뒤 (입력 텐서, 정답 번호)를 반환한다."""
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            # 저장 파일은 1채널 흑백이지만 ImageNet 사전학습 모델 입력은 3채널이다.
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.target


def parse_args() -> argparse.Namespace:
    """데이터 경로, 모델 목록, 학습 하이퍼파라미터를 읽고 검증한다."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="ResNet18/EfficientNet-B0/MobileNetV2 감정 분류 비교"
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
        "--report-dir",
        type=Path,
        default=project_root / "results" / "experiments",
        help="GitHub로 공유할 작은 JSON/CSV/PNG 결과 경로",
    )
    parser.add_argument(
        "--run-name",
        help="PC/실험 구분 이름. 생략하면 PC이름과 시각으로 자동 생성",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(DEFAULT_MODEL_NAMES),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="같은 run-name/model의 last.pt에서 중단 학습 재개",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="학습 종료 후 발표용 PNG 자동 생성을 건너뜀",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs/batch-size는 양수이고 workers는 0 이상이어야 합니다.")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio는 0과 1 사이여야 합니다.")
    if args.patience < 1:
        parser.error("--patience는 1 이상이어야 합니다.")
    if args.run_name and (
        Path(args.run_name).name != args.run_name
        or args.run_name in {".", ".."}
    ):
        parser.error("--run-name에는 폴더 구분자를 사용할 수 없습니다.")
    if args.resume and not args.run_name:
        parser.error("--resume에는 기존 --run-name을 함께 지정해야 합니다.")
    return args


def seed_everything(seed: int) -> None:
    """데이터 분리와 가중치 학습의 재현성을 위해 난수 시드를 고정한다."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_write_json(path: Path, value: object) -> None:
    """JSON을 임시 파일에 완성한 뒤 교체하여 중단 시 손상을 줄인다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, value: object) -> None:
    """PyTorch 체크포인트를 임시 파일에 저장한 뒤 최종 이름으로 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def runtime_info(device: torch.device) -> dict[str, str | int | None]:
    """서로 다른 PC의 실험 결과를 비교할 수 있도록 실행 환경을 기록한다."""
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    return {
        "computer_name": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
    }


def label_from_filename(path: Path) -> str | None:
    """데이터 파일명의 네 번째 '_' 구간에서 7개 감정 중 하나를 읽는다."""
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    label = parts[3]
    return label if label in CLASS_TO_INDEX else None


def scan_samples(folder: Path) -> list[Sample]:
    """폴더 아래 PNG를 재귀 탐색하고 파일명의 감정을 학습 target으로 바꾼다."""
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
    """샘플 목록에 실제 포함된 감정 클래스 집합을 반환한다."""
    return {CLASS_NAMES[sample.target] for sample in samples}


def stratified_split(
    samples: Sequence[Sample], val_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """각 감정의 비율을 유지하며 Training을 새 학습/검증 집합으로 나눈다."""
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.target].append(sample)

    generator = random.Random(seed)
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    # 모든 클래스에서 최소 한 장 이상을 검증에 배정해 macro-F1 비교를 가능하게 한다.
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
    """제공 Validation을 쓸지 Training 층화 분리를 쓸지 정책에 따라 결정한다."""
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

    # auto에서는 제공 Validation의 클래스 구성이 Training과 완전히 같을 때만 사용한다.
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
    """학습에는 약한 증강을, 검증에는 결정적인 변환만 적용하도록 구성한다."""
    # 좌우 반전과 작은 이동·회전·확대만 사용해 표정을 유지하면서 과적합을 줄인다.
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
            # 세 모델 모두 ImageNet 사전학습 가중치를 사용하므로 같은 통계로 정규화한다.
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
    """선택한 torchvision 모델을 만들고 마지막 분류층을 7개 감정용으로 교체한다."""
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

    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
        return model

    raise ValueError(f"지원하지 않는 모델: {name}")


def confusion_metrics(confusion: torch.Tensor) -> tuple[float, float, dict]:
    """혼동행렬에서 Accuracy, 클래스별 지표, Macro-F1을 계산한다."""
    confusion = confusion.to(torch.float64)
    total = confusion.sum().item()
    accuracy = confusion.diag().sum().item() / total if total else 0.0
    class_report: dict[str, dict[str, float | int]] = {}
    supported_f1: list[float] = []

    # Macro-F1은 표본이 많은 클래스가 결과를 독점하지 않도록 클래스 F1을 동일 가중 평균한다.
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
    """DataLoader를 한 번 순회하며 학습하거나, 가중치 변경 없이 검증한다.

    optimizer가 전달되면 학습 모드, None이면 검증 모드로 동작한다.
    """
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

        # 검증 시에는 gradient를 만들지 않아 GPU 메모리와 계산량을 줄인다.
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                # CUDA에서는 AMP가 연산을 FP16으로 수행하되 GradScaler가 underflow를 막는다.
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        # (정답, 예측) 쌍을 하나의 번호로 인코딩해 혼동행렬을 반복문 없이 누적한다.
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
        confusion_matrix=confusion.tolist(),
    )


def train_one_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    validation_source: str,
    environment: dict[str, str | int | None],
) -> dict:
    """모델 하나를 학습하고 최고/마지막 체크포인트와 결과 지표를 저장한다."""
    LOGGER.info("%s 학습 시작", model_name)
    model = build_model(
        model_name, len(CLASS_NAMES), pretrained=not args.no_pretrained
    ).to(device)
    # label smoothing과 weight decay는 한 클래스에 지나치게 확신하는 과적합을 완화한다.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # 학습 후반으로 갈수록 학습률을 부드럽게 낮춰 세밀하게 수렴시킨다.
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    model_output = args.output_dir / model_name
    report_output = args.report_dir / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    report_output.mkdir(parents=True, exist_ok=True)
    best_path = model_output / "best.pt"
    last_path = model_output / "last.pt"
    history_path = model_output / "history.json"
    report_history_path = report_output / "history.json"
    summary_path = model_output / "run_summary.json"
    report_summary_path = report_output / "run_summary.json"
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    best_validation: dict | None = None
    stale_epochs = 0
    start_epoch = 1
    previous_elapsed_seconds = 0.0
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.time()

    if args.resume:
        # last.pt에는 optimizer/scheduler까지 있어 중단 직전 상태에서 이어갈 수 있다.
        if not last_path.is_file():
            raise FileNotFoundError(f"재개할 체크포인트가 없습니다: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("model_name") != model_name:
            raise RuntimeError(f"체크포인트 모델이 다릅니다: {last_path}")
        if int(checkpoint.get("target_epochs", -1)) != args.epochs:
            raise RuntimeError(
                "재개할 때 --epochs는 최초 실행과 같아야 합니다: "
                f"checkpoint={checkpoint.get('target_epochs')}, current={args.epochs}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", [])
        best_f1 = float(checkpoint.get("best_macro_f1", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_validation = checkpoint.get("best_validation")
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        previous_elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        started_at = checkpoint.get("started_at", started_at)
        start_epoch = int(checkpoint["epoch"]) + 1
        LOGGER.info("%s 학습 재개: epoch %d부터", model_name, start_epoch)

    completed_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
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
        completed_epoch = epoch
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

        should_stop = False
        # 클래스 불균형에 덜 치우치는 검증 Macro-F1을 최종 모델 선정 기준으로 삼는다.
        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            best_epoch = epoch
            best_validation = asdict(val_metrics)
            stale_epochs = 0
            # best.pt는 추론·웹캠 프로그램에 전달할 최적 가중치와 전처리 정보를 담는다.
            atomic_torch_save(
                best_path,
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "image_size": 224,
                    "channels": 1,
                    "normalization": {
                        "mean": IMAGENET_MEAN,
                        "std": IMAGENET_STD,
                    },
                    "validation_source": validation_source,
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_f1,
                    "best_accuracy": val_metrics.accuracy,
                    "validation_metrics": best_validation,
                    "environment": environment,
                    "run_name": args.run_name,
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                LOGGER.info("%s 조기 종료: %d epoch", model_name, epoch)
                should_stop = True
                stopped_early = True

        elapsed_seconds = previous_elapsed_seconds + (time.time() - started)
        atomic_write_json(history_path, history)
        atomic_write_json(report_history_path, history)
        # last.pt는 최고 모델과 별개로 재개에 필요한 현재 학습 상태 전체를 보관한다.
        atomic_torch_save(
            last_path,
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "target_epochs": args.epochs,
                "history": history,
                "best_epoch": best_epoch,
                "best_macro_f1": best_f1,
                "best_validation": best_validation,
                "stale_epochs": stale_epochs,
                "elapsed_seconds": elapsed_seconds,
                "started_at": started_at,
                "validation_source": validation_source,
                "run_name": args.run_name,
                "environment": environment,
            },
        )
        if should_stop:
            break

    if best_validation is None:
        raise RuntimeError(f"{model_name}에서 유효한 최고 성능을 저장하지 못했습니다.")

    elapsed_seconds = previous_elapsed_seconds + (time.time() - started)
    # 그래프와 팀원 PC 결과 비교에 필요한 값을 작은 JSON으로 따로 남긴다.
    summary = {
        "run_name": args.run_name,
        "model": model_name,
        "target_epochs": args.epochs,
        "completed_epochs": completed_epoch,
        "stopped_early": stopped_early,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_accuracy": best_validation["accuracy"],
        "best_macro_f1": best_f1,
        "best_validation": best_validation,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_minutes": elapsed_seconds / 60,
        "validation_source": validation_source,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": environment,
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": str(history_path),
    }
    atomic_write_json(summary_path, summary)
    atomic_write_json(report_summary_path, summary)
    return {
        "run_name": args.run_name,
        "model": model_name,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epoch,
        "best_accuracy": best_validation["accuracy"],
        "best_macro_f1": best_f1,
        "parameters": summary["parameters"],
        "elapsed_minutes": summary["elapsed_minutes"],
        "checkpoint": str(best_path),
        "history": str(history_path),
    }


def main() -> None:
    """데이터 구성 후 요청 모델을 차례로 학습하고 순위표와 그래프를 생성한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    if not args.no_visualize:
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "시각화를 사용하려면 현재 Python 환경에 matplotlib이 필요합니다. "
                "'python -m pip install -r requirements.txt'를 먼저 실행하거나, "
                "시각화 없이 학습하려면 --no-visualize를 사용하세요."
            ) from error

    args.data_dir = args.data_dir.resolve()
    args.run_name = args.run_name or (
        f"{socket.gethostname()}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    args.output_dir = args.output_dir.resolve() / args.run_name
    args.report_dir = args.report_dir.resolve() / args.run_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
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
    # CUDA가 인식되면 자동으로 GPU를 선택하며 DataLoader도 pinned memory를 사용한다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("사용 장치: %s", device)
    environment = runtime_info(device)
    LOGGER.info("실험 이름: %s", args.run_name)
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

    # 같은 데이터와 하이퍼파라미터를 사용해 모델별 비교 조건을 맞춘다.
    comparison = [
        train_one_model(
            model_name,
            train_loader,
            val_loader,
            args,
            device,
            validation_source,
            environment,
        )
        for model_name in args.models
    ]
    comparison.sort(key=lambda row: row["best_macro_f1"], reverse=True)

    comparison_path = args.output_dir / "model_comparison.csv"
    report_comparison_path = args.report_dir / "model_comparison.csv"
    for path in (comparison_path, report_comparison_path):
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=comparison[0].keys())
            writer.writeheader()
            writer.writerows(comparison)

    LOGGER.info(
        "비교 완료. 최고 모델: %s (macro-F1 %.4f)",
        comparison[0]["model"],
        comparison[0]["best_macro_f1"],
    )
    LOGGER.info("결과: %s", comparison_path)

    if not args.no_visualize:
        try:
            from visualize_results import generate_visualizations

            generated = generate_visualizations(
                [args.report_dir], args.report_dir / "figures"
            )
            LOGGER.info("발표용 시각화 %d개 생성: %s", len(generated), args.report_dir / "figures")
        except Exception:
            # 그림 생성 오류가 성공한 모델 체크포인트를 무효화하지 않게 한다.
            LOGGER.exception(
                "모델 학습은 완료됐지만 시각화 생성에 실패했습니다. "
                "visualize_results.py를 별도로 실행하세요."
            )


if __name__ == "__main__":
    main()
