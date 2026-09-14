"""학습 history/run_summary를 발표용 PNG와 비교표로 변환한다."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

# 화면이 없는 터미널/원격 PC에서도 PNG를 만들 수 있는 렌더링 백엔드를 사용한다.
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ("기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립")


def configure_font() -> None:
    """운영체제에 설치된 한글 글꼴을 찾아 그래프의 글자 깨짐을 방지한다."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def safe_name(value: str) -> str:
    """실험 이름을 Windows에서도 안전한 파일명으로 정리한다."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "experiment"


def find_history_files(inputs: Iterable[Path]) -> list[Path]:
    """파일 또는 폴더 입력에서 모든 history.json을 중복 없이 찾는다."""
    found: set[Path] = set()
    for value in inputs:
        path = value.resolve()
        if path.is_file() and path.name == "history.json":
            found.add(path)
        elif path.is_dir():
            found.update(item.resolve() for item in path.rglob("history.json"))
    return sorted(found)


def load_experiments(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    """history와 run_summary를 합치고 검증 Macro-F1이 가장 높은 epoch를 찾는다."""
    experiments: list[dict[str, Any]] = []
    for history_path in find_history_files(inputs):
        history = json.loads(history_path.read_text(encoding="utf-8-sig"))
        if not history:
            continue
        summary_path = history_path.with_name("run_summary.json")
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8-sig"))
            if summary_path.is_file()
            else {}
        )
        model = str(summary.get("model") or history_path.parent.name)
        run_name = str(summary.get("run_name") or history_path.parent.parent.name)
        # 마지막 epoch가 아니라 학습 중 검증 Macro-F1이 최고인 epoch를 비교한다.
        best = max(history, key=lambda row: row["validation"]["macro_f1"])
        label = f"{run_name}/{model}"
        experiments.append(
            {
                "label": label,
                "run_name": run_name,
                "model": model,
                "history": history,
                "best": best,
                "summary": summary,
                "history_path": history_path,
            }
        )
    if not experiments:
        raise RuntimeError("입력 경로에서 history.json을 찾지 못했습니다.")
    return experiments


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """발표 자료에 사용할 고해상도 PNG를 저장하고 figure 메모리를 해제한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def comparison_labels(experiments: list[dict[str, Any]]) -> list[str]:
    """모델이 서로 다르면 발표 그래프에는 간결한 모델명만 표시한다."""
    model_names = [experiment["model"] for experiment in experiments]
    if len(set(model_names)) == len(model_names):
        return model_names
    return [experiment["label"] for experiment in experiments]


def plot_training_curves(experiment: dict[str, Any], output_dir: Path) -> Path:
    """epoch별 Loss, Accuracy, Macro-F1, 학습률 변화를 한 장에 그린다."""
    history = experiment["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"학습 곡선 — {experiment['label']}", fontsize=15)

    axes[0, 0].plot(epochs, [row["train"]["loss"] for row in history], marker="o", label="Train")
    axes[0, 0].plot(epochs, [row["validation"]["loss"] for row in history], marker="o", label="Validation")
    axes[0, 0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")

    axes[0, 1].plot(epochs, [row["train"]["accuracy"] for row in history], marker="o", label="Train")
    axes[0, 1].plot(epochs, [row["validation"]["accuracy"] for row in history], marker="o", label="Validation")
    axes[0, 1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))

    axes[1, 0].plot(epochs, [row["train"]["macro_f1"] for row in history], marker="o", label="Train")
    axes[1, 0].plot(epochs, [row["validation"]["macro_f1"] for row in history], marker="o", label="Validation")
    axes[1, 0].set(title="Macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))

    axes[1, 1].plot(epochs, [row["learning_rate"] for row in history], marker="o")
    axes[1, 1].set(title="Learning rate", xlabel="Epoch", ylabel="Learning rate")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        if len(axis.lines) > 1:
            axis.legend()
    fig.tight_layout()
    name = safe_name(experiment["label"])
    return save_figure(fig, output_dir / f"training_curves_{name}.png")


def plot_confusion_matrix(experiment: dict[str, Any], output_dir: Path) -> Path | None:
    """최고 epoch의 혼동행렬을 실제 클래스별 비율로 정규화해 그린다."""
    matrix_value = experiment["best"]["validation"].get("confusion_matrix")
    if not matrix_value:
        return None
    matrix = np.asarray(matrix_value, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    # 클래스마다 데이터 수가 달라도 색상 강도를 공정하게 비교하도록 행별 정규화한다.
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_title(
        f"정규화 혼동행렬 — {experiment['label']} (best epoch {experiment['best']['epoch']})"
    )
    axis.set_xlabel("예측 클래스")
    axis.set_ylabel("실제 클래스")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="행 기준 비율")
    fig.tight_layout()
    name = safe_name(experiment["label"])
    return save_figure(fig, output_dir / f"confusion_matrix_{name}.png")


def plot_model_comparison(experiments: list[dict[str, Any]], output_dir: Path) -> Path:
    """모델별 최고 검증 Accuracy와 Macro-F1을 나란한 막대로 비교한다."""
    labels = comparison_labels(experiments)
    accuracy = [experiment["best"]["validation"]["accuracy"] for experiment in experiments]
    macro_f1 = [experiment["best"]["validation"]["macro_f1"] for experiment in experiments]
    positions = np.arange(len(experiments))
    width = 0.36

    fig, axis = plt.subplots(figsize=(max(10, len(experiments) * 2.2), 6.5))
    bars_accuracy = axis.bar(positions - width / 2, accuracy, width, label="Accuracy")
    bars_f1 = axis.bar(positions + width / 2, macro_f1, width, label="Macro-F1")
    axis.set(title="최고 검증 성능 비교", xlabel="실험 / 모델", ylabel="Score", ylim=(0, 1))
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(bars_accuracy, fmt="%.3f", padding=3)
    axis.bar_label(bars_f1, fmt="%.3f", padding=3)
    fig.tight_layout()
    return save_figure(fig, output_dir / "model_comparison.png")


def plot_per_class_f1(experiments: list[dict[str, Any]], output_dir: Path) -> Path:
    """모델마다 어떤 감정에 강하고 약한지 클래스별 F1 막대로 보여 준다."""
    positions = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(experiments)
    labels = comparison_labels(experiments)
    fig, axis = plt.subplots(figsize=(13, 6.5))
    for index, experiment in enumerate(experiments):
        values = [
            experiment["best"]["validation"]["per_class"][name]["f1"]
            for name in CLASS_NAMES
        ]
        offset = (index - (len(experiments) - 1) / 2) * width
        axis.bar(positions + offset, values, width, label=labels[index])
    axis.set(title="클래스별 검증 F1 비교", xlabel="감정 클래스", ylabel="F1", ylim=(0, 1))
    axis.set_xticks(positions, CLASS_NAMES)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(3, len(experiments)))
    fig.tight_layout()
    return save_figure(fig, output_dir / "per_class_f1.png")


def write_comparison_files(experiments: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Macro-F1 순위와 환경·시간·파라미터 수를 CSV와 발표용 Markdown으로 저장한다."""
    # 학습 코드와 동일하게 Macro-F1 내림차순을 공식 모델 순위로 사용한다.
    ranked = sorted(
        experiments,
        key=lambda item: item["best"]["validation"]["macro_f1"],
        reverse=True,
    )
    csv_path = output_dir / "comparison_table.csv"
    fields = (
        "rank",
        "run_name",
        "model",
        "completed_epochs",
        "best_epoch",
        "best_accuracy",
        "best_macro_f1",
        "parameters",
        "elapsed_minutes",
        "gpu_name",
        "history_path",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, experiment in enumerate(ranked, start=1):
            summary = experiment["summary"]
            writer.writerow(
                {
                    "rank": rank,
                    "run_name": experiment["run_name"],
                    "model": experiment["model"],
                    "completed_epochs": summary.get("completed_epochs", ""),
                    "best_epoch": experiment["best"]["epoch"],
                    "best_accuracy": experiment["best"]["validation"]["accuracy"],
                    "best_macro_f1": experiment["best"]["validation"]["macro_f1"],
                    "parameters": summary.get("parameters", ""),
                    "elapsed_minutes": summary.get("elapsed_minutes", ""),
                    "gpu_name": summary.get("environment", {}).get("gpu_name", "unknown"),
                    "history_path": str(experiment["history_path"]),
                }
            )

    markdown_path = output_dir / "presentation_summary.md"
    lines = [
        "# 모델 비교 요약",
        "",
        "| 순위 | 모델 | 완료/Best epoch | Accuracy | Macro-F1 | Params(M) | 학습 시간(분) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, experiment in enumerate(ranked, start=1):
        best = experiment["best"]
        summary = experiment["summary"]
        parameters_m = float(summary.get("parameters", 0)) / 1_000_000
        elapsed_minutes = float(summary.get("elapsed_minutes", 0))
        lines.append(
            f"| {rank} | {experiment['model']} | "
            f"{summary.get('completed_epochs', '-')}/{best['epoch']} | "
            f"{best['validation']['accuracy']:.4f} | "
            f"{best['validation']['macro_f1']:.4f} | "
            f"{parameters_m:.2f} | {elapsed_minutes:.1f} |"
        )
    lines.extend(
        [
            "",
            "최종 모델은 Accuracy만이 아니라 Macro-F1과 클래스별 F1을 함께 확인해 선택한다.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, markdown_path]


def generate_visualizations(inputs: Iterable[Path], output_dir: Path) -> list[Path]:
    """찾은 모든 실험의 개별 그래프와 통합 비교 자료를 한 번에 생성한다."""
    configure_font()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(inputs)
    generated: list[Path] = []
    for experiment in experiments:
        generated.append(plot_training_curves(experiment, output_dir))
        confusion_path = plot_confusion_matrix(experiment, output_dir)
        if confusion_path is not None:
            generated.append(confusion_path)
    generated.append(plot_model_comparison(experiments, output_dir))
    generated.append(plot_per_class_f1(experiments, output_dir))
    generated.extend(write_comparison_files(experiments, output_dir))
    return generated


def parse_args() -> argparse.Namespace:
    """분석할 실험 경로와 그래프 출력 폴더를 읽는다."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="학습 결과 발표용 그래프 생성")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="history.json 또는 이를 포함한 폴더들",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "figures",
    )
    args = parser.parse_args()
    if not args.inputs:
        experiment_dir = project_root / "results" / "experiments"
        args.inputs = [experiment_dir if experiment_dir.exists() else project_root / "models"]
    return args


def main() -> None:
    """명령행 입력으로 시각화를 생성하고 만들어진 파일 위치를 출력한다."""
    args = parse_args()
    generated = generate_visualizations(args.inputs, args.output_dir)
    for path in generated:
        print(f"[FIGURE] {path}")
    print(f"[DONE] {len(generated)} files")


if __name__ == "__main__":
    main()
