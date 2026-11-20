# Đánh giá pipeline đã lưu trên tập test.
#
# Tập test chỉ nên dùng để báo cáo kết quả cuối cùng, không dùng để chọn
# mô hình hay chỉnh siêu tham số (việc đó làm bằng cross-validation).
#
#   python -m src.train fit --models decision_tree random_forest
#   python -m src.evaluate --models decision_tree random_forest
#
# Mô hình đã tune + ngưỡng chọn ở bước 4 (đọc <model-dir>/thresholds.json):
#   python -m src.evaluate --model-dir outputs/models/tuned
#   python -m src.evaluate --model-dir outputs/models/tuned \
#       --thresholds outputs/thresholds/<run beta=2>/thresholds.json
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.data import load_train_test
from src.models import MODEL_NAMES
from src.train import MODEL_DIR, PROJECT_ROOT

EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def load_pipeline(
    model_name: str,
    model_dir: Path = MODEL_DIR,
) -> Pipeline:
    model_path = Path(model_dir) / f"{model_name}.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Chưa có pipeline đã lưu: {model_path}. "
            f"Hãy chạy: python -m src.train fit --models {model_name}"
        )

    return joblib.load(model_path)


def load_thresholds(
    path: Path | None,
    model_names: list[str],
) -> tuple[dict[str, float], str]:
    """Đọc ngưỡng từ thresholds.json (bước 4). Không có file -> dùng 0.5."""

    if path is None or not Path(path).is_file():
        return {name: 0.5 for name in model_names}, "mặc định 0.5"

    with Path(path).open(encoding="utf-8") as file:
        saved = json.load(file)["thresholds"]

    missing = [name for name in model_names if name not in saved]
    if missing:
        raise ValueError(f"{path} không có ngưỡng cho: {missing}")

    return {name: float(saved[name]) for name in model_names}, str(path)


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple[dict, pd.Series, pd.Series]:
    """Trả về (các chỉ số, nhãn dự đoán, xác suất churn).

    Dự đoán "rời" khi xác suất > threshold (cùng quy ước với predict.py).
    """

    y_proba = pd.Series(
        pipeline.predict_proba(X_test)[:, 1],
        index=y_test.index,
    )
    y_pred = (y_proba > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "threshold": threshold,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        # Để tham chiếu: F1 nếu vẫn dùng ngưỡng 0.5.
        "test_f1_at_0.5": f1_score(
            y_test, (y_proba > 0.5).astype(int), zero_division=0
        ),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
        "test_pr_auc": average_precision_score(y_test, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, y_pred, y_proba


def plot_results(
    y_test: pd.Series,
    probabilities: dict[str, pd.Series],
    matrices: dict[str, list[list[int]]],
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Đường ROC của các mô hình trên cùng một hình.
    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name, y_proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS[model_name]} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — tập test")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(figure_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Đường Precision-Recall: hợp với dữ liệu lệch lớp hơn ROC.
    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name, y_proba in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{MODEL_LABELS[model_name]} (AP = {ap:.4f})")

    ax.axhline(y_test.mean(), linestyle="--", color="grey", label="Đoán ngẫu nhiên")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — tập test")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(figure_dir / "pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Ma trận nhầm lẫn, mỗi mô hình một ô.
    fig, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(5 * len(matrices), 4.5),
        squeeze=False,
    )

    for ax, (model_name, matrix) in zip(axes[0], matrices.items()):
        ax.imshow(matrix, cmap="Blues")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, matrix[i][j], ha="center", va="center")

        ax.set_xticks([0, 1], labels=["Không rời", "Rời"])
        ax.set_yticks([0, 1], labels=["Không rời", "Rời"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(MODEL_LABELS[model_name])

    fig.tight_layout()
    fig.savefig(
        figure_dir / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="Mô hình cần đánh giá (mặc định: mọi mô hình đã có file .joblib).",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="File thresholds.json từ bước 4 (mặc định: <model-dir>/thresholds.json "
        "nếu có, không thì dùng 0.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục lưu kết quả (mặc định: outputs/evaluation/test_<thời gian>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_names = args.models or [
        name
        for name in MODEL_NAMES
        if (args.model_dir / f"{name}.joblib").is_file()
    ]

    if not model_names:
        raise FileNotFoundError(
            f"Không có pipeline nào trong {args.model_dir}. "
            "Hãy chạy: python -m src.train fit"
        )

    _, X_test, _, y_test = load_train_test()

    thresholds, threshold_source = load_thresholds(
        args.thresholds or args.model_dir / "thresholds.json",
        model_names,
    )

    print("Dữ liệu test:", X_test.shape)
    print("Ngưỡng:", threshold_source, thresholds)

    all_metrics = []
    probabilities = {}
    matrices = {}

    for model_name in model_names:
        pipeline = load_pipeline(model_name, args.model_dir)

        metrics, _, y_proba = evaluate_pipeline(
            pipeline, X_test, y_test, threshold=thresholds[model_name]
        )

        all_metrics.append({"model": model_name, **metrics})
        probabilities[model_name] = y_proba
        matrices[model_name] = [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]

    results = pd.DataFrame(all_metrics)

    output_dir = args.output_dir or (
        EVALUATION_DIR / datetime.now().strftime("test_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    results.to_csv(output_dir / "test_metrics.csv", index=False)

    with (output_dir / "confusion_matrices.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "labels": ["Không rời (0)", "Rời (1)"],
                "layout": "[[TN, FP], [FN, TP]]",
                "model_dir": str(args.model_dir),
                "threshold_source": threshold_source,
                "thresholds": thresholds,
                "matrices": matrices,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    plot_results(y_test, probabilities, matrices, output_dir / "figures")

    display_columns = [
        "model",
        "threshold",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_f1_at_0.5",
        "test_roc_auc",
        "test_pr_auc",
    ]

    print("\nKết quả trên tập test:")
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    try:
        shown_dir = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_dir = output_dir

    print("\nĐã lưu kết quả:", shown_dir)


if __name__ == "__main__":
    main()