# Đánh giá pipeline đã lưu trên tập test.
#
# Tập test chỉ nên dùng để báo cáo kết quả cuối cùng, không dùng để chọn
# mô hình hay chỉnh siêu tham số (việc đó làm bằng cross-validation).
#
#   python -m src.train fit --models decision_tree random_forest
#   python -m src.evaluate --models decision_tree random_forest
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
    confusion_matrix,
    f1_score,
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


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, pd.Series, pd.Series]:
    """Trả về (các chỉ số, nhãn dự đoán, xác suất churn)."""

    y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index)
    y_proba = pd.Series(
        pipeline.predict_proba(X_test)[:, 1],
        index=y_test.index,
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
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

    print("Dữ liệu test:", X_test.shape)

    all_metrics = []
    probabilities = {}
    matrices = {}

    for model_name in model_names:
        pipeline = load_pipeline(model_name, args.model_dir)

        metrics, _, y_proba = evaluate_pipeline(pipeline, X_test, y_test)

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
                "matrices": matrices,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    plot_results(y_test, probabilities, matrices, output_dir / "figures")

    display_columns = [
        "model",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
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
