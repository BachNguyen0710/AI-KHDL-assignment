# Điều phối quá trình train: cross-validation (cv) và huấn luyện mô hình cuối (fit)
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import load_train_test
from src.features import ChurnFeatureEngineer
from src.models import MODEL_NAMES, build_model, requires_scaling
from src.preprocessing import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Nơi lưu pipeline đã huấn luyện trên toàn bộ tập train (dùng cho evaluate/predict).
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

RANDOM_STATE = 42
N_SPLITS = 5

# Chạy tuần tự trước để dễ theo dõi và kiểm tra lỗi.
CV_N_JOBS = 1

BASELINE_MODELS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
)

# Cách xử lý mất cân bằng dữ liệu (Experiment 5).
IMBALANCE_METHODS = ("none", "class_weight", "smote")

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    # Diện tích dưới đường Precision-Recall, nhạy hơn ROC-AUC với lớp thiểu số.
    "pr_auc": "average_precision",
}


def experiment_name(feature_engineering: bool, imbalance: str) -> str:
    """Tên thí nghiệm, dùng để đặt tên thư mục kết quả."""

    name = "feature_engineering" if feature_engineering else "baseline"

    if imbalance != "none":
        name += f"_{imbalance}"

    return name


def build_training_pipeline(
    model_name: str,
    feature_engineering: bool = False,
    imbalance: str = "none",
    balance_ratio: float | None = None,
) -> Pipeline:
    if imbalance not in IMBALANCE_METHODS:
        raise ValueError(f"imbalance phải thuộc {IMBALANCE_METHODS}.")

    if imbalance == "class_weight" and not balance_ratio:
        raise ValueError("class_weight cần balance_ratio (số lớp 0 / số lớp 1).")

    steps = []

    # Tạo đặc trưng mới trước, để preprocessor encode/scale cả cột mới.
    if feature_engineering:
        steps.append(("features", ChurnFeatureEngineer()))

    steps.append(
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=requires_scaling(model_name),
                feature_engineering=feature_engineering,
            ),
        )
    )

    # SMOTE sinh mẫu churn giả sau khi đã encode/scale. Pipeline của imblearn
    # chỉ chạy bước này lúc fit, nên fold validation và tập test giữ nguyên
    # dữ liệu thật.
    if imbalance == "smote":
        from imblearn.over_sampling import SMOTE

        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    steps.append(
        (
            "model",
            build_model(
                model_name,
                random_state=RANDOM_STATE,
                balance_ratio=balance_ratio if imbalance == "class_weight" else None,
            ),
        )
    )

    if imbalance == "smote":
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(steps=steps)

    return Pipeline(steps=steps)


def compute_balance_ratio(y: pd.Series) -> float:
    """Số mẫu không churn / số mẫu churn (khoảng 2,77 với dataset này)."""

    counts = y.value_counts()

    return float(counts[0] / counts[1])


def evaluate_baseline(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list,
    feature_engineering: bool = False,
    imbalance: str = "none",
) -> tuple[pd.DataFrame, dict]:
    """Chạy cross-validation và trả về điểm từng fold, điểm tổng hợp."""

    pipeline = build_training_pipeline(
        model_name,
        feature_engineering=feature_engineering,
        imbalance=imbalance,
        balance_ratio=compute_balance_ratio(y_train),
    )

    scores = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv_splits,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=CV_N_JOBS,
        error_score="raise",
    )

    fold_results = pd.DataFrame(
        {
            "model": [model_name] * len(cv_splits),
            "fold": range(1, len(cv_splits) + 1),
            "fit_time_seconds": scores["fit_time"],
            "score_time_seconds": scores["score_time"],
        }
    )

    summary = {"model": model_name}

    for metric in SCORING:
        for split_name, score_prefix in (
            ("train", "train"),
            ("val", "test"),
        ):
            column_name = f"{split_name}_{metric}"

            fold_results[column_name] = scores[
                f"{score_prefix}_{metric}"
            ]

            summary[f"{column_name}_mean"] = float(
                fold_results[column_name].mean()
            )

            summary[f"{column_name}_std"] = float(
                fold_results[column_name].std(ddof=1)
            )

    summary["fit_time_seconds_mean"] = float(
        fold_results["fit_time_seconds"].mean()
    )

    return fold_results, summary


def run_cross_validation(
    model_names: tuple[str, ...],
    feature_engineering: bool = False,
    imbalance: str = "none",
) -> None:
    """Chạy 5-fold CV trên tập train cho các mô hình và lưu kết quả."""

    X_train, _, y_train, _ = load_train_test()

    print("Dữ liệu train:", X_train.shape)
    print("Số fold:", N_SPLITS)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_splits = list(cv.split(X_train, y_train))

    balance_ratio = compute_balance_ratio(y_train)

    experiment = experiment_name(feature_engineering, imbalance)
    run_id = datetime.now().strftime(f"{experiment}_%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT_ROOT / "outputs" / "experiments" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "experiment": experiment,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "cv_n_jobs": CV_N_JOBS,
        "metrics": list(SCORING),
        "feature_engineering": feature_engineering,
        "imbalance": imbalance,
        "balance_ratio": balance_ratio,
        "train_rows": len(X_train),
        "input_features": X_train.columns.tolist(),
        "models": {
            name: {
                "scale_numeric": requires_scaling(name),
                "parameters": build_model(
                    name,
                    random_state=RANDOM_STATE,
                    balance_ratio=(
                        balance_ratio if imbalance == "class_weight" else None
                    ),
                ).get_params(),
            }
            for name in model_names
        },
    }

    with (output_dir / "config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    all_fold_results = []
    all_summaries = []

    for model_name in model_names:
        print(f"\nĐang chạy: {model_name}", flush=True)

        fold_results, summary = evaluate_baseline(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            feature_engineering=feature_engineering,
            imbalance=imbalance,
        )
        fold_results.to_csv(
            output_dir / f"{model_name}_fold_results.csv",
            index=False,
        )

        all_fold_results.append(fold_results)
        all_summaries.append(summary)

        print(
            f"Validation F1: "
            f"{summary['val_f1_mean']:.4f} "
            f"± {summary['val_f1_std']:.4f}"
        )

        print(
            f"Validation ROC-AUC: "
            f"{summary['val_roc_auc_mean']:.4f} "
            f"± {summary['val_roc_auc_std']:.4f}"
        )

    combined_folds = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    comparison = pd.DataFrame(all_summaries)

    combined_folds.to_csv(
        output_dir / "cv_results.csv",
        index=False,
    )

    comparison.to_csv(
        output_dir / "comparison.csv",
        index=False,
    )

    display_columns = [
        "model",
        "val_accuracy_mean",
        "val_precision_mean",
        "val_recall_mean",
        "val_f1_mean",
        "val_roc_auc_mean",
        "val_pr_auc_mean",
    ]

    print(f"\nKết quả {experiment}:")
    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nĐã lưu:", output_dir.relative_to(PROJECT_ROOT))

def fit_and_save(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_dir: Path = MODEL_DIR,
    feature_engineering: bool = False,
    imbalance: str = "none",
) -> Path:
    """Huấn luyện pipeline trên toàn bộ tập train rồi lưu bằng joblib.

    File .joblib chứa cả bộ tiền xử lý lẫn mô hình, nên predict/evaluate
    chỉ cần nạp lại file này, không phải tiền xử lý lại thủ công.
    """

    pipeline = build_training_pipeline(
        model_name,
        feature_engineering=feature_engineering,
        imbalance=imbalance,
        balance_ratio=compute_balance_ratio(y_train),
    )
    pipeline.fit(X_train, y_train)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model": model_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "train_rows": len(X_train),
        "feature_engineering": feature_engineering,
        "imbalance": imbalance,
        "input_features": X_train.columns.tolist(),
        "sklearn_version": sklearn.__version__,
        "parameters": pipeline.named_steps["model"].get_params(),
    }

    with (model_dir / f"{model_name}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return model_path


def run_fit(
    model_names: tuple[str, ...],
    feature_engineering: bool = False,
    imbalance: str = "none",
) -> None:
    X_train, _, y_train, _ = load_train_test()

    # Mỗi cấu hình một thư mục con để không ghi đè pipeline baseline,
    # ví dụ outputs/models/fe, outputs/models/class_weight, outputs/models/fe_smote.
    parts = (["fe"] if feature_engineering else []) + (
        [imbalance] if imbalance != "none" else []
    )
    model_dir = MODEL_DIR / "_".join(parts) if parts else MODEL_DIR

    print("Dữ liệu train:", X_train.shape)

    for model_name in model_names:
        print(f"\nĐang huấn luyện: {model_name}", flush=True)

        model_path = fit_and_save(
            model_name,
            X_train,
            y_train,
            model_dir=model_dir,
            feature_engineering=feature_engineering,
            imbalance=imbalance,
        )

        print("Đã lưu pipeline:", model_path.relative_to(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    for command, help_text in (
        ("cv", "Cross-validation trên tập train (mặc định)"),
        ("fit", "Huấn luyện trên toàn bộ tập train và lưu pipeline"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--models",
            nargs="+",
            choices=MODEL_NAMES,
            default=None,
            help="Danh sách mô hình (cv mặc định: cả 4 baseline; "
            "fit mặc định: tất cả).",
        )
        subparser.add_argument(
            "--feature-engineering",
            action="store_true",
            help="Thêm các đặc trưng mới (Experiment 2).",
        )
        subparser.add_argument(
            "--imbalance",
            choices=IMBALANCE_METHODS,
            default="none",
            help="Xử lý mất cân bằng dữ liệu (Experiment 5).",
        )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Không truyền lệnh con => chạy cross-validation cho cả 4 baseline.
    command = args.command or "cv"
    models = getattr(args, "models", None)
    feature_engineering = getattr(args, "feature_engineering", False)
    imbalance = getattr(args, "imbalance", "none")

    if command == "cv":
        run_cross_validation(
            tuple(models) if models else BASELINE_MODELS,
            feature_engineering=feature_engineering,
            imbalance=imbalance,
        )
    else:
        run_fit(
            tuple(models) if models else MODEL_NAMES,
            feature_engineering=feature_engineering,
            imbalance=imbalance,
        )


if __name__ == "__main__":
    main()