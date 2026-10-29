# Điều phối quá trình train và đánh giá cross-validation
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import load_train_test
from src.models import build_model, requires_scaling
from src.preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42
N_SPLITS = 5

# Chạy tuần tự trước để dễ theo dõi và kiểm tra lỗi.
CV_N_JOBS = 1

BASELINE_MODELS = (
    "logistic_regression",
    "xgboost",
)

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}

def build_training_pipeline(model_name: str) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    scale_numeric=requires_scaling(model_name),
                ),
            ),
            (
                "model",
                build_model(
                    model_name,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

def evaluate_baseline(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list,
) -> tuple[pd.DataFrame, dict]:
    """Chạy cross-validation và trả về điểm từng fold, điểm tổng hợp."""

    pipeline = build_training_pipeline(model_name)

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


def main() -> None:


    X_train, _, y_train, _ = load_train_test()

    print("Dữ liệu train:", X_train.shape)
    print("Số fold:", N_SPLITS)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_splits = list(cv.split(X_train, y_train))

    run_id = datetime.now().strftime("baseline_%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT_ROOT / "outputs" / "experiments" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "experiment": "baseline",
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "cv_n_jobs": CV_N_JOBS,
        "metrics": list(SCORING),
        "feature_engineering": False,
        "resampling": None,
        "train_rows": len(X_train),
        "input_features": X_train.columns.tolist(),
        "models": {
            name: {
                "scale_numeric": requires_scaling(name),
                "parameters": build_model(
                    name,
                    random_state=RANDOM_STATE,
                ).get_params(),
            }
            for name in BASELINE_MODELS
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

    for model_name in BASELINE_MODELS:
        print(f"\nĐang chạy: {model_name}", flush=True)

        fold_results, summary = evaluate_baseline(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
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
    ]

    print("\nKết quả baseline:")
    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

if __name__ == "__main__":
    main()