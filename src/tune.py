# Tuning siêu tham số bằng RandomizedSearchCV trên đúng 5 fold của train.py.
#
#   python -m src.tune                                  # cả 4 mô hình, class_weight
#   python -m src.tune --models logistic_regression xgboost --n-iter 40
#   python -m src.tune --fit                            # lưu pipeline tốt nhất
#
# Kết quả lưu cùng định dạng với outputs/experiments, nên so sánh được bằng
# src.compare với run baseline_class_weight.
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.data import load_train_test
from src.models import MODEL_NAMES
from src.train import (
    IMBALANCE_METHODS,
    MODEL_DIR,
    N_SPLITS,
    PROJECT_ROOT,
    RANDOM_STATE,
    SCORING,
    build_training_pipeline,
    compute_balance_ratio,
)

TUNING_DIR = PROJECT_ROOT / "outputs" / "tuning"

# Không gian tìm kiếm. Tiền tố "model__" trỏ tới bước "model" trong Pipeline.
PARAM_SPACES = {
    "logistic_regression": {
        # C nhỏ = regularization mạnh. Thang log vì C trải nhiều bậc độ lớn.
        "model__C": loguniform(1e-3, 1e2),
    },
    "decision_tree": {
        "model__max_depth": randint(2, 16),
        "model__min_samples_leaf": randint(1, 60),
        "model__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "model__n_estimators": randint(200, 601),
        "model__max_depth": [4, 6, 8, 10, 12, 16, None],
        "model__min_samples_leaf": randint(1, 30),
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "model__n_estimators": randint(100, 601),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(2, 8),
        "model__min_child_weight": randint(1, 11),
        "model__subsample": uniform(0.6, 0.4),          # 0.6 – 1.0
        "model__colsample_bytree": uniform(0.6, 0.4),   # 0.6 – 1.0
        "model__reg_lambda": loguniform(0.1, 10),
    },
}

# Số tổ hợp thử cho mỗi mô hình (LR chỉ có 1 tham số nên ít hơn).
DEFAULT_N_ITER = {
    "logistic_regression": 30,
    "decision_tree": 50,
    "random_forest": 40,
    "xgboost": 60,
}


def fold_results_from_search(
    model_name: str,
    search: RandomizedSearchCV,
) -> tuple[pd.DataFrame, dict]:
    """Lấy điểm từng fold của tổ hợp tốt nhất, cùng định dạng với train.py."""

    results = search.cv_results_
    best = search.best_index_

    fold_results = pd.DataFrame(
        {"model": model_name, "fold": range(1, N_SPLITS + 1)}
    )
    summary = {"model": model_name}

    for metric in SCORING:
        for split_name, prefix in (("train", "train"), ("val", "test")):
            column = f"{split_name}_{metric}"
            fold_results[column] = [
                results[f"split{i}_{prefix}_{metric}"][best]
                for i in range(N_SPLITS)
            ]
            summary[f"{column}_mean"] = float(fold_results[column].mean())
            summary[f"{column}_std"] = float(fold_results[column].std(ddof=1))

    return fold_results, summary


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list,
    refit: str,
    n_iter: int,
    feature_engineering: bool,
    imbalance: str,
) -> RandomizedSearchCV:
    pipeline = build_training_pipeline(
        model_name,
        feature_engineering=feature_engineering,
        imbalance=imbalance,
        balance_ratio=compute_balance_ratio(y_train),
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_SPACES[model_name],
        n_iter=n_iter,
        scoring=SCORING,
        refit=refit,                # metric dùng để chọn tổ hợp tốt nhất
        cv=cv_splits,               # đúng 5 fold như các thí nghiệm trước
        return_train_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,                  # chạy song song các lần fit
        error_score="raise",
    )
    # Sau khi tìm xong, search tự huấn luyện lại tổ hợp tốt nhất trên
    # toàn bộ tập train -> search.best_estimator_.
    search.fit(X_train, y_train)

    return search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument(
        "--refit",
        choices=list(SCORING),
        default="pr_auc",
        help="Metric để chọn tham số tốt nhất (mặc định pr_auc).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="Số tổ hợp thử cho mỗi mô hình (mặc định theo DEFAULT_N_ITER).",
    )
    parser.add_argument("--feature-engineering", action="store_true")
    parser.add_argument(
        "--imbalance", choices=IMBALANCE_METHODS, default="class_weight"
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Lưu pipeline tốt nhất (đã fit trên toàn bộ train) để evaluate.",
    )
    parser.add_argument("--output-dir", type=Path, default=TUNING_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR / "tuned")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = args.models or list(MODEL_NAMES)

    X_train, _, y_train, _ = load_train_test()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X_train, y_train))

    name = "tuned" + ("_fe" if args.feature_engineering else "")
    if args.imbalance != "none":
        name += f"_{args.imbalance}"
    run_dir = args.output_dir / datetime.now().strftime(f"{name}_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    all_folds, all_summaries, best_params = [], [], {}

    for model_name in model_names:
        n_iter = args.n_iter or DEFAULT_N_ITER[model_name]
        print(f"\nĐang tune: {model_name} ({n_iter} tổ hợp x {N_SPLITS} fold)", flush=True)

        search = tune_model(
            model_name, X_train, y_train, cv_splits,
            refit=args.refit, n_iter=n_iter,
            feature_engineering=args.feature_engineering,
            imbalance=args.imbalance,
        )

        fold_results, summary = fold_results_from_search(model_name, search)
        all_folds.append(fold_results)
        all_summaries.append(summary)
        best_params[model_name] = search.best_params_

        # Toàn bộ các tổ hợp đã thử, xếp theo metric refit.
        pd.DataFrame(search.cv_results_).sort_values(
            f"rank_test_{args.refit}"
        ).to_csv(run_dir / f"{model_name}_search.csv", index=False)

        print("Tham số tốt nhất:", search.best_params_)
        print(
            f"Validation {args.refit}: {summary[f'val_{args.refit}_mean']:.4f}"
            f" | F1: {summary['val_f1_mean']:.4f} ± {summary['val_f1_std']:.4f}"
            f" | train F1: {summary['train_f1_mean']:.4f}"
        )

        if args.fit:
            args.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(search.best_estimator_, args.model_dir / f"{model_name}.joblib")
            with (args.model_dir / f"{model_name}.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": model_name,
                        "trained_at": datetime.now().isoformat(timespec="seconds"),
                        "tuning_run": run_dir.name,
                        "refit": args.refit,
                        "feature_engineering": args.feature_engineering,
                        "imbalance": args.imbalance,
                        "best_params": search.best_params_,
                        "sklearn_version": sklearn.__version__,
                    },
                    f, ensure_ascii=False, indent=2, default=str,
                )

    pd.concat(all_folds, ignore_index=True).to_csv(run_dir / "cv_results.csv", index=False)
    comparison = pd.DataFrame(all_summaries)
    comparison.to_csv(run_dir / "comparison.csv", index=False)

    with (run_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2, default=str)

    # random_state và n_splits giúp src.compare kiểm tra hai run cùng fold.
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": name,
                "random_state": RANDOM_STATE,
                "n_splits": N_SPLITS,
                "refit": args.refit,
                "feature_engineering": args.feature_engineering,
                "imbalance": args.imbalance,
                "n_iter": {m: args.n_iter or DEFAULT_N_ITER[m] for m in model_names},
                "param_spaces": {m: str(PARAM_SPACES[m]) for m in model_names},
            },
            f, ensure_ascii=False, indent=2,
        )

    columns = ["model", "val_precision_mean", "val_recall_mean", "val_f1_mean",
               "val_roc_auc_mean", "val_pr_auc_mean", "train_f1_mean"]
    print("\nKết quả tuning:")
    print(comparison[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nĐã lưu:", run_dir)


if __name__ == "__main__":
    main()