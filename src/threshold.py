# Chọn ngưỡng quyết định cho các pipeline đã tune, chỉ dùng tập train.
#
#   python -m src.threshold
#   python -m src.threshold --models xgboost random_forest --beta 2
#
# Cách làm: với mỗi mô hình, chạy lại 5-fold CV (cùng fold với train.py) để
# lấy xác suất "out-of-fold" (OOF): mỗi khách hàng được dự đoán bởi mô hình
# KHÔNG nhìn thấy khách hàng đó lúc fit. Ngưỡng được chọn trên các xác suất
# này, nên tập test vẫn chưa bị dùng tới.
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import load_train_test
from src.models import MODEL_NAMES
from src.train import MODEL_DIR, N_SPLITS, PROJECT_ROOT, RANDOM_STATE

THRESHOLD_DIR = PROJECT_ROOT / "outputs" / "thresholds"

# Lưới ngưỡng cần thử: 0.05, 0.06, ..., 0.95.
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def oof_probabilities(pipeline, X, y, cv_splits) -> np.ndarray:
    """Xác suất churn out-of-fold. clone() lấy cấu hình, bỏ phần đã fit."""

    return cross_val_predict(
        clone(pipeline),
        X,
        y,
        cv=cv_splits,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


def scores_by_threshold(y, proba, beta: float) -> pd.DataFrame:
    rows = []

    for t in THRESHOLDS:
        pred = (proba > t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "fbeta": fbeta_score(y, pred, beta=beta, zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def per_fold_scores(y, proba, cv_splits, threshold: float) -> list[float]:
    """F1 trên từng fold validation, để xem ngưỡng có ổn định không."""

    return [
        f1_score(y.iloc[val], (proba[val] > threshold).astype(int))
        for _, val in cv_splits
    ]


def plot_curves(curves: dict[str, pd.DataFrame], chosen: dict, path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(curves), figsize=(4.5 * len(curves), 4), squeeze=False, sharey=True
    )

    for ax, (model_name, table) in zip(axes[0], curves.items()):
        ax.plot(table["threshold"], table["precision"], label="Precision")
        ax.plot(table["threshold"], table["recall"], label="Recall")
        ax.plot(table["threshold"], table["f1"], label="F1", linewidth=2)
        ax.axvline(0.5, color="grey", linestyle=":", label="0.5")
        ax.axvline(chosen[model_name], color="red", linestyle="--", label="Ngưỡng chọn")
        ax.set_title(model_name)
        ax.set_xlabel("Ngưỡng")

    axes[0][0].set_ylabel("Điểm (OOF trên train)")
    axes[0][0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR / "tuned",
        help="Thư mục chứa pipeline đã tune (mặc định outputs/models/tuned).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Chọn ngưỡng tối đa F-beta. 1 = F1; 2 = coi recall quan trọng gấp đôi.",
    )
    parser.add_argument("--output-dir", type=Path, default=THRESHOLD_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_names = args.models or [
        m for m in MODEL_NAMES if (args.model_dir / f"{m}.joblib").is_file()
    ]
    if not model_names:
        raise FileNotFoundError(f"Không có pipeline nào trong {args.model_dir}.")

    X_train, _, y_train, _ = load_train_test()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X_train, y_train))

    run_dir = args.output_dir / datetime.now().strftime("thresholds_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    oof = pd.DataFrame({"churn": y_train})
    curves, chosen, summary = {}, {}, []

    for model_name in model_names:
        print(f"Đang lấy xác suất OOF: {model_name}", flush=True)

        pipeline = joblib.load(args.model_dir / f"{model_name}.joblib")
        proba = oof_probabilities(pipeline, X_train, y_train, cv_splits)
        oof[model_name] = proba

        table = scores_by_threshold(y_train, proba, args.beta)
        table.to_csv(run_dir / f"{model_name}_threshold_scores.csv", index=False)
        curves[model_name] = table

        best = table.loc[table["fbeta"].idxmax()]
        at_05 = table.loc[np.isclose(table["threshold"], 0.5)].iloc[0]
        chosen[model_name] = float(best["threshold"])

        fold_05 = per_fold_scores(y_train, proba, cv_splits, 0.5)
        fold_best = per_fold_scores(y_train, proba, cv_splits, best["threshold"])

        summary.append(
            {
                "model": model_name,
                "threshold": best["threshold"],
                "precision_at_0.5": at_05["precision"],
                "recall_at_0.5": at_05["recall"],
                "f1_at_0.5": at_05["f1"],
                "precision_at_best": best["precision"],
                "recall_at_best": best["recall"],
                "f1_at_best": best["f1"],
                "f1_gain": best["f1"] - at_05["f1"],
                "fold_wins": f"{sum(b > a for a, b in zip(fold_05, fold_best))}/{N_SPLITS}",
            }
        )

    summary = pd.DataFrame(summary)
    summary.to_csv(run_dir / "summary.csv", index=False)
    oof.to_csv(run_dir / "oof_probabilities.csv", index=False)
    plot_curves(curves, chosen, run_dir / "threshold_curves.png")

    result = {
        "beta": args.beta,
        "chosen_on": "out-of-fold probabilities, train set, 5-fold CV",
        "source_run": run_dir.name,
        "thresholds": chosen,
    }
    for path in (run_dir / "thresholds.json", args.model_dir / "thresholds.json"):
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nKết quả chọn ngưỡng (trên train, OOF):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nĐã lưu:", run_dir)
    print("Đã ghi ngưỡng vào:", args.model_dir / "thresholds.json")


if __name__ == "__main__":
    main()