# So sánh hai lần chạy cross-validation (ví dụ baseline và feature engineering).
#
#   python -m src.compare \
#       outputs/experiments/baseline_<...> \
#       outputs/experiments/feature_engineering_<...>
#
# Hai run phải dùng cùng random_state và số fold, nên fold i của hai run là
# cùng một tập dữ liệu. Vì vậy có thể so sánh cặp theo từng fold (paired):
# chênh lệch được tính trên từng fold, rồi lấy trung bình và đếm số fold
# mà run sau tốt hơn.
import argparse
import json
from pathlib import Path

import pandas as pd

from src.train import PROJECT_ROOT, SCORING


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    with (run_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    return pd.read_csv(run_dir / "cv_results.csv"), config


def compare_runs(base_dir: Path, new_dir: Path) -> pd.DataFrame:
    base_folds, base_config = load_run(base_dir)
    new_folds, new_config = load_run(new_dir)

    for key in ("random_state", "n_splits"):
        if base_config[key] != new_config[key]:
            raise ValueError(f"Hai run khác {key}, không so sánh cặp được.")

    merged = base_folds.merge(
        new_folds,
        on=["model", "fold"],
        suffixes=("_base", "_new"),
    )

    # Chỉ so các metric mà cả hai run đều có (run cũ chưa có pr_auc).
    metrics = [
        metric
        for metric in SCORING
        if f"val_{metric}" in base_folds and f"val_{metric}" in new_folds
    ]

    rows = []

    for model_name, group in merged.groupby("model", sort=False):
        row = {"model": model_name}

        for metric in metrics:
            diff = group[f"val_{metric}_new"] - group[f"val_{metric}_base"]

            row[f"{metric}_base"] = group[f"val_{metric}_base"].mean()
            row[f"{metric}_new"] = group[f"val_{metric}_new"].mean()
            row[f"{metric}_diff"] = diff.mean()
            row[f"{metric}_wins"] = f"{int((diff > 0).sum())}/{len(diff)}"

        rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_run", type=Path, help="Run gốc (ví dụ baseline).")
    parser.add_argument("new_run", type=Path, help="Run cần so sánh.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Lưu bảng so sánh ra .csv (mặc định: <new_run>/compare_vs_base.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = compare_runs(args.base_run, args.new_run)

    output = args.output or args.new_run / "compare_vs_base.csv"
    result.to_csv(output, index=False)

    for metric in ("f1", "recall", "precision", "roc_auc", "pr_auc"):
        if f"{metric}_diff" not in result:
            continue
        columns = ["model"] + [
            f"{metric}_{suffix}" for suffix in ("base", "new", "diff", "wins")
        ]

        print(f"\n{metric.upper()}:")
        print(
            result[columns].to_string(
                index=False,
                float_format=lambda value: f"{value:+.4f}"
                if abs(value) < 0.1
                else f"{value:.4f}",
            )
        )

    try:
        shown = output.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = output

    print("\nĐã lưu:", shown)


if __name__ == "__main__":
    main()