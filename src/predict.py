# Dùng pipeline đã lưu để dự đoán khả năng rời bỏ của khách hàng mới.
#
#   python -m src.predict --model random_forest \
#       --input data/examples/new_customers.csv
#
# File đầu vào (.csv hoặc .json) cần đủ 19 đặc trưng gốc của dataset.
# Cột customerID là tùy chọn (nếu có sẽ được giữ lại ở kết quả);
# các cột thừa khác (ví dụ Churn) bị bỏ qua.
import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline

from src.evaluate import load_pipeline
from src.models import MODEL_NAMES
from src.preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from src.train import MODEL_DIR

INPUT_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

DEFAULT_THRESHOLD = 0.5


def read_customers(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp đầu vào: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as file:
            data = json.load(file)

        # Chấp nhận một object (1 khách hàng) hoặc danh sách object.
        return pd.DataFrame([data] if isinstance(data, dict) else data)

    raise ValueError("Chỉ hỗ trợ tệp .csv hoặc .json.")


def prepare_features(customers: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu khách hàng mới giống cách load_and_clean_data làm."""

    missing_columns = [
        column for column in INPUT_FEATURES if column not in customers.columns
    ]

    if missing_columns:
        raise ValueError(f"Thiếu các cột đặc trưng: {missing_columns}")

    features = customers[INPUT_FEATURES].copy()

    text_columns = features.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        features[column] = features[column].str.strip().replace("", float("nan"))

    # TotalCharges có thể để trống với khách mới (tenure = 0);
    # imputer trong pipeline sẽ điền bằng median của tập train.
    total_charges = pd.to_numeric(features["TotalCharges"], errors="coerce")
    invalid = features["TotalCharges"].notna() & total_charges.isna()

    if invalid.any():
        raise ValueError(
            "TotalCharges chứa giá trị không phải số: "
            f"{features.loc[invalid, 'TotalCharges'].unique().tolist()}"
        )

    features["TotalCharges"] = total_charges

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise")

    if not features["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen chỉ nhận giá trị 0 hoặc 1.")

    return features


def warn_unseen_categories(pipeline: Pipeline, features: pd.DataFrame) -> None:
    """Báo nếu có giá trị phân loại chưa từng thấy lúc train.

    OneHotEncoder(handle_unknown="ignore") vẫn chạy được nhưng sẽ mã hóa
    giá trị đó thành toàn số 0, nên cần cho người dùng biết.
    """

    encoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )

    for column, known in zip(CATEGORICAL_FEATURES, encoder.categories_):
        unseen = set(features[column].dropna().unique()) - set(known)

        if unseen:
            warnings.warn(
                f"Cột {column} có giá trị chưa gặp khi train: "
                f"{sorted(unseen)}",
                stacklevel=2,
            )


def predict_customers(
    customers: pd.DataFrame,
    model_name: str,
    threshold: float = DEFAULT_THRESHOLD,
    model_dir: Path = MODEL_DIR,
) -> pd.DataFrame:
    """Dự đoán churn cho khách hàng mới bằng pipeline đã lưu.

    Dự đoán "rời" khi xác suất > threshold. Với ngưỡng mặc định 0,5 kết quả
    trùng với pipeline.predict().
    """

    pipeline = load_pipeline(model_name, model_dir)
    features = prepare_features(customers)

    warn_unseen_categories(pipeline, features)

    probability = pipeline.predict_proba(features)[:, 1]
    prediction = (probability > threshold).astype(int)

    result = pd.DataFrame(index=customers.index)

    if "customerID" in customers.columns:
        result["customerID"] = customers["customerID"]

    result["churn_probability"] = probability
    result["prediction"] = prediction
    result["label"] = pd.Series(prediction, index=result.index).map(
        {0: "No", 1: "Yes"}
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Lưu kết quả ra file .csv (tùy chọn).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Dự đoán rời khi xác suất > ngưỡng (mặc định 0.5).",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold phải nằm trong [0, 1].")

    customers = read_customers(args.input)

    result = predict_customers(
        customers,
        model_name=args.model,
        threshold=args.threshold,
        model_dir=args.model_dir,
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print("\nĐã lưu:", args.output)


if __name__ == "__main__":
    main()
