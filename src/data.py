from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

def load_and_clean_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Dataset missing columns: {sorted(missing_columns)}")

    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        df[column] = df[column].str.strip().replace("", float("nan"))

    if df["customerID"].isna().any():
        raise ValueError("customerID has missing values.")

    if df["customerID"].duplicated().any():
        raise ValueError("customerID has duplicated values.")

    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")

    invalid_values = df["TotalCharges"].notna() & total_charges.isna()

    if invalid_values.any():
        raise ValueError(
            "TotalCharges chứa giá trị không phải số: "
            f"{df.loc[invalid_values, 'TotalCharges'].unique().tolist()}"
        )

    df["TotalCharges"] = total_charges

    if df["Churn"].isna().any():
        raise ValueError("Churn chứa giá trị thiếu.")

    unexpected_labels = set(df["Churn"].unique()) - {"Yes", "No"}

    if unexpected_labels:
        raise ValueError(f"Churn chứa nhãn không hợp lệ: {unexpected_labels}")

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df

def load_split_ids(path: str | Path) -> list[str]:
    """Đọc và kiểm tra danh sách customerID của một tập dữ liệu."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp split: {path}")

    split_df = pd.read_csv(path, dtype={"customerID": "string"})

    if "customerID" not in split_df.columns:
        raise ValueError(f"Tệp {path.name} thiếu cột customerID.")

    ids = split_df["customerID"].str.strip()

    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"Tệp {path.name} chứa ID thiếu hoặc rỗng.")

    if ids.duplicated().any():
        raise ValueError(f"Tệp {path.name} chứa ID trùng.")

    return ids.tolist()
#Prepare train/test
def load_train_test(
    data_path: str | Path = DEFAULT_DATA_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Khôi phục train/test từ ID đã lưu và tách đặc trưng, nhãn."""

    df = load_and_clean_data(data_path)
    split_dir = Path(split_dir)

    train_ids = load_split_ids(split_dir / "train_ids.csv")
    test_ids = load_split_ids(split_dir / "test_ids.csv")

    train_id_set = set(train_ids)
    test_id_set = set(test_ids)
    dataset_ids = set(df["customerID"])


    if not train_ids or not test_ids:
        raise ValueError("Tập train hoặc test đang rỗng.")

    if train_id_set & test_id_set:
        raise ValueError("Train và test có customerID giao nhau.")

    split_ids = train_id_set | test_id_set

    unknown_ids = split_ids - dataset_ids

    if unknown_ids:
        raise ValueError(
            f"Có {len(unknown_ids)} ID trong tệp split không tồn tại trong dataset."
        )
    unassigned_ids = dataset_ids - split_ids

    if unassigned_ids:
        raise ValueError(
            f"Có {len(unassigned_ids)} khách hàng chưa được gán vào train/test."
        )

    indexed_df = df.set_index("customerID")

    train_df = indexed_df.loc[train_ids].copy()
    test_df = indexed_df.loc[test_ids].copy()

    X_train = train_df.drop(columns="Churn").reset_index(drop=True)
    X_test = test_df.drop(columns="Churn").reset_index(drop=True)

    y_train = train_df["Churn"].reset_index(drop=True)
    y_test = test_df["Churn"].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_test()

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\nTỷ lệ nhãn trong train:")
    print(y_train.value_counts(normalize=True).sort_index())

    print("\nMissing values trong X_train:")
    print(X_train.isna().sum().loc[lambda counts: counts > 0])

    assert X_train.columns.equals(X_test.columns)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert "customerID" not in X_train.columns
    assert "Churn" not in X_train.columns
