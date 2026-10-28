from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_FEATURES = [
    "SeniorCitizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

def build_preprocessor(
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Tạo bộ tiền xử lý cho các đặc trưng gốc của dataset."""
    
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from src.data import load_train_test

    X_train, _, _, _ = load_train_test()

    # Chỉ lấy một phần nhỏ của train để kiểm tra cơ chế transform.
    # Đây không phải bước chuẩn bị dữ liệu cho cross-validation.
    sample = X_train.head(100).copy()

    # Chủ động đưa một missing vào để kiểm tra imputer.
    sample.loc[sample.index[0], "TotalCharges"] = np.nan

    preprocessor = build_preprocessor(scale_numeric=True)

    transformed = preprocessor.fit_transform(sample)
    feature_names = preprocessor.get_feature_names_out()

    print("Kích thước đầu vào:", sample.shape)
    print("Kích thước sau tiền xử lý:", transformed.shape)

    print(
        "Số giá trị thiếu sau tiền xử lý:",
        np.isnan(transformed).sum(),
    )

    print("\nMột số dòng sau tiền xử lý:")
    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=sample.index,
    )
    print(transformed_df.head())

    assert transformed.shape[0] == len(sample)
    assert transformed.shape[1] == len(feature_names)
    assert np.isfinite(transformed).all()

    # Kiểm tra encoder xử lý được giá trị phân loại chưa từng gặp.
    unseen_sample = sample.iloc[[0]].copy()
    unseen_sample["PaymentMethod"] = "Unseen payment method"

    unseen_transformed = preprocessor.transform(unseen_sample)

    assert unseen_transformed.shape[1] == transformed.shape[1]
    assert np.isfinite(unseen_transformed).all()

    print("\nKiểm tra tiền xử lý thành công.")