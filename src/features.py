# Feature Engineering cho bài toán Customer Churn
#
# ChurnFeatureEngineer được đặt làm bước đầu tiên trong Pipeline, trước
# preprocessor. Nhờ vậy khi chạy cross-validation, những gì cần "học" từ dữ
# liệu (ngưỡng chia nhóm MonthlyCharges) chỉ được fit trên fold train,
# không bị rò rỉ thông tin từ fold validation.
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

# Các dịch vụ cộng thêm, giá trị "Yes" nghĩa là khách có đăng ký.
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]

# Tên các cột mới, chia theo cách preprocessor sẽ xử lý.
ENGINEERED_NUMERIC_FEATURES = [
    "num_services",
    "avg_monthly_charge",
    "charge_diff",
]

ENGINEERED_BINARY_FEATURES = [
    "is_new_month_to_month",
    "fiber_no_tech_support",
    "is_auto_payment",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "tenure_group",
    "monthly_charge_group",
    "contract_tenure",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, n_charge_bins: int = 4):
        self.n_charge_bins = n_charge_bins

    def fit(self, X: pd.DataFrame, y=None):

        quantiles = np.linspace(0, 1, self.n_charge_bins + 1)
        edges = np.quantile(X["MonthlyCharges"], quantiles)

        edges[0], edges[-1] = -np.inf, np.inf
        self.charge_edges_ = np.unique(edges)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        tenure = X["tenure"]
        monthly = X["MonthlyCharges"]

        # 1. Nhóm thời gian sử dụng dịch vụ.
        X["tenure_group"] = pd.cut(
            tenure,
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
        ).astype(str)

        # 2. Số lượng dịch vụ khách hàng đăng ký (tính cả Internet).
        X["num_services"] = X[SERVICE_COLUMNS].eq("Yes").sum(axis=1) + X[
            "InternetService"
        ].ne("No").astype(int)

        # 3. Mức phí trung bình thực tế mỗi tháng. Khách tenure = 0 chưa có
        # TotalCharges nên lấy luôn MonthlyCharges.
        X["avg_monthly_charge"] = np.where(
            tenure > 0,
            X["TotalCharges"] / tenure.where(tenure > 0),
            monthly,
        )
        X["avg_monthly_charge"] = X["avg_monthly_charge"].fillna(monthly)

        # Chênh lệch dương: phí hiện tại cao hơn mức trung bình đã trả,
        # tức khách vừa bị tăng giá hoặc vừa nâng gói.
        X["charge_diff"] = monthly - X["avg_monthly_charge"]

        # 4. Nhóm phí hàng tháng (ngưỡng học từ tập train).
        X["monthly_charge_group"] = pd.cut(
            monthly,
            bins=self.charge_edges_,
            labels=[f"Q{i + 1}" for i in range(len(self.charge_edges_) - 1)],
        ).astype(str)

        # 5. Đặc trưng kết hợp hợp đồng, thời gian sử dụng và dịch vụ.
        is_month_to_month = X["Contract"].eq("Month-to-month")

        X["is_new_month_to_month"] = (is_month_to_month & (tenure <= 12)).astype(
            int
        )
        X["fiber_no_tech_support"] = (
            X["InternetService"].eq("Fiber optic") & X["TechSupport"].eq("No")
        ).astype(int)
        X["is_auto_payment"] = (
            X["PaymentMethod"].str.contains("automatic", na=False).astype(int)
        )
        X["contract_tenure"] = X["Contract"] + "_" + X["tenure_group"]

        return X


if __name__ == "__main__":
    from src.data import load_train_test

    X_train, X_test, _, _ = load_train_test()

    engineer = ChurnFeatureEngineer().fit(X_train)
    train_fe = engineer.transform(X_train)
    test_fe = engineer.transform(X_test)

    new_columns = (
        ENGINEERED_NUMERIC_FEATURES
        + ENGINEERED_BINARY_FEATURES
        + ENGINEERED_CATEGORICAL_FEATURES
    )

    print("Ngưỡng nhóm MonthlyCharges:", engineer.charge_edges_.round(2))
    print("\nMột số dòng đặc trưng mới:")
    print(train_fe[new_columns].head())

    print("\nGiá trị thiếu ở cột mới (train):")
    print(train_fe[new_columns].isna().sum().loc[lambda s: s > 0])

    assert list(train_fe.columns[: X_train.shape[1]]) == list(X_train.columns)
    assert not train_fe[new_columns].isna().any().any()
    assert not test_fe[new_columns].isna().any().any()
