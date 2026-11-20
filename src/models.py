from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
)

def build_model(
    model_name: str,
    random_state: int = 42,
    balance_ratio: float | None = None,
) -> BaseEstimator:

    class_weight = "balanced" if balance_ratio else None

    if model_name == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
        )

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=random_state,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        # XGBoost không có class_weight, dùng scale_pos_weight thay thế.
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            scale_pos_weight=balance_ratio or 1.0,
            random_state=random_state,
            n_jobs=1,
        )

    raise ValueError(
        f"Models not accepted: {model_name!r}. "
        f"Names availabled: {MODEL_NAMES}"
    )

def requires_scaling(model_name: str) -> bool:
    """Cho biết có chuẩn hóa biến số cho mô hình hay không."""

    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Models not accepted: {model_name!r}. "
            f"Names availabled: {MODEL_NAMES}"
        )
    return model_name == "logistic_regression"

if __name__ == "__main__":
    for model_name in MODEL_NAMES:
        model = build_model(model_name)

        print(f"\nTên mô hình: {model_name}")
        print(f"Cần scaling: {requires_scaling(model_name)}")
        print(f"Cấu hình: {model}")

    print("\nKhởi tạo các mô hình thành công.")