# AI-KHDL-assignment — Dự đoán khách hàng rời bỏ (Customer Churn)

Dự đoán khách hàng của một công ty viễn thông có rời bỏ dịch vụ hay không, dựa trên bộ dữ liệu
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7.043 khách hàng, 19 đặc trưng, khoảng 26,5% khách hàng rời bỏ).

## Kết quả chính

Mô hình cuối cùng: **XGBoost đã tune, cân bằng lớp bằng `class_weight`, ngưỡng quyết định 0,57**.

| Tập test (1.409 khách, 374 người rời) | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0,550 | 0,749 | **0,634** | **0,846** | **0,662** |
| Logistic Regression | 0,539 | 0,706 | 0,611 | 0,842 | 0,633 |
| Random Forest | 0,566 | 0,695 | 0,624 | 0,843 | 0,651 |
| Decision Tree | 0,559 | 0,719 | 0,629 | 0,835 | 0,624 |

- Mô hình phát hiện được **280 / 374** khách hàng thực sự rời bỏ.
- Kết quả test khớp với cross-validation (PR-AUC 0,671 → 0,662), không có dấu hiệu overfit.
- Cải thiện chủ yếu đến từ **xử lý mất cân bằng lớp** và **tuning chống overfit**;
  feature engineering và việc dời ngưỡng gần như không thay đổi F1.
- Đặc trưng quan trọng nhất: `tenure`, `Contract`, `InternetService`, `TechSupport`, `OnlineSecurity`.

Phân tích đầy đủ: [`notebooks/04_final_analysis.ipynb`](notebooks/04_final_analysis.ipynb).

## Cấu trúc thư mục

```
data/
  raw/                  Dữ liệu gốc
  splits/               customerID của tập train (80%) và test (20%), chia stratified
  examples/             Ví dụ khách hàng mới để thử predict
notebooks/
  01_eda.ipynb                     Khám phá dữ liệu
  02_analysis_LG-XGBoost.ipynb     Phân tích baseline Logistic Regression, XGBoost
  03_tree_forest_analysis.ipynb    Phân tích baseline Decision Tree, Random Forest
  04_final_analysis.ipynb          Tổng hợp thực nghiệm, kết quả test, giải thích mô hình
src/
  data.py           Nạp, làm sạch dữ liệu và khôi phục train/test từ splits
  preprocessing.py  Impute, chuẩn hóa, one-hot (ColumnTransformer)
  features.py       Feature engineering (ChurnFeatureEngineer)
  models.py         Khởi tạo 4 mô hình
  train.py          Cross-validation (cv) và huấn luyện mô hình cuối (fit)
  compare.py        So sánh cặp hai lần chạy CV theo từng fold
  tune.py           Tuning siêu tham số bằng RandomizedSearchCV
  threshold.py      Chọn ngưỡng quyết định trên xác suất out-of-fold
  evaluate.py       Đánh giá trên tập test
  predict.py        Dự đoán cho khách hàng mới
outputs/
  experiments/      Kết quả cross-validation của từng thực nghiệm
  tuning/           Kết quả tuning
  thresholds/       Kết quả chọn ngưỡng
  evaluation/       Kết quả trên tập test (tuned_f1: kết quả chính, tuned_f2: ưu tiên recall)
  models/tuned/     Pipeline cuối cùng (.joblib) và thresholds.json
```

## Cài đặt

Yêu cầu Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate          # Windows; Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

File `.joblib` chỉ dùng được với đúng phiên bản scikit-learn đã lưu nó (các mô hình trong
`outputs/models/tuned/` được lưu bằng scikit-learn 1.9.1). Nếu khác phiên bản, chạy lại
`python -m src.tune --fit` để huấn luyện lại.

## Cách chạy

Tất cả lệnh chạy từ thư mục gốc của dự án.

**1. Cross-validation các thực nghiệm** (5-fold trên tập train):

```bash
python -m src.train cv                                          # baseline
python -m src.train cv --feature-engineering                    # thêm đặc trưng mới
python -m src.train cv --imbalance class_weight                 # hoặc --imbalance smote
python -m src.compare outputs/experiments/<run gốc> outputs/experiments/<run mới>
```

**2. Tuning siêu tham số và lưu pipeline tốt nhất:**

```bash
python -m src.tune --fit
```

**3. Chọn ngưỡng quyết định** (trên xác suất out-of-fold của tập train):

```bash
python -m src.threshold              # tối đa F1
python -m src.threshold --beta 2     # ưu tiên recall
```

**4. Đánh giá trên tập test:**

```bash
python -m src.evaluate --model-dir outputs/models/tuned
```

**5. Dự đoán cho khách hàng mới:**

```bash
python -m src.predict --model xgboost --model-dir outputs/models/tuned --threshold 0.57 --input data/examples/new_customers.csv
```

## Quy trình đánh giá

- Tập test được tách một lần (theo `customerID`, lưu trong `data/splits/`) và chỉ dùng ở bước
  đánh giá cuối cùng.
- Mọi lựa chọn — feature engineering, cách xử lý mất cân bằng, siêu tham số, ngưỡng, mô hình
  chính — đều được quyết định bằng 5-fold cross-validation trên tập train với cùng các fold
  (`random_state=42`), để so sánh cặp theo từng fold.
- Tiền xử lý, feature engineering và SMOTE nằm trong Pipeline, nên chỉ được fit trên fold train,
  không rò rỉ thông tin từ fold validation.
