# Machine Failure Prediction

Ứng dụng hỗ trợ đánh giá nguy cơ hỏng hóc máy móc công nghiệp bằng Machine Learning, sử dụng dữ liệu vận hành làm đầu vào và cung cấp dự đoán cho từng thiết bị hoặc nhiều thiết bị cùng lúc.

## 1. Giới thiệu

Dự án sử dụng AI4I 2020 Predictive Maintenance Dataset để xây dựng và so sánh các mô hình phân loại tình trạng hỏng hóc. Mô hình Random Forest được tích hợp vào ứng dụng Streamlit. Phiên bản hiện tại sử dụng ngưỡng phân loại được lựa chọn bằng xác suất Out-of-Fold trên tập huấn luyện.

Ứng dụng phục vụ mục đích học tập và nghiên cứu. Kết quả dự đoán không thay thế việc kiểm tra kỹ thuật hoặc quy trình bảo trì thực tế.

## 2. Công nghệ

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib
- SHAP

**Lưu ý tương thích:** các mô hình `.pkl` trong repository được tuần tự hóa bằng scikit-learn 1.9.1. Cần cài đúng phiên bản này để tránh cảnh báo hoặc lỗi không tương thích khi tải mô hình.

## 3. Các thuật toán đã thử nghiệm

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)

Các chỉ số đánh giá gồm Accuracy, Precision, Recall, F1-score, ROC-AUC và Confusion Matrix. Dự án cũng thực hiện thí nghiệm xử lý mất cân bằng dữ liệu và điều chỉnh ngưỡng phân loại.

## 4. Chức năng ứng dụng

### Dashboard
Hiển thị quy mô dữ liệu, số lượng mẫu bình thường, số mẫu hỏng và tỷ lệ hỏng trong dataset.

### Dự đoán một máy
Người dùng nhập loại máy và năm thông số vận hành. Ứng dụng trả về xác suất ước lượng, nhãn phân loại theo threshold đang cấu hình và biểu đồ giải thích SHAP cho mẫu đầu vào.

### Dự đoán hàng loạt
Người dùng tải CSV, kiểm tra cột bắt buộc, kiểu dữ liệu, giá trị thiếu và miền giá trị đầu vào; sau đó ứng dụng dự đoán đồng loạt và cho phép tải kết quả CSV.

### So sánh mô hình
Hiển thị bảng và biểu đồ so sánh năm thuật toán, Confusion Matrix, Feature Importance của Random Forest và kết quả đánh giá trước/sau điều chỉnh threshold.

## 5. Mô hình được sử dụng trong ứng dụng

Ứng dụng tải mô hình `results/threshold_tuned_random_forest.pkl`. Ngưỡng phân loại được đọc từ `results/threshold_experiment_metadata.json`, hiện là **0,45**. Cả dự đoán đơn lẻ và dự đoán hàng loạt đều sử dụng chung hàm suy luận trong `src/inference.py` để tránh khác biệt logic.

Threshold được lựa chọn bằng F1-score trên xác suất Out-of-Fold của tập huấn luyện. Tập kiểm thử được giữ riêng để đánh giá cuối cùng. Trong kết quả hiện có, threshold 0,45 tăng Recall từ 0,6765 lên 0,7500 và F1-score từ 0,6301 lên 0,6456; Precision giảm từ 0,5897 xuống 0,5667. Đây là sự đánh đổi giữa phát hiện thêm trường hợp hỏng và tăng cảnh báo nhầm.

## 6. Cấu trúc dự án

```text
MachineFailurePrediction/
├── data/
│   └── ai4i2020.csv
├── models/
│   ├── best_model.pkl
│   └── metadata.json
├── results/
│   ├── threshold_tuned_random_forest.pkl
│   ├── threshold_experiment_metadata.json
│   ├── threshold_test_comparison.csv
│   ├── threshold_oof_comparison.csv
│   ├── threshold_confusion_matrices.csv
│   ├── model_comparison.csv
│   └── ...
├── src/
│   ├── inference.py
│   ├── train.py
│   ├── imbalance_experiment.py
│   └── threshold_experiment.py
├── tests/
│   └── test_inference.py
├── app.py
├── requirements.txt
└── README.md
```

## 7. Cài đặt trên Windows

Yêu cầu Python 3.10 trở lên và scikit-learn 1.9.1.

```powershell
git clone <GITHUB_REPOSITORY_URL>
cd MachineFailurePrediction
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Chạy ứng dụng

```powershell
streamlit run app.py
```

Mở địa chỉ `http://localhost:8501` trên trình duyệt.

## 9. Chạy kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Các kiểm thử bao gồm dữ liệu hợp lệ, thiếu cột, loại máy không hợp lệ, giá trị ngoài miền và việc áp dụng threshold. Một kiểm thử đối chiếu các chỉ số trên tập kiểm thử với kết quả thực nghiệm đã lưu.

## 10. Dataset và giới hạn

**Dataset:** AI4I 2020 Predictive Maintenance Dataset.

**Nguồn:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Dataset có tỷ lệ mẫu hỏng thấp hơn đáng kể so với mẫu bình thường. Vì vậy, Accuracy không nên được dùng làm tiêu chí đánh giá duy nhất. Recall, Precision, F1-score và PR-AUC cần được xem xét cùng nhau.

Xác suất do mô hình đưa ra là ước lượng của mô hình trên dữ liệu đầu vào, không phải xác suất đã được hiệu chuẩn hoặc xác nhận chắc chắn tình trạng thực tế của thiết bị. SHAP mô tả đóng góp của đặc trưng vào đầu ra mô hình, không chứng minh quan hệ nhân quả.
