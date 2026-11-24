import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

from src.inference import (
    FEATURES, predict_with_threshold as run_prediction,
    validate_input_data,
)


from src.database import (
    add_machine,
    get_all_machines,
    get_machine_by_id,
    update_machine,
    delete_machine,
    save_prediction,
    get_prediction_history,
    add_maintenance_record,
    get_maintenance_records,
    update_maintenance_record,
)



# =====================================================
# 1. CẤU HÌNH
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_PATH = os.path.join(BASE_DIR, "results", "threshold_tuned_random_forest.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "results", "threshold_experiment_metadata.json")
THRESHOLD_RESULTS_PATH = os.path.join(BASE_DIR, "results", "threshold_test_comparison.csv")
RESULT_PATH = os.path.join(BASE_DIR, "results", "model_comparison.csv")
CM_PATH = os.path.join(BASE_DIR, "results", "confusion_matrix.csv")

FEATURE_LABELS = {
    "Type": "Loại máy",
    "Air temperature [K]": "Nhiệt độ không khí (K)",
    "Process temperature [K]": "Nhiệt độ quy trình (K)",
    "Rotational speed [rpm]": "Tốc độ quay (rpm)",
    "Torque [Nm]": "Mô-men xoắn (Nm)",
    "Tool wear [min]": "Độ mài mòn dụng cụ (phút)"
}

MODEL_LABELS = {
    "Logistic Regression": "Hồi quy Logistic",
    "KNN": "K-Nearest Neighbors",
    "Decision Tree": "Cây quyết định",
    "Random Forest": "Rừng ngẫu nhiên",
    "SVM": "Support Vector Machine"
}

st.set_page_config(
    page_title="Dự đoán hỏng hóc máy móc",
    page_icon="⚙️",
    layout="wide"
)


# =====================================================
# INDUSTRIAL DARK UI - GIAO DIỆN CÔNG NGHIỆP
# =====================================================

st.markdown("""
<style>
:root {
    --bg: #151719;
    --surface: #202326;
    --surface-2: #292d30;
    --border: #3b4145;
    --orange: #ff8a00;
    --yellow: #ffc533;
    --text: #f2f3f4;
    --muted: #a6adb3;
}
.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(255,138,0,.07), transparent 28%),
        var(--bg);
    color: var(--text);
}
[data-testid="stSidebar"] {display: none;}
.block-container {
    max-width: 1600px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}
h1, h2, h3 { color: #f5f5f5 !important; letter-spacing: -.025em; }
h1 { border-left: 5px solid var(--orange); padding-left: .7rem; }
h2, h3 { border-bottom: 1px solid rgba(255,138,0,.22); padding-bottom: .45rem; }
p, label, .stCaption { color: var(--muted); }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #24282b, #1c1f21);
    border: 1px solid var(--border);
    border-top: 3px solid var(--orange);
    border-radius: 10px;
    padding: 15px 17px;
    min-height: 112px;
}
[data-testid="stMetricLabel"] { color: #aeb5ba !important; }
[data-testid="stMetricValue"] { color: #fff !important; }
[data-testid="stMetricDelta"] { color: var(--yellow) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 9px;
    overflow: hidden;
}
[data-testid="stForm"], [data-testid="stExpander"] {
    background: #1c1f21;
    border: 1px solid var(--border);
    border-radius: 10px;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stDateInput input, [data-baseweb="select"] > div {
    background: #25292c !important;
    border-color: #454b50 !important;
    color: #f5f5f5 !important;
    border-radius: 6px !important;
}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: linear-gradient(110deg, #ff8a00, #e87500) !important;
    color: #171717 !important;
    border: 1px solid #ffad42 !important;
    border-radius: 7px !important;
    font-weight: 700 !important;
    min-height: 2.5rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    background: #ffb02e !important;
    border-color: #ffd06a !important;
    color: #111 !important;
}
div[role="radiogroup"] { gap: .45rem; }
div[role="radiogroup"] label {
    background: #24282b;
    border: 1px solid #41474b;
    border-radius: 7px;
    padding: .42rem .8rem;
    color: #d9dddf !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background: rgba(255,138,0,.16);
    border-color: var(--orange);
    color: #ffc16b !important;
}
[data-testid="stAlert"] { border-radius: 8px; }
hr { border-color: #3a4044 !important; }
[data-testid="stProgressBar"] > div > div > div > div {
    background: var(--orange) !important;
}
header[data-testid="stHeader"] { background: transparent; }
footer {visibility: hidden;}
@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    div[role="radiogroup"] { flex-wrap: wrap; }
}
</style>
""", unsafe_allow_html=True)

# Đồng bộ màu biểu đồ với giao diện công nghiệp
plt.rcParams.update({
    "figure.facecolor": "#202326",
    "axes.facecolor": "#202326",
    "axes.edgecolor": "#596168",
    "axes.labelcolor": "#d7dcdf",
    "text.color": "#f2f3f4",
    "xtick.color": "#c3c9cd",
    "ytick.color": "#c3c9cd",
    "grid.color": "#3b4145",
})


# =====================================================
# 2. TẢI MÔ HÌNH VÀ DỮ LIỆU
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_threshold_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_threshold_results():
    return pd.read_csv(THRESHOLD_RESULTS_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_comparison():
    return pd.read_csv(RESULT_PATH)


@st.cache_data
def load_confusion_matrix():
    return pd.read_csv(CM_PATH, index_col=0)


try:
    model = load_model()
    df = load_data()
    comparison = load_comparison()
    threshold_metadata = load_threshold_metadata()
    threshold_results = load_threshold_results()
    PREDICTION_THRESHOLD = float(threshold_metadata["selected_threshold"])
except FileNotFoundError as error:
    st.error(f"Không tìm thấy tệp cần thiết: {error}")
    st.stop()
except Exception as error:
    st.error(f"Không thể tải dữ liệu hoặc mô hình: {error}")
    st.stop()


# =====================================================
# 3. THANH ĐIỀU HƯỚNG NGANG
# =====================================================

brand_col, status_col = st.columns([3, 1])
with brand_col:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
          <div style="font-size:30px;">⚙️</div>
          <div>
            <div style="font-size:24px;font-weight:800;color:#fff;letter-spacing:-.5px;">
              MACHINE <span style="color:#ff8a00;">FAILURE</span>
            </div>
            <div style="font-size:12px;color:#9da5aa;letter-spacing:1.4px;">
              PREDICTIVE MAINTENANCE CONTROL CENTER
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with status_col:
    st.markdown(
        f"""
        <div style="text-align:right;padding-top:6px;">
          <span style="color:#ffc533;font-size:12px;font-weight:700;"></span><br>
          <span style="color:#9da5aa;font-size:11px;">Threshold: {PREDICTION_THRESHOLD:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    '<div style="height:3px;background:linear-gradient(90deg,#ff8a00 0%,#ffc533 28%,#343a3e 75%);border-radius:3px;margin:12px 0 16px 0;"></div>',
    unsafe_allow_html=True
)

page = st.radio(
    "Điều hướng chính",
    [
        "Tổng quan",
        "Quản lý thiết bị",
        "Giám sát & Cảnh báo",
        "Quản lý bảo trì",
        "Dự đoán một máy",
        "Dự đoán hàng loạt",
        "So sánh mô hình"
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation"
)

st.markdown(
    '<div style="border-bottom:1px solid #3b4145;margin:0 0 24px 0;"></div>',
    unsafe_allow_html=True
)


# =====================================================
# 4. HÀM HỖ TRỢ
# =====================================================

@st.cache_resource
def load_shap_explainer():
    import shap
    return shap.TreeExplainer(model.named_steps["classifier"])



def show_local_explanation(input_df):
    """Explain one prediction using SHAP values of the fitted RF classifier."""
    try:
        import shap

        # Tiền xử lý dữ liệu đầu vào
        transformed = model.named_steps["preprocessor"].transform(input_df)

        explainer = load_shap_explainer()
        explanation = explainer(transformed)

        # Lấy tên đặc trưng sau tiền xử lý
        feature_names = model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        feature_label_map = {
            "num__Air temperature [K]": "Nhiệt độ không khí",
            "num__Process temperature [K]": "Nhiệt độ quy trình",
            "num__Rotational speed [rpm]": "Tốc độ quay",
            "num__Torque [Nm]": "Mô-men xoắn",
            "num__Tool wear [min]": "Độ mài mòn dụng cụ",
            "cat__Type_H": "Loại máy H",
            "cat__Type_L": "Loại máy L",
            "cat__Type_M": "Loại máy M",
        }

        feature_names = [
            feature_label_map.get(name, name)
            for name in feature_names
        ]

        # Trích xuất SHAP values cho lớp hỏng hóc (class 1)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)

        if values.ndim == 3:
            local_values = values[0, :, 1]

            if base_values.ndim >= 2:
                base_value = float(base_values[0, 1])
            else:
                base_value = float(np.ravel(base_values)[1])

        elif values.ndim == 2:
            local_values = values[0]

            if base_values.ndim >= 1:
                base_value = float(np.ravel(base_values)[0])
            else:
                base_value = float(base_values)

        else:
            raise ValueError(
                f"SHAP values có kích thước không được hỗ trợ: {values.shape}"
            )

        # Lấy dữ liệu sau tiền xử lý
        transformed_data = (
            transformed.toarray()
            if hasattr(transformed, "toarray")
            else np.asarray(transformed)
        )

        local_explanation = shap.Explanation(
            values=local_values,
            base_values=base_value,
            data=transformed_data[0],
            feature_names=list(feature_names),
        )

        st.subheader("Giải thích dự đoán bằng SHAP")

        st.caption(
            "Biểu đồ thể hiện mức đóng góp của từng đặc trưng "
            "vào xác suất dự đoán hỏng hóc của mẫu đang xét. "
            "Đây không phải kết luận nhân quả."
        )

        # Tạo biểu đồ SHAP
        fig = plt.figure(figsize=(10, 6))

        shap.plots.waterfall(
            local_explanation,
            max_display=8,
            show=False,
        )

        # Lấy đúng figure do SHAP sử dụng
        fig = plt.gcf()
        fig.set_size_inches(10, 6)

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True,
            width="stretch",
        )

        plt.close(fig)

    except Exception as error:
        st.warning(
            f"Chưa thể tạo giải thích SHAP cho mẫu này: {error}"
        )



def risk_description(prediction, probability):
    if prediction == 1:
        return (
            "Mô hình xếp thiết bị vào nhóm có nguy cơ hỏng. "
            "Nên ưu tiên kiểm tra thông số vận hành và xác minh tình trạng thực tế."
        )
    return (
        "Mô hình chưa xếp thiết bị vào nhóm có nguy cơ hỏng tại ngưỡng "
        f"{PREDICTION_THRESHOLD:.2f}. Tiếp tục giám sát theo quy trình vận hành."
    )


def show_prediction_result(prediction, probability):
    st.subheader("Kết quả dự đoán")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ CÓ NGUY CƠ HỎNG HÓC")
            st.write(
                "Mô hình phân loại thiết bị vào nhóm "
                "có nguy cơ hỏng hóc."
            )
        else:
            st.success("✅ THIẾT BỊ BÌNH THƯỜNG")
            st.write(
                "Mô hình phân loại thiết bị vào nhóm "
                "bình thường."
            )

    with col2:
        st.metric(
            "Điểm xác suất dự đoán hỏng",
            f"{probability * 100:.2f}%"
        )
        st.progress(float(probability))

    st.caption(f"Ngưỡng phân loại đang áp dụng: {PREDICTION_THRESHOLD:.2f}")
    st.info(risk_description(prediction, probability))

    st.caption(
        "Đây là xác suất do mô hình ước lượng, "
        "không phải xác nhận chắc chắn tình trạng "
        "thực tế của thiết bị."
    )




# =====================================================
# 5. HÀM ĐỐI CHIẾU TRƯỚC / SAU BẢO TRÌ
# =====================================================

def get_maintenance_comparison(record):
    """So sánh lần dự đoán gần nhất trước hoàn thành và lần đầu sau hoàn thành."""
    if not record["completed_at"]:
        return None

    machine_history = get_prediction_history(
        machine_id=int(record["machine_id"]),
        limit=1000
    )
    if not machine_history:
        return None

    history_df = pd.DataFrame([dict(row) for row in machine_history])
    history_df["created_dt"] = pd.to_datetime(
        history_df["created_at"], errors="coerce"
    )
    completed_dt = pd.to_datetime(record["completed_at"], errors="coerce")

    if pd.isna(completed_dt):
        return None

    before = history_df[history_df["created_dt"] <= completed_dt].sort_values(
        ["created_dt", "id"], ascending=[False, False]
    )
    after = history_df[history_df["created_dt"] > completed_dt].sort_values(
        ["created_dt", "id"], ascending=[True, True]
    )

    if before.empty:
        return None

    before_row = before.iloc[0]
    after_row = after.iloc[0] if not after.empty else None

    return {
        "before": before_row,
        "after": after_row,
        "delta": (
            float(after_row["failure_probability"])
            - float(before_row["failure_probability"])
            if after_row is not None else None
        )
    }


# =====================================================
# 6. TRANG TỔNG QUAN - DASHBOARD VẬN HÀNH
# =====================================================

if page == "Tổng quan":

    st.title("Trung tâm điều hành bảo trì dự đoán")
    st.caption(
        "Tổng hợp tình trạng thiết bị, cảnh báo từ mô hình và tiến độ bảo trì "
        "từ dữ liệu đang lưu trong hệ thống."
    )

    dashboard_machines = get_all_machines()
    dashboard_predictions = get_prediction_history(limit=2000)
    dashboard_maintenance = get_maintenance_records(limit=1000)

    machine_df = pd.DataFrame([dict(m) for m in dashboard_machines]) if dashboard_machines else pd.DataFrame()
    prediction_df = pd.DataFrame([dict(r) for r in dashboard_predictions]) if dashboard_predictions else pd.DataFrame()

    # Lấy kết quả mới nhất của từng thiết bị, không đếm các lần dự đoán cũ.
    if not prediction_df.empty:
        prediction_df["created_dt"] = pd.to_datetime(prediction_df["created_at"], errors="coerce")
        prediction_df["failure_probability"] = pd.to_numeric(prediction_df["failure_probability"], errors="coerce")
        prediction_df["predicted_failure"] = pd.to_numeric(prediction_df["predicted_failure"], errors="coerce")
        registered_predictions = prediction_df[prediction_df["machine_id"].notna()].copy()
        latest_dashboard = registered_predictions.sort_values(
            ["created_dt", "id"], ascending=[False, False]
        ).drop_duplicates("machine_id", keep="first")
    else:
        latest_dashboard = pd.DataFrame()

    total_machines = len(dashboard_machines)
    active_machines = sum(m["status"] == "active" for m in dashboard_machines)
    in_maintenance_machines = sum(m["status"] == "maintenance" for m in dashboard_machines)
    inactive_machines = sum(m["status"] == "inactive" for m in dashboard_machines)
    monitored_machines = len(latest_dashboard)
    risky_machines = int((latest_dashboard["predicted_failure"] == 1).sum()) if monitored_machines else 0
    pending_jobs = sum(r["status"] == "pending" for r in dashboard_maintenance)
    active_jobs = sum(r["status"] == "in_progress" for r in dashboard_maintenance)
    completed_jobs = sum(r["status"] == "completed" for r in dashboard_maintenance)

    # Hàng KPI chính
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TỔNG THIẾT BỊ", total_machines, help="Số hồ sơ thiết bị đã đăng ký")
    k2.metric("ĐANG HOẠT ĐỘNG", active_machines)
    k3.metric("CẢNH BÁO NGUY CƠ", risky_machines, delta="Cần kiểm tra" if risky_machines else "Không có cảnh báo mới", delta_color="inverse")
    k4.metric("ĐANG BẢO TRÌ", in_maintenance_machines)

    st.markdown("### Tình hình vận hành")
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Trạng thái thiết bị")
        status_df = pd.DataFrame({
            "Trạng thái": ["Đang hoạt động", "Đang bảo trì", "Ngừng hoạt động"],
            "Số lượng": [active_machines, in_maintenance_machines, inactive_machines]
        })
        if total_machines:
            fig, ax = plt.subplots(figsize=(7, 3.3))
            bars = ax.barh(status_df["Trạng thái"], status_df["Số lượng"], color=["#e58a22", "#ffc533", "#68737a"])
            ax.set_xlabel("Số thiết bị")
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
            for bar in bars:
                ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2, f"{int(bar.get_width())}", va="center", color="#f2f3f4")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        else:
            st.info("Chưa có thiết bị. Hãy đăng ký thiết bị để bắt đầu theo dõi.")

    with right:
        st.markdown("#### Công việc bảo trì")
        job_cols = st.columns(3)
        job_cols[0].metric("Chờ xử lý", pending_jobs)
        job_cols[1].metric("Đang thực hiện", active_jobs)
        job_cols[2].metric("Hoàn thành", completed_jobs)
        total_jobs = len(dashboard_maintenance)
        if total_jobs:
            progress = completed_jobs / total_jobs
            st.progress(progress, text=f"{progress*100:.0f}% phiếu bảo trì đã hoàn thành")
        else:
            st.caption("Chưa có phiếu bảo trì nào được tạo.")
        st.markdown("#### Phạm vi giám sát")
        st.metric("Thiết bị đã có dự đoán", monitored_machines, help="Thiết bị có ít nhất một bản ghi dự đoán được gắn mã máy")
        st.caption(f"Ngưỡng phân loại hiện tại: {PREDICTION_THRESHOLD:.2f}")

    st.divider()
    st.markdown("### Thiết bị cần chú ý")
    if not latest_dashboard.empty:
        risk_table = latest_dashboard[latest_dashboard["predicted_failure"] == 1].copy()
        if not risk_table.empty:
            risk_view = risk_table[["machine_code", "machine_name", "failure_probability", "threshold", "created_at"]].copy()
            risk_view["failure_probability"] = (risk_view["failure_probability"] * 100).round(2)
            risk_view.columns = ["Mã thiết bị", "Tên thiết bị", "Xác suất hỏng (%)", "Threshold", "Lần dự đoán gần nhất"]
            st.dataframe(risk_view, width="stretch", hide_index=True)
            st.warning(f"Có {len(risk_table)} thiết bị đang có kết quả dự đoán thuộc nhóm nguy cơ hỏng.")
        else:
            st.success("Không có thiết bị nào thuộc nhóm nguy cơ hỏng theo lần dự đoán gần nhất.")
    else:
        st.info("Chưa có lịch sử dự đoán gắn với thiết bị. Hãy thực hiện dự đoán để kích hoạt giám sát.")

    st.divider()
    st.markdown("### Hoạt động gần đây")
    recent_cols = st.columns(2)
    with recent_cols[0]:
        st.markdown("#### Dự đoán gần đây")
        if not prediction_df.empty:
            recent_predictions = prediction_df.head(6).copy()
            recent_predictions["Thiết bị"] = recent_predictions["machine_code"].fillna("Dự đoán thử nghiệm")
            recent_predictions["Xác suất hỏng (%)"] = (recent_predictions["failure_probability"] * 100).round(2)
            recent_predictions["Kết quả"] = recent_predictions["predicted_failure"].map({0: "Bình thường", 1: "Có nguy cơ"})
            st.dataframe(recent_predictions[["created_at", "Thiết bị", "Xác suất hỏng (%)", "Kết quả"]].rename(columns={"created_at": "Thời gian"}), width="stretch", hide_index=True)
        else:
            st.caption("Chưa có hoạt động dự đoán.")
    with recent_cols[1]:
        st.markdown("#### Phiếu bảo trì gần đây")
        if dashboard_maintenance:
            recent_jobs = pd.DataFrame([{
                "Mã phiếu": r["id"],
                "Thiết bị": r["machine_code"],
                "Công việc": r["title"],
                "Ưu tiên": r["priority"].upper(),
                "Trạng thái": {"pending": "Chờ xử lý", "in_progress": "Đang thực hiện", "completed": "Hoàn thành", "cancelled": "Đã hủy"}.get(r["status"], r["status"])
            } for r in dashboard_maintenance[:6]])
            st.dataframe(recent_jobs, width="stretch", hide_index=True)
        else:
            st.caption("Chưa có phiếu bảo trì.")

    # Đối chiếu trước/sau bảo trì: dựa trên timestamp hoàn thành và lịch sử dự đoán.
    completed_for_compare = [r for r in dashboard_maintenance if r["status"] == "completed" and r["completed_at"]]
    st.divider()
    st.markdown("### Hiệu quả đánh giá sau bảo trì")
    if completed_for_compare:
        comparison_rows = []
        for maintenance in completed_for_compare:
            comparison = get_maintenance_comparison(maintenance)
            if comparison is None:
                continue
            before_row, after_row = comparison["before"], comparison["after"]
            comparison_rows.append({
                "Phiếu": f'#{maintenance["id"]}',
                "Thiết bị": maintenance["machine_code"],
                "Xác suất trước (%)": round(float(before_row["failure_probability"]) * 100, 2),
                "Xác suất sau (%)": round(float(after_row["failure_probability"]) * 100, 2) if after_row is not None else "Chưa đánh giá",
                "Thay đổi (điểm %)": round(comparison["delta"] * 100, 2) if comparison["delta"] is not None else "—",
                "Kết quả sau bảo trì": ("Có nguy cơ" if int(after_row["predicted_failure"]) == 1 else "Chưa phát hiện nguy cơ") if after_row is not None else "Chờ đánh giá lại"
            })
        if comparison_rows:
            st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
            st.caption("So sánh lần dự đoán gần nhất trước thời điểm hoàn thành phiếu với lần đánh giá đầu tiên sau thời điểm đó. Đây là so sánh kết quả mô hình, không phải bằng chứng nhân quả về hiệu quả sửa chữa.")
        else:
            st.info("Chưa đủ dữ liệu dự đoán trước thời điểm hoàn thành để lập đối chiếu.")
    else:
        st.info("Khi có phiếu bảo trì hoàn thành và đánh giá lại sau bảo trì, bảng đối chiếu sẽ xuất hiện tại đây.")

    with st.expander("Thông tin bộ dữ liệu AI4I 2020 (phục vụ nghiên cứu mô hình)"):
        total = len(df)
        failures = int(df["Machine failure"].sum())
        normal = total - failures
        st.caption("Các số liệu dưới đây thuộc bộ dữ liệu huấn luyện/đánh giá, không phải thống kê tình trạng thiết bị đang quản lý.")
        d1, d2, d3 = st.columns(3)
        d1.metric("Tổng bản ghi AI4I", f"{total:,}")
        d2.metric("Bản ghi bình thường", f"{normal:,}")
        d3.metric("Bản ghi hỏng", f"{failures:,}")
        st.dataframe(df.head(8), width="stretch", hide_index=True)


# =====================================================
# 6. QUẢN LÝ THIẾT BỊ
# =====================================================

elif page == "Quản lý thiết bị":

    st.title("Quản lý thiết bị")
    st.write(
        "Quản lý hồ sơ thiết bị, thông tin vận hành "
        "và trạng thái hoạt động trong hệ thống."
    )

    machines = get_all_machines()

    # -------------------------------------------------
    # 1. THỐNG KÊ THIẾT BỊ
    # -------------------------------------------------

    total_machines = len(machines)
    active_count = sum(m["status"] == "active" for m in machines)
    maintenance_count = sum(
        m["status"] == "maintenance" for m in machines
    )
    inactive_count = sum(
        m["status"] == "inactive" for m in machines
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tổng thiết bị", total_machines)
    col2.metric("Đang hoạt động", active_count)
    col3.metric("Đang bảo trì", maintenance_count)
    col4.metric("Ngừng hoạt động", inactive_count)

    st.divider()

    # -------------------------------------------------
    # 2. DANH SÁCH THIẾT BỊ
    # -------------------------------------------------

    st.subheader("Danh sách thiết bị")

    if machines:
        machine_data = []

        status_labels = {
            "active": "Đang hoạt động",
            "maintenance": "Đang bảo trì",
            "inactive": "Ngừng hoạt động"
        }

        type_labels = {
            "L": "Low",
            "M": "Medium",
            "H": "High"
        }

        for machine in machines:
            machine_data.append({
                "Mã thiết bị": machine["machine_code"],
                "Tên thiết bị": machine["machine_name"],
                "Loại": type_labels.get(
                    machine["machine_type"],
                    machine["machine_type"]
                ),
                "Vị trí": machine["location"] or "Chưa cập nhật",
                "Nhà sản xuất": machine["manufacturer"] or "Chưa cập nhật",
                "Model": machine["model_number"] or "Chưa cập nhật",
                "Trạng thái": status_labels.get(
                    machine["status"],
                    machine["status"]
                ),
                "Ngày tạo": machine["created_at"]
            })

        st.dataframe(
            pd.DataFrame(machine_data),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("Chưa có thiết bị nào trong hệ thống.")

    st.divider()

    # -------------------------------------------------
    # 3. THÊM THIẾT BỊ
    # -------------------------------------------------

    st.subheader("Thêm thiết bị mới")

    with st.form("add_machine_form", clear_on_submit=True):

        col1, col2 = st.columns(2)

        with col1:
            new_code = st.text_input(
                "Mã thiết bị *",
                placeholder="Ví dụ: CNC-002"
            )

            new_name = st.text_input(
                "Tên thiết bị *",
                placeholder="Ví dụ: Máy CNC số 2"
            )

            new_type = st.selectbox(
                "Loại máy *",
                ["L", "M", "H"],
                help="L: Low, M: Medium, H: High"
            )

            new_location = st.text_input(
                "Vị trí",
                placeholder="Ví dụ: Xưởng A"
            )

        with col2:
            new_manufacturer = st.text_input(
                "Nhà sản xuất"
            )

            new_model = st.text_input(
                "Model thiết bị"
            )

            new_installation_date = st.text_input(
                "Ngày lắp đặt",
                placeholder="YYYY-MM-DD"
            )

            new_notes = st.text_area(
                "Ghi chú"
            )

        add_submitted = st.form_submit_button(
            "Thêm thiết bị",
            width="stretch"
        )

    if add_submitted:

        if not new_code.strip() or not new_name.strip():
            st.error("Vui lòng nhập mã và tên thiết bị.")

        else:
            try:
                add_machine(
                    machine_code=new_code,
                    machine_name=new_name,
                    machine_type=new_type,
                    location=new_location or None,
                    manufacturer=new_manufacturer or None,
                    model_number=new_model or None,
                    installation_date=(
                        new_installation_date or None
                    ),
                    notes=new_notes or None
                )

                st.success(
                    f"Đã thêm thiết bị {new_code.strip()} thành công."
                )
                st.rerun()

            except Exception as error:
                st.error(
                    f"Không thể thêm thiết bị: {error}"
                )

    st.divider()

    # -------------------------------------------------
    # 4. CHỈNH SỬA THIẾT BỊ
    # -------------------------------------------------

    st.subheader("Chỉnh sửa thông tin thiết bị")

    if machines:

        selected_machine_id = st.selectbox(
            "Chọn thiết bị cần chỉnh sửa",
            options=[m["id"] for m in machines],
            format_func=lambda machine_id: next(
                (
                    f'{m["machine_code"]} - {m["machine_name"]}'
                    for m in machines
                    if m["id"] == machine_id
                ),
                str(machine_id)
            ),
            key="edit_machine_selector"
        )

        selected_machine = get_machine_by_id(
            selected_machine_id
        )

        if selected_machine:

            with st.form("edit_machine_form"):

                col1, col2 = st.columns(2)

                with col1:
                    edit_code = st.text_input(
                        "Mã thiết bị",
                        value=selected_machine["machine_code"]
                    )

                    edit_name = st.text_input(
                        "Tên thiết bị",
                        value=selected_machine["machine_name"]
                    )

                    edit_type = st.selectbox(
                        "Loại máy",
                        ["L", "M", "H"],
                        index=["L", "M", "H"].index(
                            selected_machine["machine_type"]
                        )
                    )

                    edit_location = st.text_input(
                        "Vị trí",
                        value=selected_machine["location"] or ""
                    )

                with col2:
                    edit_manufacturer = st.text_input(
                        "Nhà sản xuất",
                        value=selected_machine["manufacturer"] or ""
                    )

                    edit_model = st.text_input(
                        "Model thiết bị",
                        value=selected_machine["model_number"] or ""
                    )

                    edit_installation_date = st.text_input(
                        "Ngày lắp đặt (YYYY-MM-DD)",
                        value=selected_machine["installation_date"] or ""
                    )

                    edit_status = st.selectbox(
                        "Trạng thái",
                        ["active", "maintenance", "inactive"],
                        index=[
                            "active",
                            "maintenance",
                            "inactive"
                        ].index(selected_machine["status"]),
                        format_func=lambda value: {
                            "active": "Đang hoạt động",
                            "maintenance": "Đang bảo trì",
                            "inactive": "Ngừng hoạt động"
                        }[value]
                    )

                edit_notes = st.text_area(
                    "Ghi chú",
                    value=selected_machine["notes"] or ""
                )

                edit_submitted = st.form_submit_button(
                    "Lưu thay đổi",
                    width="stretch"
                )

            if edit_submitted:

                if not edit_code.strip() or not edit_name.strip():
                    st.error("Mã và tên thiết bị không được để trống.")

                else:
                    try:
                        update_machine(
                            selected_machine_id,
                            machine_code=edit_code.strip(),
                            machine_name=edit_name.strip(),
                            machine_type=edit_type,
                            location=edit_location or None,
                            manufacturer=edit_manufacturer or None,
                            model_number=edit_model or None,
                            installation_date=(
                                edit_installation_date or None
                            ),
                            status=edit_status,
                            notes=edit_notes or None
                        )

                        st.success("Đã cập nhật thiết bị.")
                        st.rerun()

                    except Exception as error:
                        st.error(
                            f"Không thể cập nhật thiết bị: {error}"
                        )

    else:
        st.info("Cần thêm thiết bị trước khi chỉnh sửa.")

    st.divider()

    # -------------------------------------------------
    # 5. NGỪNG THEO DÕI THIẾT BỊ
    # -------------------------------------------------

    st.subheader("Ngừng theo dõi thiết bị")

    active_machines = [
        m for m in machines
        if m["status"] != "inactive"
    ]

    if active_machines:

        deactivate_id = st.selectbox(
            "Chọn thiết bị",
            options=[m["id"] for m in active_machines],
            format_func=lambda machine_id: next(
                (
                    f'{m["machine_code"]} - {m["machine_name"]}'
                    for m in active_machines
                    if m["id"] == machine_id
                ),
                str(machine_id)
            ),
            key="deactivate_machine_selector"
        )

        if st.button(
            "Ngừng theo dõi thiết bị",
            type="secondary"
        ):
            delete_machine(deactivate_id)
            st.success(
                "Thiết bị đã được chuyển sang trạng thái ngừng hoạt động."
            )
            st.rerun()

    else:
        st.info("Không có thiết bị đang được theo dõi.")


# =====================================================
# 7. DỰ ĐOÁN MỘT MÁY VÀ LƯU LỊCH SỬ
# =====================================================

# =====================================================
# GIÁM SÁT VÀ CẢNH BÁO
# =====================================================

elif page == "Giám sát & Cảnh báo":

    st.title("Giám sát và cảnh báo thiết bị")

    st.write(
        "Theo dõi kết quả dự đoán gần nhất của từng thiết bị "
        "dựa trên lịch sử được lưu trong hệ thống."
    )

    # -------------------------------------------------
    # 1. LẤY DỮ LIỆU
    # -------------------------------------------------

    all_machines = get_all_machines()
    all_history = get_prediction_history(limit=1000)

    if not all_history:
        st.info(
            "Chưa có dữ liệu dự đoán. "
            "Hãy thực hiện dự đoán để bắt đầu giám sát thiết bị."
        )

    else:

        history_df = pd.DataFrame([
            dict(record) for record in all_history
        ])

        # Chuyển kiểu dữ liệu để tính toán và hiển thị
        history_df["failure_probability"] = pd.to_numeric(
            history_df["failure_probability"],
            errors="coerce"
        )

        history_df["predicted_failure"] = pd.to_numeric(
            history_df["predicted_failure"],
            errors="coerce"
        )

        # -------------------------------------------------
        # 2. LẤY LẦN DỰ ĐOÁN GẦN NHẤT CỦA TỪNG THIẾT BỊ
        # -------------------------------------------------

        machine_history = history_df[
            history_df["machine_id"].notna()
        ].copy()

        machine_history = machine_history.sort_values(
            ["created_at", "id"],
            ascending=[False, False]
        )

        latest_by_machine = machine_history.drop_duplicates(
            subset=["machine_id"],
            keep="first"
        )

        # -------------------------------------------------
        # 3. THỐNG KÊ
        # -------------------------------------------------

        monitored_count = len(latest_by_machine)

        risky_count = int(
            (latest_by_machine["predicted_failure"] == 1).sum()
        )

        normal_count = monitored_count - risky_count

        unassigned_count = int(
            history_df["machine_id"].isna().sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Thiết bị có lịch sử",
            monitored_count
        )

        col2.metric(
            "Có nguy cơ hỏng",
            risky_count
        )

        col3.metric(
            "Chưa phát hiện nguy cơ",
            normal_count
        )

        col4.metric(
            "Dự đoán thử nghiệm",
            unassigned_count
        )

        st.divider()

        # -------------------------------------------------
        # 4. DANH SÁCH CẢNH BÁO
        # -------------------------------------------------

        st.subheader("Danh sách thiết bị cần chú ý")

        risky_devices = latest_by_machine[
            latest_by_machine["predicted_failure"] == 1
        ].copy()

        if not risky_devices.empty:

            risky_display = pd.DataFrame({
                "Mã thiết bị": risky_devices["machine_code"],
                "Tên thiết bị": risky_devices["machine_name"],
                "Xác suất hỏng (%)": (
                    risky_devices["failure_probability"] * 100
                ).round(2),
                "Threshold": risky_devices["threshold"],
                "Thời điểm dự đoán": risky_devices["created_at"],
                "Trạng thái dự đoán": "Có nguy cơ hỏng"
            })

            st.error(
                f"Phát hiện {len(risky_devices)} thiết bị "
                "có kết quả dự đoán thuộc nhóm nguy cơ hỏng."
            )

            st.dataframe(
                risky_display,
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Đây là cảnh báo dựa trên kết quả mô hình. "
                "Cần kiểm tra thực tế trước khi quyết định bảo trì."
            )

            # -------------------------------------------------
            # 4A. TẠO PHIẾU BẢO TRÌ TRỰC TIẾP TỪ CẢNH BÁO
            # -------------------------------------------------

            st.markdown("#### Tạo phiếu bảo trì từ cảnh báo")

            risky_machine_ids = [
                int(value) for value in risky_devices["machine_id"].tolist()
            ]

            selected_risky_machine_id = st.selectbox(
                "Chọn thiết bị cần kiểm tra",
                options=risky_machine_ids,
                format_func=lambda machine_id: next(
                    (
                        f'{row["machine_code"]} - {row["machine_name"]}'
                        for _, row in risky_devices.iterrows()
                        if int(row["machine_id"]) == machine_id
                    ),
                    str(machine_id)
                ),
                key="alert_maintenance_machine"
            )

            selected_risky_machine = get_machine_by_id(
                selected_risky_machine_id
            )

            with st.form("create_maintenance_from_alert_form"):
                alert_title = st.text_input(
                    "Tiêu đề phiếu",
                    value="Kiểm tra thiết bị có cảnh báo nguy cơ hỏng"
                )

                alert_description = st.text_area(
                    "Nội dung cần kiểm tra",
                    value=(
                        "Thiết bị có kết quả dự đoán thuộc nhóm nguy cơ hỏng. "
                        "Cần kiểm tra thực tế và xác minh tình trạng vận hành."
                    )
                )

                alert_col1, alert_col2 = st.columns(2)

                with alert_col1:
                    alert_priority = st.selectbox(
                        "Mức ưu tiên",
                        ["medium", "high", "critical"],
                        index=1,
                        format_func=lambda value: {
                            "medium": "Trung bình",
                            "high": "Cao",
                            "critical": "Khẩn cấp"
                        }[value]
                    )

                with alert_col2:
                    alert_scheduled_date = st.date_input(
                        "Ngày dự kiến kiểm tra",
                        value=date.today(),
                        key="alert_maintenance_date"
                    )

                create_from_alert = st.form_submit_button(
                    "Tạo phiếu bảo trì",
                    width="stretch"
                )

            if create_from_alert:
                if not alert_title.strip():
                    st.error("Vui lòng nhập tiêu đề phiếu bảo trì.")
                elif selected_risky_machine is None:
                    st.error("Không tìm thấy hồ sơ thiết bị đã chọn.")
                else:
                    try:
                        new_record_id = add_maintenance_record(
                            machine_id=selected_risky_machine_id,
                            title=alert_title.strip(),
                            description=alert_description or None,
                            priority=alert_priority,
                            scheduled_date=alert_scheduled_date.isoformat()
                        )
                        st.success(
                            f"Đã tạo phiếu bảo trì #{new_record_id} "
                            f"cho {selected_risky_machine['machine_code']}."
                        )
                        st.rerun()
                    except Exception as error:
                        st.error(
                            f"Không thể tạo phiếu bảo trì: {error}"
                        )

        else:
            st.success(
                "Chưa có thiết bị nào được phân loại vào nhóm nguy cơ "
                "hỏng theo lần dự đoán gần nhất."
            )

        st.divider()

        # -------------------------------------------------
        # 5. TÌNH TRẠNG DỰ ĐOÁN GẦN NHẤT CỦA THIẾT BỊ
        # -------------------------------------------------

        st.subheader("Tình trạng dự đoán gần nhất")

        if not latest_by_machine.empty:

            latest_display = pd.DataFrame({
                "Mã thiết bị": latest_by_machine["machine_code"],
                "Tên thiết bị": latest_by_machine["machine_name"],
                "Xác suất hỏng (%)": (
                    latest_by_machine["failure_probability"] * 100
                ).round(2),
                "Kết quả": latest_by_machine[
                    "predicted_failure"
                ].map({
                    0: "Bình thường",
                    1: "Có nguy cơ hỏng"
                }),
                "Threshold": latest_by_machine["threshold"],
                "Thời điểm": latest_by_machine["created_at"]
            })

            st.dataframe(
                latest_display,
                width="stretch",
                hide_index=True
            )

        else:
            st.info(
                "Chưa có thiết bị đăng ký nào có lịch sử dự đoán."
            )

        st.divider()

        # -------------------------------------------------
        # 6. LỊCH SỬ DỰ ĐOÁN TOÀN HỆ THỐNG
        # -------------------------------------------------

        st.subheader("Lịch sử dự đoán toàn hệ thống")

        history_display = pd.DataFrame({
            "Thời gian": history_df["created_at"],
            "Mã thiết bị": history_df["machine_code"].fillna(
                "Không gắn thiết bị"
            ),
            "Tên thiết bị": history_df["machine_name"].fillna(
                "Dự đoán thử nghiệm"
            ),
            "Xác suất hỏng (%)": (
                history_df["failure_probability"] * 100
            ).round(2),
            "Kết quả": history_df["predicted_failure"].map({
                0: "Bình thường",
                1: "Có nguy cơ hỏng"
            }),
            "Threshold": history_df["threshold"],
            "Nguồn": history_df["source"]
        })

        st.dataframe(
            history_display,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Trang giám sát sử dụng tối đa 1.000 bản ghi dự đoán gần nhất."
        )

# =====================================================
# QUẢN LÝ BẢO TRÌ
# =====================================================

elif page == "Quản lý bảo trì":

    st.title("Quản lý bảo trì thiết bị")

    st.write(
        "Tạo và theo dõi các phiếu kiểm tra, bảo trì "
        "cho từng thiết bị trong hệ thống."
    )

    # -------------------------------------------------
    # 1. THỐNG KÊ PHIẾU BẢO TRÌ
    # -------------------------------------------------

    records = get_maintenance_records(limit=500)

    pending_count = sum(
        r["status"] == "pending" for r in records
    )

    progress_count = sum(
        r["status"] == "in_progress" for r in records
    )

    completed_count = sum(
        r["status"] == "completed" for r in records
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tổng phiếu", len(records))
    col2.metric("Chờ xử lý", pending_count)
    col3.metric("Đang thực hiện", progress_count)
    col4.metric("Đã hoàn thành", completed_count)

    st.divider()

    # -------------------------------------------------
    # 2. TẠO PHIẾU BẢO TRÌ
    # -------------------------------------------------

    st.subheader("Tạo phiếu bảo trì")

    machines = [
        m for m in get_all_machines()
        if m["status"] != "inactive"
    ]

    if machines:

        with st.form("create_maintenance_form"):

            selected_machine_id = st.selectbox(
                "Thiết bị cần bảo trì",
                options=[m["id"] for m in machines],
                format_func=lambda machine_id: next(
                    (
                        f'{m["machine_code"]} - {m["machine_name"]}'
                        for m in machines
                        if m["id"] == machine_id
                    ),
                    str(machine_id)
                )
            )

            title = st.text_input(
                "Tiêu đề công việc *",
                placeholder="Ví dụ: Kiểm tra tình trạng vận hành"
            )

            description = st.text_area(
                "Mô tả công việc",
                placeholder="Mô tả nội dung cần kiểm tra hoặc bảo trì..."
            )

            col1, col2 = st.columns(2)

            with col1:
                priority = st.selectbox(
                    "Mức ưu tiên",
                    ["low", "medium", "high", "critical"],
                    index=1,
                    format_func=lambda value: {
                        "low": "Thấp",
                        "medium": "Trung bình",
                        "high": "Cao",
                        "critical": "Khẩn cấp"
                    }[value]
                )

            with col2:
                scheduled_date = st.date_input(
                    "Ngày dự kiến bảo trì"
                )

            submitted = st.form_submit_button(
                "Tạo phiếu bảo trì",
                width="stretch"
            )

        if submitted:

            if not title.strip():
                st.error("Vui lòng nhập tiêu đề công việc.")

            else:
                try:
                    record_id = add_maintenance_record(
                        machine_id=selected_machine_id,
                        title=title,
                        description=description or None,
                        priority=priority,
                        scheduled_date=scheduled_date.isoformat()
                    )

                    st.success(
                        f"Đã tạo phiếu bảo trì #{record_id}."
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        f"Không thể tạo phiếu bảo trì: {error}"
                    )

    else:
        st.warning(
            "Chưa có thiết bị đang hoạt động. "
            "Hãy đăng ký thiết bị trước khi tạo phiếu."
        )

    st.divider()

    # -------------------------------------------------
    # 3. DANH SÁCH PHIẾU BẢO TRÌ
    # -------------------------------------------------

    st.subheader("Danh sách phiếu bảo trì")

    status_labels = {
        "pending": "Chờ xử lý",
        "in_progress": "Đang thực hiện",
        "completed": "Đã hoàn thành",
        "cancelled": "Đã hủy"
    }

    priority_labels = {
        "low": "Thấp",
        "medium": "Trung bình",
        "high": "Cao",
        "critical": "Khẩn cấp"
    }

    if records:

        display_records = []

        for record in records:
            display_records.append({
                "Mã phiếu": record["id"],
                "Mã thiết bị": record["machine_code"],
                "Tên thiết bị": record["machine_name"],
                "Công việc": record["title"],
                "Mức ưu tiên": priority_labels[record["priority"]],
                "Trạng thái": status_labels[record["status"]],
                "Ngày dự kiến": record["scheduled_date"] or "",
                "Ngày hoàn thành": record["completed_at"] or ""
            })

        st.dataframe(
            pd.DataFrame(display_records),
            width="stretch",
            hide_index=True
        )

        st.divider()

        # -------------------------------------------------
        # 4. CẬP NHẬT PHIẾU BẢO TRÌ
        # -------------------------------------------------

        st.subheader("Cập nhật phiếu bảo trì")

        selected_record_id = st.selectbox(
            "Chọn phiếu cần cập nhật",
            options=[r["id"] for r in records],
            format_func=lambda record_id: next(
                (
                    f'#{r["id"]} - {r["machine_code"]} - {r["title"]}'
                    for r in records
                    if r["id"] == record_id
                ),
                str(record_id)
            )
        )

        selected_record = next(
            r for r in records
            if r["id"] == selected_record_id
        )

        with st.form("update_maintenance_form"):

            new_status = st.selectbox(
                "Trạng thái",
                ["pending", "in_progress", "completed", "cancelled"],
                index=[
                    "pending",
                    "in_progress",
                    "completed",
                    "cancelled"
                ].index(selected_record["status"]),
                format_func=lambda value: status_labels[value]
            )

            new_priority = st.selectbox(
                "Mức ưu tiên",
                ["low", "medium", "high", "critical"],
                index=[
                    "low",
                    "medium",
                    "high",
                    "critical"
                ].index(selected_record["priority"]),
                format_func=lambda value: priority_labels[value]
            )

            new_description = st.text_area(
                "Mô tả",
                value=selected_record["description"] or ""
            )

            new_result_notes = st.text_area(
                "Kết quả kiểm tra / ghi chú sau bảo trì",
                value=selected_record["result_notes"] or ""
            )

            update_submitted = st.form_submit_button(
                "Lưu cập nhật",
                width="stretch"
            )

        if update_submitted:

            try:
                update_maintenance_record(
                    record_id=selected_record_id,
                    status=new_status,
                    priority=new_priority,
                    description=new_description,
                    result_notes=new_result_notes
                )

                st.success("Đã cập nhật phiếu bảo trì.")
                st.rerun()

            except Exception as error:
                st.error(
                    f"Không thể cập nhật phiếu: {error}"
                )

    else:
        st.info(
            "Chưa có phiếu bảo trì nào. "
            "Hãy tạo phiếu đầu tiên ở phía trên."
        )

    # -------------------------------------------------
    # 5. ĐÁNH GIÁ LẠI THIẾT BỊ SAU BẢO TRÌ
    # -------------------------------------------------

    st.divider()
    st.subheader("Đánh giá lại thiết bị sau bảo trì")
    st.caption(
        "Hoàn thành phiếu chỉ xác nhận công việc bảo trì đã kết thúc. "
        "Để xác nhận tình trạng thiết bị, cần nhập các thông số vận hành "
        "được đo lại sau bảo trì và chạy dự đoán mới."
    )

    completed_records = [r for r in records if r["status"] == "completed"]

    if completed_records:
        selected_recheck_record_id = st.selectbox(
            "Chọn phiếu bảo trì đã hoàn thành",
            options=[r["id"] for r in completed_records],
            format_func=lambda record_id: next(
                (
                    f'#{r["id"]} - {r["machine_code"]} - {r["title"]}'
                    for r in completed_records
                    if r["id"] == record_id
                ),
                str(record_id)
            ),
            key="post_maintenance_record_selector"
        )

        selected_recheck_record = next(
            r for r in completed_records
            if r["id"] == selected_recheck_record_id
        )
        recheck_machine_id = int(selected_recheck_record["machine_id"])
        recheck_machine = get_machine_by_id(recheck_machine_id)
        previous_readings = get_prediction_history(
            machine_id=recheck_machine_id, limit=1
        )

        if recheck_machine is not None:
            if previous_readings:
                previous = previous_readings[0]
                st.info(
                    f'Đang đánh giá lại **{recheck_machine["machine_code"]}**. '
                    "Các giá trị bên dưới được điền từ lần dự đoán gần nhất "
                    "để tham khảo. Hãy thay bằng số đo thực tế sau bảo trì."
                )
                default_air = float(previous["air_temperature"])
                default_process = float(previous["process_temperature"])
                default_speed = float(previous["rotational_speed"])
                default_torque = float(previous["torque"])
                default_wear = float(previous["tool_wear"])
            else:
                st.warning(
                    "Thiết bị chưa có thông số dự đoán trước đó. "
                    "Hãy nhập số đo thực tế sau bảo trì."
                )
                default_air, default_process = 300.0, 310.0
                default_speed, default_torque, default_wear = 1500.0, 40.0, 100.0

            st.markdown(
                f'**Thiết bị:** `{recheck_machine["machine_code"]}` · '
                f'**Loại máy:** `{recheck_machine["machine_type"]}` · '
                f'**Phiếu:** `#{selected_recheck_record_id}`'
            )

            with st.form("post_maintenance_recheck_form"):
                recheck_col1, recheck_col2 = st.columns(2)
                with recheck_col1:
                    st.text_input(
                        "Loại máy",
                        value=recheck_machine["machine_type"],
                        disabled=True
                    )
                    recheck_air = st.number_input(
                        "Nhiệt độ không khí (K)", 250.0, 350.0,
                        value=default_air, step=0.1
                    )
                    recheck_process = st.number_input(
                        "Nhiệt độ quy trình (K)", 250.0, 400.0,
                        value=default_process, step=0.1
                    )
                with recheck_col2:
                    recheck_speed = st.number_input(
                        "Tốc độ quay (rpm)", 100.0, 5000.0,
                        value=default_speed, step=10.0
                    )
                    recheck_torque = st.number_input(
                        "Mô-men xoắn (Nm)", 0.0, 100.0,
                        value=default_torque, step=0.1
                    )
                    recheck_wear = st.number_input(
                        "Độ mài mòn dụng cụ (phút)", 0.0, 300.0,
                        value=default_wear, step=1.0
                    )

                recheck_submitted = st.form_submit_button(
                    "Chạy đánh giá sau bảo trì",
                    width="stretch"
                )

            if recheck_submitted:
                recheck_input = pd.DataFrame([{
                    "Type": recheck_machine["machine_type"],
                    "Air temperature [K]": recheck_air,
                    "Process temperature [K]": recheck_process,
                    "Rotational speed [rpm]": recheck_speed,
                    "Torque [Nm]": recheck_torque,
                    "Tool wear [min]": recheck_wear
                }])

                try:
                    recheck_predictions, recheck_probabilities = run_prediction(
                        model, recheck_input, PREDICTION_THRESHOLD
                    )
                    recheck_prediction = int(recheck_predictions[0])
                    recheck_probability = float(recheck_probabilities[0])

                    recheck_prediction_id = save_prediction(
                        machine_id=recheck_machine_id,
                        input_data=recheck_input.iloc[0].to_dict(),
                        probability=recheck_probability,
                        prediction=recheck_prediction,
                        threshold=PREDICTION_THRESHOLD,
                        source="monitoring",
                        model_name="Random Forest"
                    )

                    st.markdown("#### Kết quả đánh giá sau bảo trì")
                    result_col1, result_col2 = st.columns([2, 1])
                    with result_col1:
                        if recheck_prediction == 0:
                            st.success(
                                "ĐÁNH GIÁ MỚI: CHƯA PHÁT HIỆN NGUY CƠ HỎNG. "
                                "Cảnh báo cũ sẽ không còn là trạng thái mới nhất."
                            )
                        else:
                            st.error(
                                "ĐÁNH GIÁ MỚI: VẪN CÓ NGUY CƠ HỎNG. "
                                "Thiết bị tiếp tục nằm trong danh sách cảnh báo."
                            )
                    with result_col2:
                        st.metric(
                            "Xác suất hỏng mới",
                            f"{recheck_probability * 100:.2f}%"
                        )

                    st.caption(
                        f"Đã lưu lần đánh giá mới #{recheck_prediction_id}. "
                        "Thông số hồ sơ thiết bị không bị thay đổi; đây là một "
                        "bản ghi đo lường/dự đoán mới trong lịch sử."
                    )
                except Exception as error:
                    st.error(f"Không thể đánh giá lại thiết bị: {error}")
        else:
            st.warning("Không tìm thấy hồ sơ thiết bị của phiếu đã chọn.")
    else:
        st.info(
            "Chưa có phiếu bảo trì hoàn thành. Khi có phiếu hoàn thành, "
            "khu vực đánh giá lại sẽ xuất hiện tại đây."
        )

    # -------------------------------------------------
    # 6. LỊCH SỬ BẢO TRÌ THEO THIẾT BỊ
    # -------------------------------------------------

    st.divider()
    st.subheader("Lịch sử bảo trì theo thiết bị")

    history_machines = get_all_machines()

    if history_machines:
        history_machine_id = st.selectbox(
            "Chọn thiết bị để xem lịch sử bảo trì",
            options=[m["id"] for m in history_machines],
            format_func=lambda machine_id: next(
                (
                    f'{m["machine_code"]} - {m["machine_name"]}'
                    for m in history_machines
                    if m["id"] == machine_id
                ),
                str(machine_id)
            ),
            key="maintenance_history_machine"
        )

        machine_maintenance_history = get_maintenance_records(
            machine_id=history_machine_id,
            limit=500
        )

        if machine_maintenance_history:
            machine_history_display = []

            for record in machine_maintenance_history:
                machine_history_display.append({
                    "Mã phiếu": record["id"],
                    "Công việc": record["title"],
                    "Mô tả": record["description"] or "",
                    "Mức ưu tiên": priority_labels.get(
                        record["priority"], record["priority"]
                    ),
                    "Trạng thái": status_labels.get(
                        record["status"], record["status"]
                    ),
                    "Ngày tạo": record["created_at"],
                    "Ngày dự kiến": record["scheduled_date"] or "",
                    "Ngày hoàn thành": record["completed_at"] or "",
                    "Kết quả / ghi chú": record["result_notes"] or ""
                })

            st.dataframe(
                pd.DataFrame(machine_history_display),
                width="stretch",
                hide_index=True
            )
        else:
            st.info(
                "Thiết bị này chưa có phiếu bảo trì nào."
            )
    else:
        st.info(
            "Chưa có hồ sơ thiết bị để tra cứu lịch sử bảo trì."
        )


    # -------------------------------------------------
    # 7. ĐỐI CHIẾU KẾT QUẢ TRƯỚC VÀ SAU BẢO TRÌ
    # -------------------------------------------------

    st.divider()
    st.subheader("Đối chiếu trước và sau bảo trì")
    st.caption(
        "Chọn một phiếu đã hoàn thành để so sánh lần dự đoán gần nhất "
        "trước khi hoàn thành với lần đánh giá đầu tiên sau đó."
    )

    completed_for_detail = [
        r for r in records
        if r["status"] == "completed" and r["completed_at"]
    ]

    if completed_for_detail:
        compare_record_id = st.selectbox(
            "Phiếu cần đối chiếu",
            options=[r["id"] for r in completed_for_detail],
            format_func=lambda rid: next(
                f'#{r["id"]} - {r["machine_code"]} - {r["title"]}'
                for r in completed_for_detail if r["id"] == rid
            ),
            key="maintenance_before_after_selector"
        )
        compare_record = next(r for r in completed_for_detail if r["id"] == compare_record_id)
        comparison = get_maintenance_comparison(compare_record)

        if comparison is None:
            st.warning(
                "Chưa có bản ghi dự đoán trước thời điểm hoàn thành phiếu. "
                "Không thể lập so sánh cho phiếu này."
            )
        else:
            before_row = comparison["before"]
            after_row = comparison["after"]
            before_prob = float(before_row["failure_probability"])
            before_pred = int(before_row["predicted_failure"])

            st.markdown(
                f'**Thiết bị:** `{compare_record["machine_code"]}` · '
                f'**Hoàn thành phiếu:** `{compare_record["completed_at"]}`'
            )
            before_col, after_col = st.columns(2)
            with before_col:
                st.markdown("#### Trước bảo trì")
                st.metric("Xác suất hỏng", f"{before_prob * 100:.2f}%")
                st.caption(f'Bản ghi #{int(before_row["id"])} · {before_row["created_at"]}')
                if before_pred:
                    st.error("Có nguy cơ hỏng")
                else:
                    st.success("Chưa phát hiện nguy cơ")
            with after_col:
                st.markdown("#### Sau bảo trì")
                if after_row is not None:
                    after_prob = float(after_row["failure_probability"])
                    after_pred = int(after_row["predicted_failure"])
                    st.metric(
                        "Xác suất hỏng",
                        f"{after_prob * 100:.2f}%",
                        delta=f"{(after_prob - before_prob) * 100:+.2f} điểm %",
                        delta_color="inverse"
                    )
                    st.caption(f'Bản ghi #{int(after_row["id"])} · {after_row["created_at"]}')
                    if after_pred:
                        st.error("Vẫn có nguy cơ hỏng · cần tiếp tục kiểm tra")
                    else:
                        st.success("Chưa phát hiện nguy cơ theo lần đánh giá mới")
                else:
                    st.metric("Xác suất hỏng", "Chưa đánh giá")
                    st.warning(
                        "Phiếu đã hoàn thành nhưng chưa có dự đoán mới sau thời điểm hoàn thành. "
                        "Hãy nhập số đo thực tế tại mục Đánh giá lại thiết bị sau bảo trì."
                    )

            if after_row is not None:
                comparison_chart = pd.DataFrame({
                    "Giai đoạn": ["Trước bảo trì", "Sau bảo trì"],
                    "Xác suất hỏng (%)": [before_prob * 100, float(after_row["failure_probability"]) * 100]
                })
                fig, ax = plt.subplots(figsize=(7, 3.2))
                bars = ax.bar(
                    comparison_chart["Giai đoạn"],
                    comparison_chart["Xác suất hỏng (%)"],
                    color=["#e58a22", "#ffc533"],
                    width=0.48
                )
                ax.set_ylim(0, 100)
                ax.set_ylabel("Xác suất mô hình (%)")
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
                for bar, value in zip(bars, comparison_chart["Xác suất hỏng (%)"]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{value:.2f}%", ha="center", color="#f2f3f4")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                st.caption(
                    "Diễn giải: mức thay đổi là chênh lệch giữa hai xác suất do mô hình ước lượng. "
                    "Không thể kết luận việc bảo trì là nguyên nhân duy nhất làm xác suất thay đổi."
                )
    else:
        st.info("Chưa có phiếu bảo trì hoàn thành để đối chiếu.")

elif page == "Dự đoán một máy":

    st.title("Dự đoán tình trạng một máy")

    st.write(
        "Chọn thiết bị đã đăng ký hoặc thực hiện dự đoán thử nghiệm "
        "với dữ liệu chưa gắn với thiết bị cụ thể."
    )

    # -------------------------------------------------
    # 1. CHỌN THIẾT BỊ
    # -------------------------------------------------

    available_machines = get_all_machines(
        include_inactive=False
    )

    machine_options = [None] + [
        machine["id"] for machine in available_machines
    ]

    selected_machine_id = st.selectbox(
        "Thiết bị cần dự đoán",
        options=machine_options,
        format_func=lambda machine_id: (
            "Dự đoán thử nghiệm (không gắn thiết bị)"
            if machine_id is None
            else next(
                (
                    f'{m["machine_code"]} - {m["machine_name"]}'
                    for m in available_machines
                    if m["id"] == machine_id
                ),
                str(machine_id)
            )
        ),
        key="prediction_machine_selector"
    )

    selected_machine = (
        get_machine_by_id(selected_machine_id)
        if selected_machine_id is not None
        else None
    )

    if selected_machine:
        machine_type = selected_machine["machine_type"]

        st.info(
            f"Thiết bị: **{selected_machine['machine_name']}** | "
            f"Mã: **{selected_machine['machine_code']}** | "
            f"Loại máy: **{machine_type}**"
        )
    else:
        machine_type = None
        st.info(
            "Kết quả dự đoán sẽ không được gắn với hồ sơ thiết bị."
        )

    st.divider()

    # -------------------------------------------------
    # 2. NHẬP THÔNG SỐ VẬN HÀNH
    # -------------------------------------------------

    st.subheader("Thông số vận hành")

    with st.form("single_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            if selected_machine is None:
                input_machine_type = st.selectbox(
                    "Loại máy",
                    ["L", "M", "H"],
                    help="L: Low, M: Medium, H: High"
                )
            else:
                input_machine_type = machine_type

                st.text_input(
                    "Loại máy",
                    value=machine_type,
                    disabled=True
                )

            air_temp = st.number_input(
                "Nhiệt độ không khí (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1
            )

            process_temp = st.number_input(
                "Nhiệt độ quy trình (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1
            )

        with col2:

            rotational_speed = st.number_input(
                "Tốc độ quay (rpm)",
                min_value=100.0,
                max_value=5000.0,
                value=1500.0,
                step=10.0
            )

            torque = st.number_input(
                "Mô-men xoắn (Nm)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.1
            )

            tool_wear = st.number_input(
                "Độ mài mòn dụng cụ (phút)",
                min_value=0.0,
                max_value=300.0,
                value=100.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "Dự đoán tình trạng máy",
            width="stretch"
        )

    # -------------------------------------------------
    # 3. DỰ ĐOÁN VÀ LƯU DATABASE
    # -------------------------------------------------

    if submitted:

        input_data = pd.DataFrame([{
            "Type": input_machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        try:
            predictions, probabilities = run_prediction(
                model,
                input_data,
                PREDICTION_THRESHOLD
            )

            prediction = int(predictions[0])
            probability = float(probabilities[0])

            # Lưu dữ liệu dự đoán vào SQLite
            prediction_id = save_prediction(
                machine_id=selected_machine_id,
                input_data=input_data.iloc[0].to_dict(),
                probability=probability,
                prediction=prediction,
                threshold=PREDICTION_THRESHOLD,
                source="manual",
                model_name="Random Forest"
            )

            st.session_state["last_prediction_id"] = prediction_id

            st.divider()

            show_prediction_result(
                prediction,
                probability
            )

            st.success(
                f"Đã lưu kết quả dự đoán vào lịch sử "
                f"(ID: {prediction_id})."
            )

            show_local_explanation(input_data)

        except Exception as error:
            st.error(
                f"Không thể thực hiện hoặc lưu dự đoán: {error}"
            )

    # -------------------------------------------------
    # 4. LỊCH SỬ DỰ ĐOÁN
    # -------------------------------------------------

    st.divider()
    st.subheader("Lịch sử dự đoán")

    history = get_prediction_history(
        machine_id=selected_machine_id,
        limit=20
    )

    if history:

        history_data = []

        for record in history:
            history_data.append({
                "Thời gian": record["created_at"],
                "Mã thiết bị": (
                    record["machine_code"] or "Không gắn thiết bị"
                ),
                "Xác suất hỏng (%)": round(
                    record["failure_probability"] * 100,
                    2
                ),
                "Kết quả": (
                    "Có nguy cơ hỏng"
                    if record["predicted_failure"] == 1
                    else "Bình thường"
                ),
                "Threshold": record["threshold"],
                "Nguồn": record["source"]
            })

        st.dataframe(
            pd.DataFrame(history_data),
            width="stretch",
            hide_index=True
        )

        st.caption(
            f"Hiển thị tối đa 20 lần dự đoán gần nhất."
        )

    else:
        st.info(
            "Chưa có lịch sử dự đoán cho thiết bị này."
        )

elif page == "Dự đoán hàng loạt":

    st.title("Dự đoán hàng loạt")

    st.write(
        "Tải lên tệp CSV chứa thông số vận hành "
        "của nhiều máy để dự đoán cùng lúc."
    )

    st.info(
        "Tệp CSV cần có đủ 6 cột: Type, Air temperature [K], "
        "Process temperature [K], Rotational speed [rpm], "
        "Torque [Nm], Tool wear [min]."
    )

    template = pd.DataFrame(columns=FEATURES)

    st.download_button(
        "⬇️ Tải file CSV mẫu",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="machine_input_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Chọn tệp CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(uploaded_file)

            is_valid, validated_data = validate_input_data(
                input_df
            )

            if not is_valid:
                st.error(validated_data)

            else:
                predictions, probabilities = run_prediction(model, validated_data, PREDICTION_THRESHOLD)

                result_df = input_df.copy()

                result_df["Predicted Failure"] = predictions
                result_df["Failure Probability (%)"] = (
                    probabilities * 100
                ).round(2)

                result_df["Trạng thái"] = [
                    "Có nguy cơ hỏng" if value == 1 else "Bình thường"
                    for value in predictions
                ]
                result_df["Ngưỡng phân loại"] = PREDICTION_THRESHOLD

                total = len(result_df)
                failure_count = int((predictions == 1).sum())
                normal_count = total - failure_count

                st.success(
                    f"Đã dự đoán thành công {total} bản ghi."
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Tổng số máy", total)
                col2.metric("Bình thường", normal_count)
                col3.metric("Có nguy cơ hỏng", failure_count)

                st.subheader("Kết quả dự đoán")

                st.dataframe(
                    result_df,
                    width="stretch",
                    hide_index=True
                )

                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "⬇️ Tải kết quả dự đoán",
                    data=csv_data,
                    file_name="machine_predictions.csv",
                    mime="text/csv",
                    width="stretch"
                )

        except Exception as error:
            st.error(
                f"Không thể đọc hoặc xử lý tệp CSV: {error}"
            )


# =====================================================
# 8. SO SÁNH MÔ HÌNH
# =====================================================

elif page == "So sánh mô hình":

    st.title("Đánh giá và so sánh mô hình")

    st.write(
        "So sánh hiệu suất của năm thuật toán "
        "trên cùng tập dữ liệu kiểm thử."
    )

    display_comparison = comparison.copy()

    display_comparison["Model"] = (
        display_comparison["Model"].map(MODEL_LABELS)
        .fillna(display_comparison["Model"])
    )

    st.subheader("Bảng kết quả đánh giá")

    st.dataframe(
        display_comparison.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader("Biểu đồ so sánh")

    metric_options = {
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1-score": "F1-score",
        "ROC-AUC": "ROC-AUC"
    }

    selected_metric = st.selectbox(
        "Chọn chỉ số đánh giá",
        list(metric_options.keys())
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    chart_data = comparison.sort_values(
        by=selected_metric,
        ascending=True
    )

    chart_labels = [
        MODEL_LABELS.get(name, name)
        for name in chart_data["Model"]
    ]

    bars = ax.barh(
        chart_labels,
        chart_data[selected_metric]
    )

    ax.set_xlabel(selected_metric)
    ax.set_xlim(0, 1)
    ax.set_title(f"So sánh mô hình theo {selected_metric}")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center"
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    best_row = comparison.loc[
        comparison["F1-score"].idxmax()
    ]

    best_name = MODEL_LABELS.get(
        best_row["Model"],
        best_row["Model"]
    )

    st.success(
        f"Mô hình có F1-score cao nhất: {best_name} "
        f"({best_row['F1-score']:.4f})"
    )

    st.divider()
    st.subheader("Mức độ quan trọng của đặc trưng")
    st.caption(
        "Feature Importance của Random Forest cho biết mức độ đóng góp "
        "tổng thể của các đặc trưng trong quá trình xây dựng cây."
    )
    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importances = model.named_steps["classifier"].feature_importances_
        importance_df = pd.DataFrame({
            "Đặc trưng": feature_names,
            "Mức độ quan trọng": importances,
        }).sort_values("Mức độ quan trọng", ascending=True)
        label_map = {
            "num__Air temperature [K]": "Nhiệt độ không khí",
            "num__Process temperature [K]": "Nhiệt độ quy trình",
            "num__Rotational speed [rpm]": "Tốc độ quay",
            "num__Torque [Nm]": "Mô-men xoắn",
            "num__Tool wear [min]": "Độ mài mòn dụng cụ",
            "cat__Type_H": "Loại máy H",
            "cat__Type_L": "Loại máy L",
            "cat__Type_M": "Loại máy M",
        }
        importance_df["Đặc trưng"] = importance_df["Đặc trưng"].map(label_map).fillna(importance_df["Đặc trưng"])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(importance_df["Đặc trưng"], importance_df["Mức độ quan trọng"])
        ax.set_xlabel("Feature Importance")
        ax.set_title("Tầm quan trọng của các đặc trưng - Random Forest")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as error:
        st.warning(f"Không thể hiển thị Feature Importance: {error}")

    st.divider()

    st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")

    try:
        cm_df = load_confusion_matrix()

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="YlOrBr",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title("Confusion Matrix - Random Forest")
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thực tế")

        ax.set_xticklabels(
            ["Bình thường", "Hỏng hóc"]
        )
        ax.set_yticklabels(
            ["Bình thường", "Hỏng hóc"],
            rotation=0
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as error:
        st.warning(
            f"Không thể hiển thị Confusion Matrix: {error}"
        )

    st.divider()
    st.subheader("Đánh giá tác động của điều chỉnh ngưỡng")
    st.caption(
        "Ngưỡng được lựa chọn bằng F1-score trên xác suất Out-of-Fold "
        "của tập huấn luyện; tập kiểm thử chỉ dùng để đánh giá cuối cùng."
    )
    st.dataframe(
        threshold_results.style.format({
            "Threshold": "{:.2f}", "Accuracy": "{:.4f}",
            "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"
        }),
        width="stretch",
        hide_index=True
    )
    st.info(
        f"Ứng dụng đang sử dụng threshold = {PREDICTION_THRESHOLD:.2f}. "
        "Việc giảm ngưỡng ưu tiên phát hiện nhiều trường hợp hỏng hơn, "
        "nhưng có thể làm tăng cảnh báo nhầm."
    )


# =====================================================
# 9. CHÂN TRANG
# =====================================================

st.divider()

st.caption(
    "Machine Failure Prediction | "
    "Ứng dụng học máy dự đoán nguy cơ hỏng hóc máy móc"
)
