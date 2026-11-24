"""Shared inference and input-validation utilities for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURES = FEATURES[1:]

INPUT_BOUNDS = {
    "Air temperature [K]": (250, 350),
    "Process temperature [K]": (250, 400),
    "Rotational speed [rpm]": (100, 5000),
    "Torque [Nm]": (0, 100),
    "Tool wear [min]": (0, 300),
}


def predict_with_threshold(model, input_df: pd.DataFrame, threshold: float):
    """Predict positive-class probabilities and apply the configured threshold."""
    if not 0 < float(threshold) < 1:
        raise ValueError("Threshold phải nằm trong khoảng (0, 1).")
    probabilities = model.predict_proba(input_df)[:, 1]
    predictions = (probabilities >= float(threshold)).astype(int)
    return predictions, probabilities


def validate_input_data(input_df: pd.DataFrame):
    """Validate required columns, types, finite values and supported input ranges."""
    missing_columns = [column for column in FEATURES if column not in input_df.columns]
    if missing_columns:
        return False, "File thiếu các cột bắt buộc: " + ", ".join(missing_columns)
    if input_df.empty:
        return False, "File CSV không có bản ghi nào."

    data = input_df[FEATURES].copy()
    if data.isnull().any().any():
        return False, "Dữ liệu có ô trống. Vui lòng bổ sung trước khi dự đoán."

    data["Type"] = data["Type"].astype(str).str.strip().str.upper()
    if (~data["Type"].isin(["L", "M", "H"])).any():
        return False, "Cột Type chỉ chấp nhận các giá trị L, M hoặc H."

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[NUMERIC_FEATURES].isnull().any().any():
        return False, "Một hoặc nhiều cột số chứa dữ liệu không hợp lệ."

    numeric_values = data[NUMERIC_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        return False, "Dữ liệu chứa giá trị vô hạn hoặc không hợp lệ."

    for column, (lower, upper) in INPUT_BOUNDS.items():
        invalid = ~data[column].between(lower, upper)
        if invalid.any():
            # CSV is read with a zero-based RangeIndex; add two for header + 1-based rows.
            rows = (invalid[invalid].index + 2).tolist()[:5]
            return False, (
                f"Cột '{column}' có giá trị ngoài khoảng cho phép "
                f"[{lower}, {upper}]. Kiểm tra dòng CSV: {rows}."
            )

    return True, data
